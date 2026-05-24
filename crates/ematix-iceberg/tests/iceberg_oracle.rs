//! Π.22c end-to-end oracle for the dataset layer.
//!
//! Drives the planner-side flow against a real Iceberg table with
//! real manifest avro files on disk:
//!
//! 1. **Create a Table** via `MemoryCatalog` + `TableCreation` (so
//!    location, partition spec, schema, etc. all match what a real
//!    Iceberg producer would emit).
//! 2. **Write manifest files** holding our `DataFile` entries — each
//!    with [`EmatixDataFileExtension`] JSON in its `key_metadata` —
//!    via the public `ManifestWriterBuilder` and `ManifestListWriter`.
//! 3. **Build a new TableMetadata** with a snapshot pointing at the
//!    just-written manifest list, using `set_branch_snapshot` on the
//!    metadata builder.
//! 4. **Run** [`collect_data_files`] →
//!    [`prune_data_files_eq`] / [`prune_data_files_range`] →
//!    [`pair_with_extensions`] against the resulting Table.
//!
//! ## Why not `Transaction::commit`
//!
//! `iceberg = "0.6"`'s `MemoryCatalog::update_table` is stubbed and
//! returns `FeatureUnsupported`; the canonical
//! `tx.commit(&catalog).await` flow can't be used in-process without
//! a Sql/Rest catalog. We sidestep that by writing the same manifest
//! avro files Transaction would have written, and then promoting the
//! snapshot via the public `TableMetadataBuilder::set_branch_snapshot`.
//! This faithfully exercises [`collect_data_files`] against real
//! Avro-encoded manifest entries.
//!
//! ## What this oracle does NOT cover
//!
//! Opening the sidecar parquet + masked-decoding the source. The
//! codec-side path is independently proven by every
//! `sidecar_*_oracle.rs` in `ematix-parquet-codec/tests/`. Folding
//! both into one oracle would triple the fixture cost without adding
//! coverage of any cross-layer interaction.

#![cfg(feature = "iceberg")]

use std::collections::HashMap;
use std::sync::Arc;

use ematix_iceberg::iceberg_rs::{
    collect_data_files, encode_key_metadata, extract_extension, pair_with_extensions,
    prune_data_files_eq, prune_data_files_range,
};
use ematix_iceberg::{EmatixDataFileExtension, IndexSummary, SummaryKey};
use ematix_parquet_codec::index::Key;
use iceberg::io::FileIOBuilder;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, ManifestListWriter,
    ManifestWriterBuilder, NestedField, Operation, PrimitiveType, Schema, Snapshot, Struct,
    Summary, Type, MAIN_BRANCH,
};
use iceberg::table::Table;
use iceberg::Catalog;
use iceberg::MemoryCatalog;
use iceberg::{NamespaceIdent, TableCreation};
use tempfile::TempDir;

/// One Iceberg fixture: a temp warehouse + a Table populated with N
/// data files (each carrying our extension), via a real manifest +
/// manifest-list write. The `_warehouse` guard must outlive the
/// table — drop it and the on-disk manifests vanish.
struct Fixture {
    _warehouse: TempDir,
    table: Table,
}

impl Fixture {
    async fn new(extensions: &[(&str, EmatixDataFileExtension)]) -> Self {
        let warehouse = TempDir::new().expect("temp warehouse");
        let file_io = FileIOBuilder::new_fs_io().build().expect("file io");
        let catalog = MemoryCatalog::new(
            file_io.clone(),
            Some(warehouse.path().to_str().unwrap().to_string()),
        );

        let ns = NamespaceIdent::new("db".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");

        let schema = Schema::builder()
            .with_fields(vec![Arc::new(NestedField::required(
                1,
                "v",
                Type::Primitive(PrimitiveType::Long),
            ))])
            .build()
            .unwrap();

        let creation = TableCreation::builder()
            .name("t".into())
            .schema(schema.clone())
            .build();
        let table = catalog
            .create_table(&ns, creation)
            .await
            .expect("create_table");

        // Build typed data files. file_path is anchored under the
        // table's location so resolve_sidecar_uri produces stable
        // paths.
        let table_location = table.metadata().location().to_string();
        let partition_spec = table.metadata().default_partition_spec().as_ref().clone();
        let data_files: Vec<DataFile> = extensions
            .iter()
            .map(|(relative_name, ext)| {
                let mut b = DataFileBuilder::default();
                b.content(DataContentType::Data)
                    .file_path(format!("{table_location}/data/{relative_name}"))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(100)
                    .record_count(10)
                    .partition_spec_id(table.metadata().default_partition_spec_id())
                    .partition(Struct::empty())
                    .key_metadata(Some(encode_key_metadata(ext)));
                b.build().expect("data file builder")
            })
            .collect();

        // 1. Write a manifest file containing one entry per data file.
        let snapshot_id: i64 = 1;
        let sequence_number: i64 = 1;
        let manifest_file_path = format!("{table_location}/metadata/manifest-1.avro");
        let manifest_output = file_io
            .new_output(&manifest_file_path)
            .expect("manifest output");
        let mut manifest_writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(snapshot_id),
            None,
            Arc::new(schema.clone()),
            partition_spec,
        )
        .build_v2_data();
        for df in &data_files {
            manifest_writer
                .add_file(df.clone(), sequence_number)
                .expect("add_file");
        }
        let manifest_file = manifest_writer
            .write_manifest_file()
            .await
            .expect("write_manifest_file");

        // 2. Write a manifest list pointing at the manifest file.
        let manifest_list_path = format!("{table_location}/metadata/snap-1.avro");
        let manifest_list_output = file_io
            .new_output(&manifest_list_path)
            .expect("manifest list output");
        let mut mlw =
            ManifestListWriter::v2(manifest_list_output, snapshot_id, None, sequence_number);
        mlw.add_manifests(vec![manifest_file].into_iter())
            .expect("add_manifests");
        mlw.close().await.expect("close manifest list");

        // 3. Build a Snapshot pointing at the manifest list. Timestamp
        //    must be ≥ the table's last_updated_ms — `set_branch_snapshot`
        //    rejects backdated snapshots.
        let snapshot_timestamp_ms = table.metadata().last_updated_ms() + 1;
        let snapshot = Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(None)
            .with_sequence_number(sequence_number)
            .with_timestamp_ms(snapshot_timestamp_ms)
            .with_manifest_list(manifest_list_path)
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .build();

        // 4. Promote the snapshot on the table's main branch via the
        //    metadata builder.
        let new_metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_branch_snapshot(snapshot, MAIN_BRANCH)
            .expect("set_branch_snapshot")
            .build()
            .expect("metadata build")
            .metadata;

        // 5. Build a Table with the new metadata. (Table::with_metadata
        //    is pub(crate); reconstructing via the builder is the
        //    public path.)
        let updated_table = iceberg::table::Table::builder()
            .file_io(file_io)
            .metadata(new_metadata)
            .identifier(table.identifier().clone())
            .metadata_location(table.metadata_location().unwrap_or_default().to_string())
            .build()
            .expect("rebuild table");

        Self {
            _warehouse: warehouse,
            table: updated_table,
        }
    }
}

fn ext_with_i64_range(sidecar: &str, name: &str, lo: i64, hi: i64) -> EmatixDataFileExtension {
    EmatixDataFileExtension {
        sidecar_relative_path: sidecar.into(),
        summaries: vec![
            IndexSummary::new(name).with_range(SummaryKey::I64(lo), SummaryKey::I64(hi))
        ],
    }
}

#[tokio::test]
async fn collect_walks_appended_data_files_with_extensions() {
    let ext1 = ext_with_i64_range("sidecar.idx", "idx_x", 0, 99);
    let ext2 = ext_with_i64_range("sidecar.idx", "idx_x", 100, 199);
    let ext3 = ext_with_i64_range("sidecar.idx", "idx_x", 200, 299);
    let fixture = Fixture::new(&[
        ("f1.parquet", ext1.clone()),
        ("f2.parquet", ext2.clone()),
        ("f3.parquet", ext3.clone()),
    ])
    .await;

    let files = collect_data_files(&fixture.table).await.expect("collect");
    assert_eq!(files.len(), 3);

    // Each file's extension survived the Avro round-trip through
    // ManifestWriter / ManifestListWriter / load_manifest.
    let mut recovered_summaries = Vec::new();
    for df in &files {
        let ext = extract_extension(df)
            .expect("extract")
            .expect("ext present");
        assert_eq!(ext.sidecar_relative_path, "sidecar.idx");
        assert_eq!(ext.summaries.len(), 1);
        recovered_summaries.push(ext.summaries[0].clone());
    }
    recovered_summaries.sort_by_key(|s| match s.min_key {
        Some(SummaryKey::I64(v)) => v,
        _ => i64::MIN,
    });
    assert_eq!(recovered_summaries[0].min_key, Some(SummaryKey::I64(0)));
    assert_eq!(recovered_summaries[1].min_key, Some(SummaryKey::I64(100)));
    assert_eq!(recovered_summaries[2].min_key, Some(SummaryKey::I64(200)));
}

#[tokio::test]
async fn prune_eq_against_real_table_keeps_in_range_file() {
    let fixture = Fixture::new(&[
        (
            "f1.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 0, 99),
        ),
        (
            "f2.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 100, 199),
        ),
        (
            "f3.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 200, 299),
        ),
    ])
    .await;

    let files = collect_data_files(&fixture.table).await.unwrap();
    assert_eq!(files.len(), 3);

    // Query 150 — only f2.
    let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(150)).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(
        kept[0].file_path().ends_with("data/f2.parquet"),
        "got {}",
        kept[0].file_path()
    );

    // Query 1000 — nothing.
    let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(1000)).unwrap();
    assert!(kept.is_empty());

    // Boundary 99 — only f1.
    let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(99)).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(kept[0].file_path().ends_with("data/f1.parquet"));

    // Boundary 100 — only f2.
    let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(100)).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(kept[0].file_path().ends_with("data/f2.parquet"));
}

#[tokio::test]
async fn prune_range_against_real_table_keeps_overlapping_files() {
    let fixture = Fixture::new(&[
        (
            "f1.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 0, 99),
        ),
        (
            "f2.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 100, 199),
        ),
        (
            "f3.parquet",
            ext_with_i64_range("sidecar.idx", "idx_x", 200, 299),
        ),
    ])
    .await;

    let files = collect_data_files(&fixture.table).await.unwrap();

    // [50, 250] overlaps all three.
    let kept =
        prune_data_files_range(&files, "idx_x", Some(&Key::I64(50)), Some(&Key::I64(250))).unwrap();
    assert_eq!(kept.len(), 3);

    // [50, 150] overlaps f1 + f2.
    let kept =
        prune_data_files_range(&files, "idx_x", Some(&Key::I64(50)), Some(&Key::I64(150))).unwrap();
    assert_eq!(kept.len(), 2);

    // [-∞, 50] overlaps only f1.
    let kept = prune_data_files_range(&files, "idx_x", None, Some(&Key::I64(50))).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(kept[0].file_path().ends_with("data/f1.parquet"));

    // [250, +∞) overlaps only f3.
    let kept = prune_data_files_range(&files, "idx_x", Some(&Key::I64(250)), None).unwrap();
    assert_eq!(kept.len(), 1);
    assert!(kept[0].file_path().ends_with("data/f3.parquet"));
}

#[tokio::test]
async fn pair_resolves_sidecar_uris_against_real_data_paths() {
    let fixture = Fixture::new(&[
        (
            "f1.parquet",
            ext_with_i64_range("idx_x.idx", "idx_x", 0, 99),
        ),
        (
            "f2.parquet",
            ext_with_i64_range("idx_x.idx", "idx_x", 100, 199),
        ),
    ])
    .await;

    let files = collect_data_files(&fixture.table).await.unwrap();
    let candidates = pair_with_extensions(files).unwrap();
    assert_eq!(candidates.len(), 2);

    // Each candidate's sidecar URI lands in the same directory as
    // its data file, with the basename from the extension.
    for c in &candidates {
        let data_dir = c
            .data_file
            .file_path()
            .rsplit_once('/')
            .map(|(d, _)| d.to_string())
            .unwrap();
        assert_eq!(
            c.sidecar_uri,
            format!("{data_dir}/idx_x.idx"),
            "data file: {}",
            c.data_file.file_path()
        );
    }
}

#[tokio::test]
async fn collect_yields_empty_set_for_table_with_no_commits() {
    // Fixture with zero appended files — current snapshot still
    // exists (we always emit one in the fixture), but the manifest
    // contains no entries. The walker reports an empty Vec.
    let fixture = Fixture::new(&[]).await;
    let files = collect_data_files(&fixture.table).await.unwrap();
    assert!(files.is_empty());
}
