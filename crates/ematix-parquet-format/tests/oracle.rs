//! Correctness oracle: decode the same Parquet footer bytes via our
//! `read_file_metadata` and via parquet-rs's `SerializedFileReader`,
//! then assert every field we can compare matches.
//!
//! This is the test that turns the format crate from "passes our own
//! hand-built compact bytes" into "passes real-world Parquet files
//! produced by the reference implementation."
//!
//! Data source: TPC-H lineitem.parquet at SF=1. We look up the file
//! via the `TPCH_DATA_DIR` env var first, then fall back to the
//! sibling `ematix-flow` checkout where it lives at
//! `examples/tpch/data/sf1/`. If neither resolves we skip — the test
//! is a sanity gate, not a CI dependency.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::read_file_metadata;
use ematix_parquet_format::types::{CompressionCodec, ParquetType};
use parquet::basic::{Compression as PrCompression, Type as PrType};
use parquet::file::reader::{FileReader, SerializedFileReader};

fn data_path() -> Option<PathBuf> {
    if let Ok(s) = std::env::var("TPCH_DATA_DIR") {
        let p = PathBuf::from(s).join("lineitem.parquet");
        if p.exists() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()? // crates/
        .parent()? // ematix-parquet/
        .parent()? // RustroverProjects/
        .join("ematix-flow/examples/tpch/data/sf1/lineitem.parquet");
    p.exists().then_some(p)
}

fn read_footer_bytes(path: &PathBuf) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let n = f.metadata()?.len();
    f.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    f.read_exact(&mut tail)?;
    assert_eq!(&tail[4..], b"PAR1", "missing PAR1 magic at end of file");
    let footer_len = u32::from_le_bytes(tail[0..4].try_into().unwrap()) as u64;
    let footer_start = n - 8 - footer_len;
    f.seek(SeekFrom::Start(footer_start))?;
    let mut buf = vec![0u8; footer_len as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Map our ParquetType to parquet-rs's physical Type. Going through a
/// shared mapping keeps the test from leaking either side's naming
/// quirks ("Int64" vs "INT64").
fn map_parquet_type(t: ParquetType) -> PrType {
    match t {
        ParquetType::Boolean => PrType::BOOLEAN,
        ParquetType::Int32 => PrType::INT32,
        ParquetType::Int64 => PrType::INT64,
        ParquetType::Int96 => PrType::INT96,
        ParquetType::Float => PrType::FLOAT,
        ParquetType::Double => PrType::DOUBLE,
        ParquetType::ByteArray => PrType::BYTE_ARRAY,
        ParquetType::FixedLenByteArray => PrType::FIXED_LEN_BYTE_ARRAY,
    }
}

fn map_codec(c: CompressionCodec) -> PrCompression {
    // parquet-rs's Compression has parameter variants (Brotli{level},
    // Gzip{level}, Zstd{level}); we only need to match the kind, not
    // the level. Compare via discriminant name through Debug.
    let _ = c; // kept for symmetry; comparison is done by string at call sites.
    PrCompression::UNCOMPRESSED
}

#[test]
fn lineitem_footer_matches_parquet_rs() {
    let Some(path) = data_path() else {
        eprintln!("SKIP: lineitem.parquet not found (set TPCH_DATA_DIR or place under ../ematix-flow/examples/tpch/data/sf1)");
        return;
    };
    eprintln!("oracle: comparing decoders against {}", path.display());

    let footer = read_footer_bytes(&path).expect("read footer");
    eprintln!("        footer is {} bytes", footer.len());

    // Our decoder
    let mut cur = Cursor::new(&footer);
    let our_md = read_file_metadata(&mut cur).expect("our decoder failed on real footer");

    // parquet-rs decoder
    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let pr_md = pr_reader.metadata();
    let pr_fm = pr_md.file_metadata();

    // ---- Top-level fields ------------------------------------------------
    assert_eq!(our_md.version, pr_fm.version(), "version");
    assert_eq!(our_md.num_rows, pr_fm.num_rows(), "num_rows");

    // created_by may or may not be set; if either is set, both must be.
    let ours_created_by = our_md
        .created_by
        .map(|b| std::str::from_utf8(b).expect("created_by is valid UTF-8"));
    assert_eq!(ours_created_by, pr_fm.created_by(), "created_by");

    // ---- Schema: depth-first list count ---------------------------------
    // For flat schemas (no Struct/List/Map nesting), our flat list size
    // is exactly `1 root + N leaves`. parquet-rs exposes the leaf count.
    let pr_leaves = pr_fm.schema_descr().num_columns();
    assert_eq!(
        our_md.schema.len(),
        1 + pr_leaves,
        "schema flat-list size should be 1 (root) + {pr_leaves} (leaves)"
    );

    // Spot check: root has the right child count.
    assert_eq!(
        our_md.schema[0].num_children,
        Some(pr_leaves as i32),
        "root num_children"
    );

    // Spot check: every non-root schema element has a column_type set
    // (since this file is flat).
    for (i, se) in our_md.schema.iter().enumerate().skip(1) {
        assert!(
            se.column_type.is_some(),
            "schema[{i}] is a non-root leaf and must carry a column_type"
        );
        assert!(!se.name.is_empty(), "schema[{i}] has empty name");
    }

    // ---- Row groups ------------------------------------------------------
    assert_eq!(
        our_md.row_groups.len(),
        pr_md.num_row_groups(),
        "num row groups"
    );

    for (i, (ours, theirs)) in our_md
        .row_groups
        .iter()
        .zip(pr_md.row_groups())
        .enumerate()
    {
        assert_eq!(ours.num_rows, theirs.num_rows(), "rg {i} num_rows");
        assert_eq!(
            ours.total_byte_size,
            theirs.total_byte_size(),
            "rg {i} total_byte_size"
        );
        assert_eq!(
            ours.columns.len(),
            theirs.num_columns(),
            "rg {i} column count"
        );

        for (j, (our_col, their_col)) in
            ours.columns.iter().zip(theirs.columns()).enumerate()
        {
            let our_meta = our_col
                .meta_data
                .as_ref()
                .unwrap_or_else(|| panic!("rg {i} col {j} missing inline ColumnMetaData"));

            // Physical type
            assert_eq!(
                map_parquet_type(our_meta.column_type),
                their_col.column_type(),
                "rg {i} col {j} physical type"
            );

            // Codec: parquet-rs's Compression has level-parametric
            // variants (Brotli{level}, Gzip{level}, Zstd{level}).
            // Compare the kind name only.
            let theirs_codec_name = format!("{:?}", their_col.compression());
            let ours_codec_name = format!("{:?}", our_meta.codec).to_uppercase();
            let theirs_kind = theirs_codec_name
                .split(|c: char| !c.is_ascii_alphanumeric())
                .next()
                .unwrap();
            let ours_kind = ours_codec_name
                .split(|c: char| !c.is_ascii_alphanumeric())
                .next()
                .unwrap();
            assert!(
                theirs_kind.eq_ignore_ascii_case(ours_kind),
                "rg {i} col {j} codec: ours={ours_codec_name} theirs={theirs_codec_name}"
            );
            let _ = map_codec; // keep the helper alive for future direct comparison

            assert_eq!(
                our_meta.num_values,
                their_col.num_values(),
                "rg {i} col {j} num_values"
            );
            assert_eq!(
                our_meta.total_compressed_size,
                their_col.compressed_size(),
                "rg {i} col {j} total_compressed_size"
            );
            assert_eq!(
                our_meta.total_uncompressed_size,
                their_col.uncompressed_size(),
                "rg {i} col {j} total_uncompressed_size"
            );
        }
    }

    eprintln!(
        "PASS: {} row groups × {} columns matched parquet-rs field-by-field",
        our_md.row_groups.len(),
        our_md.row_groups.first().map(|rg| rg.columns.len()).unwrap_or(0)
    );
}
