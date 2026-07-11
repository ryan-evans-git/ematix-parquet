//! Oracle for the v2 tagged-rowset sidecar format.
//!
//! v1 stored one page-BITMAP per (value, page) — `num_page_values/8`
//! bytes per distinct value, which explodes on high-cardinality keys
//! (a real 3 GB TPC-H lineitem part → 47 GB rowset column, over
//! snappy's 4 GiB single-buffer cap at build time). v2 tags every
//! rowset with a 1-byte discriminator and stores SPARSE row-id lists
//! when they are smaller than the bitmap:
//!
//!   `[0x00][bitmap …]`                                — dense pages
//!   `[0x01][u32 num_page_values][u32 count][u32 rows…]` — sparse
//!
//! The public `IndexHit` contract is unchanged: hits always carry a
//! packed bitmap; sparse rowsets are normalized at hit time. The
//! manifest KV key is bumped (`ematix_index_manifest_v2`) so a v1
//! reader refuses v2 sidecars loudly (its documented behavior for a
//! missing/unknown manifest) instead of misreading tagged bytes as a
//! bitmap. The v2 reader still reads v1 sidecars.

use ematix_parquet_codec::index::{
    compute_source_fingerprint, IndexBuilder, IndexEntry, IndexKind, IndexManifest, Key,
    ParquetIndex, PhysicalType,
};
use ematix_parquet_codec::read::{read_column_byte_array, read_column_i64};
use ematix_parquet_codec::write::{
    write_i64_column_to_path, write_table_with_options_to_path, ColumnData, WriteOptions,
};
use ematix_parquet_io::ParquetFile;

const V1_KEY: &str = "ematix_index_manifest_v1";
const V2_KEY: &str = "ematix_index_manifest_v2";

fn baseline_eq(source: &ParquetFile, col: usize, key: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let md = source.metadata().unwrap();
    for rg in 0..md.row_groups.len() {
        let values = read_column_i64(source, rg, col).unwrap();
        out.extend(values.iter().copied().filter(|v| *v == key));
    }
    out
}

fn build(
    dir: &std::path::Path,
    prefix: &str,
    values: &[i64],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = dir.join(format!("{prefix}.parquet"));
    let idx = dir.join(format!("{prefix}.parquet.idx"));
    write_i64_column_to_path(&src, "v", values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .expect("build sidecar");
    (src, idx)
}

/// KV keys present in a sidecar's footer.
fn kv_keys(idx_path: &std::path::Path) -> Vec<String> {
    let f = ParquetFile::open(idx_path).unwrap();
    let md = f.metadata().unwrap();
    md.key_value_metadata
        .iter()
        .flatten()
        .map(|kv| String::from_utf8_lossy(kv.key).into_owned())
        .collect()
}

/// Every rowset byte-array of the sidecar's sorted index body
/// (column 3 of row group 0).
fn rowsets(idx_path: &std::path::Path) -> Vec<Vec<u8>> {
    let f = ParquetFile::open(idx_path).unwrap();
    read_column_byte_array(&f, 0, 3).unwrap()
}

// ---- v2 write shape -------------------------------------------------

/// Unique keys (the lineitem-orderkey worst case): every rowset must
/// pick the sparse form, and the sidecar must stay small — the v1
/// bitmap floor for this fixture is > num_values * (page_rows/8)
/// bytes, the sparse form ~13 bytes per value.
#[test]
fn unique_keys_choose_sparse_rowsets() {
    let dir = tempfile::tempdir().unwrap();
    let n: usize = 100_000;
    let values: Vec<i64> = (0..n as i64).collect();
    let (src, idx) = build(dir.path(), "unique", &values);

    let rs = rowsets(&idx);
    assert_eq!(rs.len(), n, "one rowset per distinct (value, page)");
    assert!(
        rs.iter().all(|r| r.first() == Some(&0x01)),
        "unique keys must serialize as sparse (tag 0x01)"
    );
    // Sparse: tag + npv + count + 1 row id = 13 bytes.
    assert!(
        rs.iter().all(|r| r.len() == 13),
        "single-row sparse rowset is exactly 13 bytes"
    );

    // Parity: indexed lookup == full scan, across the domain + absent.
    let source = ParquetFile::open(&src).unwrap();
    let pidx = ParquetIndex::open(&idx, &source).unwrap();
    for key in [0i64, 1, 4_242, (n - 1) as i64, n as i64 + 5] {
        let indexed = pidx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        assert_eq!(indexed, baseline_eq(&source, 0, key), "key={key}");
    }
}

/// A page-dominating value (heavy duplication) keeps the dense bitmap
/// form — sparse would be larger there.
#[test]
fn dense_pages_keep_bitmap_rowsets() {
    let dir = tempfile::tempdir().unwrap();
    // One value fills everything: bitmap = page_rows/8 bytes; sparse
    // would be 9 + 4*page_rows.
    let values: Vec<i64> = vec![7; 50_000];
    let (src, idx) = build(dir.path(), "dense", &values);

    let rs = rowsets(&idx);
    assert!(!rs.is_empty());
    assert!(
        rs.iter().all(|r| r.first() == Some(&0x00)),
        "page-dominating values must stay bitmap (tag 0x00)"
    );

    let source = ParquetFile::open(&src).unwrap();
    let pidx = ParquetIndex::open(&idx, &source).unwrap();
    let indexed = pidx
        .read_column_i64_where_eq(&source, "idx_v", 7, 0)
        .unwrap();
    assert_eq!(indexed.len(), 50_000);
    assert_eq!(indexed, baseline_eq(&source, 0, 7));
}

/// Mixed distribution round-trips through eq AND range lookups —
/// both rowset forms inside one index.
#[test]
fn mixed_density_eq_and_range_parity() {
    let dir = tempfile::tempdir().unwrap();
    // Value 0 dominates; values 1..=2000 appear once each.
    let mut values: Vec<i64> = vec![0; 30_000];
    values.extend(1..=2_000i64);
    let (src, idx) = build(dir.path(), "mixed", &values);

    let rs = rowsets(&idx);
    let sparse = rs.iter().filter(|r| r.first() == Some(&0x01)).count();
    let bitmap = rs.iter().filter(|r| r.first() == Some(&0x00)).count();
    assert!(
        sparse > 0 && bitmap > 0,
        "expected both forms, got sparse={sparse} bitmap={bitmap}"
    );

    let source = ParquetFile::open(&src).unwrap();
    let pidx = ParquetIndex::open(&idx, &source).unwrap();
    for key in [0i64, 1, 1_000, 2_000, 2_001] {
        let indexed = pidx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        assert_eq!(indexed, baseline_eq(&source, 0, key), "eq key={key}");
    }
    // Range across the sparse tail.
    let ranged = pidx
        .read_column_i64_where_range(&source, "idx_v", 100, 110, 0)
        .unwrap();
    let mut expect: Vec<i64> = (100..=110).collect();
    expect.sort_unstable();
    let mut got = ranged.clone();
    got.sort_unstable();
    assert_eq!(got, expect, "range 100..=110");
}

// ---- version fencing ------------------------------------------------

/// v2 sidecars carry ONLY the v2 manifest key: a v1 reader sees no
/// v1 manifest and refuses (its documented fail-loud path), instead
/// of silently misreading tagged rowsets as bitmaps.
#[test]
fn v2_sidecar_has_only_v2_manifest_key() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..1_000i64).collect();
    let (_, idx) = build(dir.path(), "fence", &values);
    let keys = kv_keys(&idx);
    assert!(
        keys.iter().any(|k| k == V2_KEY),
        "v2 manifest key missing: {keys:?}"
    );
    assert!(
        !keys.iter().any(|k| k == V1_KEY),
        "v1 manifest key must NOT be present on a v2 sidecar: {keys:?}"
    );
}

/// A legacy v1 sidecar (raw bitmaps, v1 manifest key) still reads —
/// constructed here exactly as the 0.17.0 writer laid it out.
#[test]
fn legacy_v1_sidecar_still_reads() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("legacy.parquet");
    let idx = dir.path().join("legacy.parquet.idx");
    let values: Vec<i64> = vec![10, 10, 20, 30, 30, 30, 40, 50, 50, 60];
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();

    // v1 body: one bitmap per distinct value (single RG, single page
    // fixture), bitmap length = ceil(10/8) = 2 bytes, raw (no tag).
    let distinct: Vec<i64> = vec![10, 20, 30, 40, 50, 60];
    let mut col_rowsets: Vec<Vec<u8>> = Vec::new();
    for d in &distinct {
        let mut bm = vec![0u8; values.len().div_ceil(8)];
        for (i, v) in values.iter().enumerate() {
            if v == d {
                bm[i / 8] |= 1 << (i % 8);
            }
        }
        col_rowsets.push(bm);
    }
    let col_rgs: Vec<i32> = vec![0; distinct.len()];
    let col_pages: Vec<i32> = vec![0; distinct.len()];
    let manifest_json = IndexManifest {
        source_fingerprint: compute_source_fingerprint(&source).unwrap(),
        indexes: vec![IndexEntry {
            name: "idx_v".to_owned(),
            kind: IndexKind::Sorted {
                source_column: "v".to_owned(),
                physical_type: PhysicalType::Int64,
            },
            sidecar_row_group: 0,
        }],
    }
    .to_json();
    let rowset_slices: Vec<&[u8]> = col_rowsets.iter().map(|v| v.as_slice()).collect();
    let kvs = [(V1_KEY, manifest_json.as_str())];
    let cols: &[(&str, ColumnData<'_>)] = &[
        ("value", ColumnData::I64(&distinct)),
        ("target_rg", ColumnData::I32(&col_rgs)),
        ("target_page", ColumnData::I32(&col_pages)),
        ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
    ];
    let opts = WriteOptions {
        kv_metadata: Some(&kvs),
        ..WriteOptions::default()
    };
    write_table_with_options_to_path(&idx, cols, &opts).unwrap();

    let pidx = ParquetIndex::open(&idx, &source).expect("v2 reader must open v1 sidecars");
    for key in [10i64, 30, 50, 60, 99] {
        let indexed = pidx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        assert_eq!(indexed, baseline_eq(&source, 0, key), "v1 key={key}");
    }
    // Raw hits carry normalized bitmaps regardless of stored form.
    let hits = pidx.lookup_eq("idx_v", &Key::I64(30)).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].rowset.len(), values.len().div_ceil(8));
}

/// The build that motivated all of this: high-cardinality keys build
/// without tripping any compression buffer cap and produce an index
/// a small multiple of the value column's size — NOT page_rows/8
/// bytes per value.
#[test]
fn high_cardinality_build_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let n: usize = 2_000_000;
    // Clustered duplicates (the real l_orderkey shape, ~4 rows/key).
    let values: Vec<i64> = (0..n).map(|i| (i / 4) as i64).collect();
    let (src, idx) = build(dir.path(), "highcard", &values);
    let idx_bytes = std::fs::metadata(&idx).unwrap().len();
    // 500k distinct values; sparse rowsets ≈ 25 B/value plus parquet
    // overhead. Anything within 64 MB proves the encoding is sparse;
    // the v1 bitmap form for this fixture would be orders of
    // magnitude larger.
    assert!(
        idx_bytes < 64 * 1024 * 1024,
        "index unexpectedly large: {idx_bytes} bytes"
    );
    let source = ParquetFile::open(&src).unwrap();
    let pidx = ParquetIndex::open(&idx, &source).unwrap();
    for key in [0i64, 123_456, 499_999] {
        let indexed = pidx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        assert_eq!(indexed.len(), 4, "clustered key {key} has 4 rows");
        assert_eq!(indexed, baseline_eq(&source, 0, key), "key={key}");
    }
}
