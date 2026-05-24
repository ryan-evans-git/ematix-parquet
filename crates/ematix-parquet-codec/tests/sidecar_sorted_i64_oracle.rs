//! Oracle for sorted INT64 sidecar indexes (Π.17b).
//!
//! The load-bearing correctness invariant: **`read_column_i64_where_eq`
//! returns exactly the rows a full-scan filter on the same predicate
//! would return**, for every key — present, absent, or at the edges of
//! the column's value distribution. Anything that breaks this property
//! breaks indexes in general; the tests below cover the cases most
//! likely to expose it.

use ematix_parquet_codec::index::{IndexBuilder, IndexHit, Key, ParquetIndex};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::write_i64_column_to_path;
use ematix_parquet_io::ParquetFile;

/// Full-scan baseline: open the source, decode the column, return all
/// rows where the value equals `key`. Order matches the source's
/// row-major order — which is what `read_column_i64_where_eq` returns
/// too (per-row-group, per-page traversal under the hood).
fn baseline_eq(source: &ParquetFile, col: usize, key: i64) -> Vec<i64> {
    let values = read_column_i64(source, 0, col).unwrap();
    values.iter().copied().filter(|v| *v == key).collect()
}

/// Convenience to build a small parquet, then build its sidecar.
fn build_source_and_sidecar(
    dir: &std::path::Path,
    name_prefix: &str,
    values: &[i64],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src_path = dir.join(format!("{name_prefix}.parquet"));
    let idx_path = dir.join(format!("{name_prefix}.parquet.idx"));
    write_i64_column_to_path(&src_path, "v", values).unwrap();
    let source = ParquetFile::open(&src_path).unwrap();
    let builder = IndexBuilder::new(&source);
    builder
        .write_sorted_i64(&idx_path, "idx_v", 0)
        .expect("build sidecar");
    (src_path, idx_path)
}

#[test]
fn empty_lookup_returns_empty() {
    // Key not in the dataset → no hits, no values.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..1_000i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "absent", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    let hits: Vec<IndexHit> = idx.lookup_eq("idx_v", &Key::I64(99_999)).unwrap();
    assert!(hits.is_empty(), "expected no hits, got {}", hits.len());

    let vals = idx
        .read_column_i64_where_eq(&source, "idx_v", 99_999, 0)
        .unwrap();
    assert!(vals.is_empty());

    // And the baseline agrees.
    assert!(baseline_eq(&source, 0, 99_999).is_empty());
}

#[test]
fn point_lookup_single_match_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..1_000i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "unique", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    for key in [0i64, 1, 42, 500, 999] {
        let indexed = idx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline = baseline_eq(&source, 0, key);
        assert_eq!(indexed, baseline, "key={key}");
        assert_eq!(indexed.len(), 1, "key={key}: expected 1 match");
    }
}

#[test]
fn point_lookup_multi_match_matches_baseline() {
    // Repeated values: every key appears 7 times (interleaved). This
    // exercises the binary-search-then-linear-scan duplicate walk
    // *and* the per-row-group bitmap assembly when matches span
    // multiple pages.
    let dir = tempfile::tempdir().unwrap();
    let mut values: Vec<i64> = Vec::with_capacity(7 * 5_000);
    for _ in 0..7 {
        for v in 0..5_000i64 {
            values.push(v);
        }
    }
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "repeated", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    for key in [0i64, 1, 1234, 4999] {
        let indexed = idx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline = baseline_eq(&source, 0, key);
        assert_eq!(indexed.len(), 7, "key={key}: expected 7 matches");
        assert_eq!(indexed, baseline, "key={key}");
    }
}

#[test]
fn edge_keys_min_and_max_match_baseline() {
    // The smallest and largest values in the column hit the binary-
    // search boundary conditions (start of vector / end of vector).
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (-500i64..500).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "signed", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    for key in [-500i64, 499] {
        let indexed = idx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline = baseline_eq(&source, 0, key);
        assert_eq!(indexed, baseline, "key={key}");
        assert_eq!(indexed.len(), 1);
    }
}

#[test]
fn all_present_keys_match_baseline_exhaustively() {
    // Smaller dataset; check every present key. Catches any
    // off-by-one in the duplicate walk or bitmap-assembly logic.
    let dir = tempfile::tempdir().unwrap();
    // 200 rows; values 0..40 each appearing 5 times.
    let mut values: Vec<i64> = Vec::with_capacity(200);
    for v in 0..40i64 {
        for _ in 0..5 {
            values.push(v);
        }
    }
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "exhaustive", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    for key in 0i64..40 {
        let indexed = idx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline = baseline_eq(&source, 0, key);
        assert_eq!(indexed, baseline, "key={key}");
    }
}

#[test]
fn lookup_eq_returns_one_hit_per_distinct_page_match() {
    // Single-RG, small page count. Verifies the IndexHit shape:
    // .row_group is the source RG ordinal, .page is the data-page
    // ordinal, .rowset is page-relative.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..100i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "small", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    let hits = idx.lookup_eq("idx_v", &Key::I64(42)).unwrap();
    assert_eq!(hits.len(), 1, "unique key → exactly one hit");
    assert_eq!(hits[0].row_group, 0);
    // Exactly one bit set in the rowset.
    let popcount: u32 = hits[0].rowset.iter().map(|b| b.count_ones()).sum();
    assert_eq!(
        popcount, 1,
        "rowset for a single match must have one bit set"
    );
}

#[test]
fn rejects_wrong_key_variant() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..50i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "wrong_type", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    let err = idx.lookup_eq("idx_v", &Key::I32(42)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("variant") || msg.contains("physical type"),
        "expected variant-mismatch error, got: {msg}"
    );
}

#[test]
fn rejects_missing_index_name() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..50i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "missing_name", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    let err = idx
        .lookup_eq("idx_does_not_exist", &Key::I64(0))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("idx_does_not_exist"),
        "expected name-mention in error, got: {msg}"
    );
}

#[test]
fn rejects_fingerprint_mismatch() {
    // Build a sidecar against one source, then re-open against a
    // different source (different row count). The fingerprint check
    // must fire at `ParquetIndex::open`.
    let dir = tempfile::tempdir().unwrap();
    let original: Vec<i64> = (0..500i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "fp", &original);

    // Rewrite the source file with different data (same column name,
    // different rows). Its fingerprint must differ.
    let new_values: Vec<i64> = (0..600i64).collect();
    write_i64_column_to_path(&src_path, "v", &new_values).unwrap();
    let new_source = ParquetFile::open(&src_path).unwrap();

    let err = ParquetIndex::open(&idx_path, &new_source).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("fingerprint"),
        "expected fingerprint-mismatch error, got: {msg}"
    );
}

#[test]
fn manifest_round_trips_through_open() {
    // Sanity: what the builder wrote, the reader sees.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..50i64).collect();
    let (src_path, idx_path) = build_source_and_sidecar(dir.path(), "manifest", &values);
    let source = ParquetFile::open(&src_path).unwrap();
    let idx = ParquetIndex::open(&idx_path, &source).unwrap();

    let m = idx.manifest();
    assert_eq!(m.indexes.len(), 1);
    assert_eq!(m.indexes[0].name, "idx_v");
    assert_eq!(m.indexes[0].sidecar_row_group, 0);
}
