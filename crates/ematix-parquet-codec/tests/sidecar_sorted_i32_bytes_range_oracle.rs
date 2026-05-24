//! Π.18 oracle: sorted indexes over INT32 + BYTE_ARRAY, plus range
//! queries on all three sorted types.
//!
//! These are *type/shape* tests on top of the Π.17b pipeline. The
//! end-to-end correctness invariant — indexed reads return exactly
//! the rows a full-scan filter would — is already proven by
//! `sidecar_sorted_i64_oracle`. This file proves the new variants
//! plumb through the same pipeline without divergence.

use ematix_parquet_codec::index::{IndexBuilder, Key, ParquetIndex};
use ematix_parquet_codec::read::{read_column_byte_array, read_column_i32, read_column_i64};
use ematix_parquet_codec::write::{
    write_byte_array_column_to_path, write_i32_column_to_path, write_i64_column_to_path,
};
use ematix_parquet_io::ParquetFile;

// ============================================================
// Baselines
// ============================================================

fn baseline_eq_i32(source: &ParquetFile, key: i32) -> Vec<i32> {
    let values = read_column_i32(source, 0, 0).unwrap();
    values.iter().copied().filter(|v| *v == key).collect()
}

fn baseline_range_i64(source: &ParquetFile, lo: i64, hi: i64) -> Vec<i64> {
    let values = read_column_i64(source, 0, 0).unwrap();
    values
        .iter()
        .copied()
        .filter(|v| *v >= lo && *v <= hi)
        .collect()
}

fn baseline_range_i32(source: &ParquetFile, lo: i32, hi: i32) -> Vec<i32> {
    let values = read_column_i32(source, 0, 0).unwrap();
    values
        .iter()
        .copied()
        .filter(|v| *v >= lo && *v <= hi)
        .collect()
}

fn baseline_eq_bytes(source: &ParquetFile, key: &[u8]) -> Vec<Vec<u8>> {
    let values = read_column_byte_array(source, 0, 0).unwrap();
    values.into_iter().filter(|v| v.as_slice() == key).collect()
}

fn baseline_range_bytes(source: &ParquetFile, lo: &[u8], hi: &[u8]) -> Vec<Vec<u8>> {
    let values = read_column_byte_array(source, 0, 0).unwrap();
    values
        .into_iter()
        .filter(|v| v.as_slice() >= lo && v.as_slice() <= hi)
        .collect()
}

// ============================================================
// INT32 sidecar
// ============================================================

#[test]
fn i32_point_lookup_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("i32_eq.parquet");
    let idx = dir.path().join("i32_eq.parquet.idx");

    // Mix of duplicates and uniques to exercise the duplicate walk.
    let mut values: Vec<i32> = Vec::new();
    for v in 0..200i32 {
        for _ in 0..3 {
            values.push(v);
        }
    }
    write_i32_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i32(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    for key in [0i32, 1, 42, 100, 199] {
        let indexed = reader
            .read_column_i32_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline = baseline_eq_i32(&source, key);
        assert_eq!(indexed, baseline, "key={key}");
        assert_eq!(indexed.len(), 3, "key={key}: 3 duplicates expected");
    }

    // Absent key.
    let indexed = reader
        .read_column_i32_where_eq(&source, "idx_v", 999, 0)
        .unwrap();
    assert!(indexed.is_empty());
}

#[test]
fn i32_rejects_wrong_key_variant() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("i32_wrong.parquet");
    let idx = dir.path().join("i32_wrong.parquet.idx");
    let values: Vec<i32> = (0..50i32).collect();
    write_i32_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i32(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // INT64 key on an INT32 index → reject.
    let err = reader.lookup_eq("idx_v", &Key::I64(0)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("variant") || msg.contains("physical type"),
        "got: {msg}"
    );
}

#[test]
fn i32_builder_rejects_non_int32_column() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("i32_typecheck.parquet");
    let idx = dir.path().join("i32_typecheck.parquet.idx");
    // Write INT64; ask the i32 builder to use column 0.
    let values: Vec<i64> = (0..10i64).collect();
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    let err = IndexBuilder::new(&source)
        .write_sorted_i32(&idx, "idx_v", 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("INT32"), "got: {msg}");
}

// ============================================================
// BYTE_ARRAY sidecar
// ============================================================

#[test]
fn bytes_point_lookup_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("bytes_eq.parquet");
    let idx = dir.path().join("bytes_eq.parquet.idx");

    // Mixed-length byte strings, some duplicates.
    let raw: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta".to_vec(),
        b"alpha".to_vec(),
        b"gamma".to_vec(),
        b"delta".to_vec(),
        b"alpha".to_vec(),
        b"epsilon".to_vec(),
        b"beta".to_vec(),
        b"".to_vec(),
        b"zeta".to_vec(),
    ];
    let view: Vec<&[u8]> = raw.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&src, "v", &view).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_byte_array(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Three occurrences of "alpha".
    let hits = reader
        .read_column_byte_array_where_eq(&source, "idx_v", b"alpha", 0)
        .unwrap();
    let baseline = baseline_eq_bytes(&source, b"alpha");
    assert_eq!(hits, baseline);
    assert_eq!(hits.len(), 3);

    // Two occurrences of "beta".
    let hits = reader
        .read_column_byte_array_where_eq(&source, "idx_v", b"beta", 0)
        .unwrap();
    assert_eq!(hits, baseline_eq_bytes(&source, b"beta"));
    assert_eq!(hits.len(), 2);

    // Empty-string key — Parquet BYTE_ARRAY allows zero-length values.
    let hits = reader
        .read_column_byte_array_where_eq(&source, "idx_v", b"", 0)
        .unwrap();
    assert_eq!(hits, baseline_eq_bytes(&source, b""));
    assert_eq!(hits.len(), 1);

    // Missing key.
    let hits = reader
        .read_column_byte_array_where_eq(&source, "idx_v", b"omega", 0)
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn bytes_builder_rejects_non_byte_array_column() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("bytes_typecheck.parquet");
    let idx = dir.path().join("bytes_typecheck.parquet.idx");
    let values: Vec<i64> = (0..10i64).collect();
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    let err = IndexBuilder::new(&source)
        .write_sorted_byte_array(&idx, "idx_v", 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("BYTE_ARRAY"), "got: {msg}");
}

// ============================================================
// Range queries — all three types
// ============================================================

#[test]
fn i64_range_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("i64_range.parquet");
    let idx = dir.path().join("i64_range.parquet.idx");
    let values: Vec<i64> = (0..1_000i64).collect();
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    for (lo, hi) in [
        (100i64, 200i64), // mid-range
        (0, 0),           // single-element
        (0, 999),         // full span
        (995, 999),       // tail
        (-10, 5),         // partial overlap on the low side
        (500, 100),       // inverted — empty
    ] {
        let indexed = reader
            .read_column_i64_where_range(&source, "idx_v", lo, hi, 0)
            .unwrap();
        let baseline = if lo <= hi {
            baseline_range_i64(&source, lo, hi)
        } else {
            Vec::new()
        };
        assert_eq!(indexed, baseline, "range=[{lo}, {hi}]");
    }
}

#[test]
fn i32_range_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("i32_range.parquet");
    let idx = dir.path().join("i32_range.parquet.idx");
    let values: Vec<i32> = (-200i32..200).collect();
    write_i32_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i32(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    for (lo, hi) in [
        (-200i32, -100), // negative range
        (-5, 5),         // straddles zero
        (100, 199),      // positive
        (-300, -250),    // entirely below
        (250, 300),      // entirely above
    ] {
        let indexed = reader
            .read_column_i32_where_range(&source, "idx_v", lo, hi, 0)
            .unwrap();
        let baseline = baseline_range_i32(&source, lo, hi);
        assert_eq!(indexed, baseline, "range=[{lo}, {hi}]");
    }
}

#[test]
fn bytes_range_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("bytes_range.parquet");
    let idx = dir.path().join("bytes_range.parquet.idx");
    let raw: Vec<Vec<u8>> = vec![
        b"apple".to_vec(),
        b"banana".to_vec(),
        b"cherry".to_vec(),
        b"date".to_vec(),
        b"elderberry".to_vec(),
        b"fig".to_vec(),
        b"grape".to_vec(),
        b"honeydew".to_vec(),
    ];
    let view: Vec<&[u8]> = raw.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&src, "v", &view).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_byte_array(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // "banana" .. "elderberry" (inclusive) should give 4 values.
    let hits = reader
        .read_column_byte_array_where_range(&source, "idx_v", b"banana", b"elderberry", 0)
        .unwrap();
    let baseline = baseline_range_bytes(&source, b"banana", b"elderberry");
    assert_eq!(hits, baseline);
    assert_eq!(hits.len(), 4);

    // Prefix "f.." sweep: lo = "f", hi = "fz" — should match "fig".
    let hits = reader
        .read_column_byte_array_where_range(&source, "idx_v", b"f", b"fz", 0)
        .unwrap();
    let baseline = baseline_range_bytes(&source, b"f", b"fz");
    assert_eq!(hits, baseline);
    assert_eq!(hits.len(), 1);

    // Empty range — lo > hi should return empty.
    let hits = reader
        .read_column_byte_array_where_range(&source, "idx_v", b"z", b"a", 0)
        .unwrap();
    assert!(hits.is_empty());
}
