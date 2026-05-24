//! Π.19b oracle: composite leading-prefix index over (INT64, INT64).
//!
//! Two correctness invariants:
//! 1. **Exact 2-tuple equality matches a full-scan filter.** For
//!    every `(a, b)` query, `read_column_*_where_composite_eq`
//!    returns exactly the rows a manual `col_a == a && col_b == b`
//!    filter would.
//! 2. **Leading-prefix equality matches a full-scan filter on the
//!    leading column.** `read_column_*_where_composite_prefix(a)`
//!    returns the rows where `col_a == a`, regardless of `col_b`.

use ematix_parquet_codec::index::{IndexBuilder, Key, ParquetIndex};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::{write_table_to_path, ColumnData};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

/// Two-column INT64 parquet at `path`. Both columns have the same
/// row count.
fn write_two_i64_columns(
    path: &std::path::Path,
    name_a: &str,
    values_a: &[i64],
    name_b: &str,
    values_b: &[i64],
) {
    assert_eq!(values_a.len(), values_b.len());
    let cols: &[(&str, ColumnData<'_>)] = &[
        (name_a, ColumnData::I64(values_a)),
        (name_b, ColumnData::I64(values_b)),
    ];
    write_table_to_path(path, cols, CompressionCodec::Snappy).unwrap();
}

/// Full-scan baseline: rows where col_a == a AND col_b == b.
fn baseline_composite_eq(
    source: &ParquetFile,
    col_a: usize,
    col_b: usize,
    a: i64,
    b: i64,
) -> Vec<i64> {
    let va = read_column_i64(source, 0, col_a).unwrap();
    let vb = read_column_i64(source, 0, col_b).unwrap();
    va.iter()
        .zip(vb.iter())
        .filter(|(x, y)| **x == a && **y == b)
        .map(|(x, _)| *x)
        .collect()
}

/// Full-scan baseline: rows where col_a == a (col_b is free).
fn baseline_composite_prefix(source: &ParquetFile, col_a: usize, a: i64) -> Vec<i64> {
    let va = read_column_i64(source, 0, col_a).unwrap();
    va.iter().copied().filter(|x| *x == a).collect()
}

#[test]
fn composite_eq_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("composite_eq.parquet");
    let idx = dir.path().join("composite_eq.parquet.idx");

    // Synthetic 2-D grid: a in 0..50, b in 0..20, 5 repeats per cell.
    let mut a: Vec<i64> = Vec::new();
    let mut b: Vec<i64> = Vec::new();
    for av in 0..50i64 {
        for bv in 0..20i64 {
            for _ in 0..5 {
                a.push(av);
                b.push(bv);
            }
        }
    }
    write_two_i64_columns(&src, "a", &a, "b", &b);
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_composite_prefix_i64_i64(&idx, "idx_ab", 0, 1)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Read back col_a where (a, b) matches — expect 5 copies of a
    // (one per repeat in the cell).
    for av in [0i64, 1, 25, 49] {
        for bv in [0i64, 5, 19] {
            let indexed = reader
                .read_column_i64_where_composite_eq(&source, "idx_ab", av, bv, 0)
                .unwrap();
            let baseline = baseline_composite_eq(&source, 0, 1, av, bv);
            assert_eq!(indexed, baseline, "(a={av}, b={bv})");
            assert_eq!(indexed.len(), 5);
        }
    }

    // Absent tuple.
    let indexed = reader
        .read_column_i64_where_composite_eq(&source, "idx_ab", 999, 999, 0)
        .unwrap();
    assert!(indexed.is_empty());

    // (a, b) where a is missing entirely.
    let indexed = reader
        .read_column_i64_where_composite_eq(&source, "idx_ab", 999, 0, 0)
        .unwrap();
    assert!(indexed.is_empty());

    // (a, b) where a is present but b is missing.
    let indexed = reader
        .read_column_i64_where_composite_eq(&source, "idx_ab", 0, 999, 0)
        .unwrap();
    assert!(indexed.is_empty());
}

#[test]
fn composite_prefix_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("composite_prefix.parquet");
    let idx = dir.path().join("composite_prefix.parquet.idx");

    let mut a: Vec<i64> = Vec::new();
    let mut b: Vec<i64> = Vec::new();
    for av in 0..30i64 {
        for bv in 0..10i64 {
            for _ in 0..3 {
                a.push(av);
                b.push(bv);
            }
        }
    }
    write_two_i64_columns(&src, "a", &a, "b", &b);
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_composite_prefix_i64_i64(&idx, "idx_ab", 0, 1)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Prefix on a only — expect 10 * 3 = 30 rows per a.
    for av in [0i64, 1, 15, 29] {
        let indexed = reader
            .read_column_i64_where_composite_prefix(&source, "idx_ab", av, 0)
            .unwrap();
        let baseline = baseline_composite_prefix(&source, 0, av);
        assert_eq!(indexed, baseline, "a={av}");
        assert_eq!(indexed.len(), 30);
    }

    // Absent leading value.
    let indexed = reader
        .read_column_i64_where_composite_prefix(&source, "idx_ab", 999, 0)
        .unwrap();
    assert!(indexed.is_empty());
}

#[test]
fn composite_target_column_can_differ_from_leading() {
    // Read back col_b (the trailing key) via a composite lookup on
    // (a, b). The bitmap is anchored to col_a's pages, but the
    // masked decode works against col_b without issue — both
    // columns share the row-group's row count.
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("composite_target.parquet");
    let idx = dir.path().join("composite_target.parquet.idx");

    let a: Vec<i64> = (0..100i64).map(|i| i / 10).collect(); // 0..9 each repeated 10
    let b: Vec<i64> = (0..100i64).map(|i| i % 10).collect(); // 0..9 cycled
    write_two_i64_columns(&src, "a", &a, "b", &b);
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_composite_prefix_i64_i64(&idx, "idx_ab", 0, 1)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // (a=5, b=3) matches one row at position 53. Decoding col_b
    // should return [3].
    let indexed = reader
        .read_column_i64_where_composite_eq(&source, "idx_ab", 5, 3, 1)
        .unwrap();
    assert_eq!(indexed, vec![3]);
}

#[test]
fn rejects_self_referential_composite() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("self_ref.parquet");
    let idx = dir.path().join("self_ref.parquet.idx");
    let v: Vec<i64> = vec![1, 2, 3];
    write_two_i64_columns(&src, "a", &v, "b", &v);
    let source = ParquetFile::open(&src).unwrap();
    let err = IndexBuilder::new(&source)
        .write_sorted_composite_prefix_i64_i64(&idx, "idx_ab", 0, 0)
        .unwrap_err();
    assert!(format!("{err}").contains("must differ"));
}

#[test]
fn rejects_wrong_key_variants_on_composite_lookup() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("wrong_types.parquet");
    let idx = dir.path().join("wrong_types.parquet.idx");
    let v: Vec<i64> = vec![1, 2, 3];
    write_two_i64_columns(&src, "a", &v, "b", &v);
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_composite_prefix_i64_i64(&idx, "idx_ab", 0, 1)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let err = reader
        .lookup_composite_eq("idx_ab", &Key::I32(1), &Key::I64(1))
        .unwrap_err();
    assert!(format!("{err}").contains("variants"));

    let err = reader
        .lookup_composite_prefix("idx_ab", &Key::Bytes(b"x"))
        .unwrap_err();
    assert!(format!("{err}").contains("INT64"));
}

#[test]
fn rejects_cross_kind_composite_calls() {
    // composite API on a sorted index → reject.
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("sorted_for_composite.parquet");
    let idx = dir.path().join("sorted_for_composite.parquet.idx");
    let v: Vec<i64> = vec![1, 2, 3];
    let cols: &[(&str, ematix_parquet_codec::write::ColumnData<'_>)] =
        &[("a", ematix_parquet_codec::write::ColumnData::I64(&v))];
    ematix_parquet_codec::write::write_table_to_path(
        &src,
        cols,
        ematix_parquet_format::types::CompressionCodec::Snappy,
    )
    .unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let err = reader
        .lookup_composite_eq("idx_v", &Key::I64(1), &Key::I64(1))
        .unwrap_err();
    assert!(format!("{err}").contains("composite-prefix"));
}
