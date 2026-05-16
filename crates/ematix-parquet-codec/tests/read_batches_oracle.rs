//! Π.9c oracle: streaming/batched decode API.
//!
//! `read_column_*_batches(file, rg, col, batch_size)` yields a
//! sequence of `Vec<T>` batches. The acceptance contract:
//!   1. Concatenating every batch reproduces the same `Vec<T>` as
//!      the non-streaming `read_column_*` entry point (byte-identical).
//!   2. All batches except the final one have exactly `batch_size`
//!      values; the final may be shorter.
//!   3. `batch_size = 0` is rejected.
//!   4. `batch_size` larger than the whole chunk emits exactly one
//!      batch with every value.
//!   5. Empty chunk emits zero batches.

use ematix_parquet_codec::read::{
    read_column_f64, read_column_f64_batches, read_column_i32, read_column_i32_batches,
    read_column_i64, read_column_i64_batches,
};
use ematix_parquet_codec::write::{
    write_f64_column_to_path, write_i32_column_to_path, write_i64_column_to_path,
};
use ematix_parquet_io::ParquetFile;

fn collect<T, I: Iterator<Item = ematix_parquet_codec::error::Result<Vec<T>>>>(
    iter: I,
) -> Vec<Vec<T>> {
    iter.map(|r| r.unwrap()).collect()
}

#[test]
fn i64_batches_concatenate_to_full_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i * 7 - 100).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_i64(&file, 0, 0).unwrap();

    for batch_size in [1usize, 7, 100, 1024, 5_000, 9_999, 10_000, 50_000] {
        let iter = read_column_i64_batches(&file, 0, 0, batch_size).unwrap();
        let batches = collect(iter);
        // (2) all-but-last is exactly batch_size
        let n_batches = batches.len();
        for (i, b) in batches.iter().enumerate() {
            if i + 1 < n_batches {
                assert_eq!(
                    b.len(),
                    batch_size,
                    "batch_size={batch_size} non-final batch {i} has len {}",
                    b.len()
                );
            } else {
                assert!(
                    !b.is_empty() && b.len() <= batch_size,
                    "batch_size={batch_size} final batch len {} not in (0,{batch_size}]",
                    b.len()
                );
            }
        }
        let flat: Vec<i64> = batches.into_iter().flatten().collect();
        assert_eq!(flat, full, "batch_size={batch_size}: concat mismatch");
    }
}

#[test]
fn i32_batches_concatenate_to_full_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.parquet");
    let values: Vec<i32> = (0..5_555i32).map(|i| i * 3).collect();
    write_i32_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_i32(&file, 0, 0).unwrap();

    let iter = read_column_i32_batches(&file, 0, 0, 1024).unwrap();
    let flat: Vec<i32> = collect(iter).into_iter().flatten().collect();
    assert_eq!(flat, full);
}

#[test]
fn f64_batches_concatenate_to_full_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.parquet");
    let values: Vec<f64> = (0..2_500).map(|i| i as f64 * 0.25).collect();
    write_f64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_f64(&file, 0, 0).unwrap();

    let iter = read_column_f64_batches(&file, 0, 0, 256).unwrap();
    let flat: Vec<f64> = collect(iter).into_iter().flatten().collect();
    assert_eq!(flat, full);
}

#[test]
fn batch_size_zero_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("z.parquet");
    write_i64_column_to_path(&path, "v", &[0i64, 1, 2]).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    let r = read_column_i64_batches(&file, 0, 0, 0);
    assert!(r.is_err(), "batch_size=0 must error");
}

#[test]
fn batch_size_exceeding_chunk_yields_one_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.parquet");
    let values: Vec<i64> = (0..123).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let iter = read_column_i64_batches(&file, 0, 0, 10_000).unwrap();
    let batches = collect(iter);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], values);
}

#[test]
fn iterator_returns_none_after_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("done.parquet");
    let values: Vec<i64> = (0..50).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mut iter = read_column_i64_batches(&file, 0, 0, 25).unwrap();
    let b1 = iter.next().unwrap().unwrap();
    let b2 = iter.next().unwrap().unwrap();
    assert_eq!(b1.len(), 25);
    assert_eq!(b2.len(), 25);
    assert!(iter.next().is_none());
    assert!(iter.next().is_none(), "must remain None on repeated calls");
}

/// Dict-encoded column streams correctly too (the iterator must build
/// the dict before the first data page).
#[test]
fn dict_encoded_i64_batches_concatenate() {
    use ematix_parquet_codec::write::write_i64_column_dict_to_path;
    use ematix_parquet_format::types::CompressionCodec;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dict_i64.parquet");
    // 5K values cycling over 10 distinct → bw=4 dict pages.
    let palette: [i64; 10] = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110];
    let values: Vec<i64> = (0..5_000).map(|i| palette[i % 10]).collect();
    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_i64(&file, 0, 0).unwrap();
    let iter = read_column_i64_batches(&file, 0, 0, 333).unwrap();
    let flat: Vec<i64> = collect(iter).into_iter().flatten().collect();
    assert_eq!(flat, full);
    assert_eq!(flat, values);
}

/// Multi-row-group file: the iterator targets a single row group;
/// each RG produces its own stream. Concatenating both reproduces the
/// full table.
#[test]
fn multi_row_group_each_rg_streams_independently() {
    use ematix_parquet_codec::write::{write_table_to_path_with_row_group_size, ColumnData};
    use ematix_parquet_format::types::CompressionCodec;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_rg.parquet");
    let values: Vec<i64> = (0..3_000i64).collect();
    let cols = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Snappy, 1_000)
        .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 3);

    let mut all: Vec<i64> = Vec::new();
    for rg in 0..md.row_groups.len() {
        let iter = read_column_i64_batches(&file, rg, 0, 256).unwrap();
        for b in iter {
            all.extend(b.unwrap());
        }
    }
    assert_eq!(all, values);
}
