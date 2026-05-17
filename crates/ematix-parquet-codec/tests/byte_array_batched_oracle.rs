//! `read_column_byte_array_batches` oracle.
//!
//! Mirrors the scalar batched-API tests but for BYTE_ARRAY:
//!   * Streams batches that, when concatenated, equal the full-decode
//!     result (PLAIN + dict-encoded paths).
//!   * Final batch may be shorter than `batch_size`.
//!   * Works across multiple row groups (one iter per column chunk).
//!   * batch_size == 0 errors.

use ematix_parquet_codec::read::{read_column_byte_array, read_column_byte_array_batches};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path,
    write_table_to_path_with_row_group_size, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

fn collect_all_batches(file: &ParquetFile, rg: usize, batch_size: usize) -> Vec<Vec<u8>> {
    let iter = read_column_byte_array_batches(file, rg, 0, batch_size).unwrap();
    let mut out = Vec::new();
    for batch in iter {
        out.extend(batch.unwrap());
    }
    out
}

#[test]
fn plain_byte_array_batches_match_full_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.parquet");
    let owned: Vec<String> = (0..1024).map(|i| format!("row-{i:04}-payload")).collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
    write_byte_array_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(full.len(), 1024);

    // Try a handful of batch sizes; concat must equal full.
    for &bs in &[1usize, 7, 64, 1024, 4096] {
        let collected = collect_all_batches(&file, 0, bs);
        assert_eq!(collected, full, "batch_size {bs} mismatch");
    }
}

#[test]
fn dict_byte_array_batches_match_full_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dict.parquet");
    // Low-cardinality dict path.
    let dict_words: [&[u8]; 8] = [
        b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf", b"hotel",
    ];
    let values: Vec<&[u8]> = (0..2048).map(|i| dict_words[i % 8]).collect();
    write_byte_array_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let full = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(full.len(), 2048);

    for &bs in &[1usize, 13, 256, 2048, 99999] {
        let collected = collect_all_batches(&file, 0, bs);
        assert_eq!(collected, full, "batch_size {bs} mismatch");
    }
}

#[test]
fn multi_rg_byte_array_per_rg_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_rg.parquet");
    let owned: Vec<String> = (0..3000).map(|i| format!("v-{i}")).collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
    write_table_to_path_with_row_group_size(
        &path,
        &[("v", ColumnData::ByteArray(&values))],
        CompressionCodec::Uncompressed,
        500, // 6 row groups
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 6);

    let mut all_rows: Vec<Vec<u8>> = Vec::new();
    for rg_ix in 0..6 {
        let batch_size = 73; // odd, forces multi-batch per RG
        let batches = collect_all_batches(&file, rg_ix, batch_size);
        assert!(!batches.is_empty(), "rg {rg_ix} produced 0 rows");
        all_rows.extend(batches);
    }
    assert_eq!(all_rows.len(), 3000);
    for (i, row) in all_rows.iter().enumerate() {
        assert_eq!(row, &format!("v-{i}").into_bytes());
    }
}

#[test]
fn batch_size_zero_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("err.parquet");
    let v: Vec<&[u8]> = vec![b"a", b"b"];
    write_byte_array_column_to_path(&path, "v", &v).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    let r = read_column_byte_array_batches(&file, 0, 0, 0);
    assert!(r.is_err());
}

#[test]
fn final_batch_can_be_shorter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.parquet");
    let owned: Vec<String> = (0..100).map(|i| format!("v{i}")).collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
    write_byte_array_column_to_path(&path, "v", &values).unwrap();
    let file = ParquetFile::open(&path).unwrap();

    let iter = read_column_byte_array_batches(&file, 0, 0, 30).unwrap();
    let sizes: Vec<usize> = iter.map(|b| b.unwrap().len()).collect();
    // 100 rows / 30 batch_size → 30, 30, 30, 10
    assert_eq!(sizes, vec![30, 30, 30, 10]);
}
