//! Π.11d oracle: async streaming Stream API parity.
//!
//! Concatenating every yielded batch reproduces the full
//! `read_column_*_async` output. Non-final batches are exactly
//! `batch_size`; the final batch may be shorter.

use std::sync::Arc;

use ematix_parquet_async::{
    read_column_f64_async, read_column_f64_async_stream, read_column_i32_async,
    read_column_i32_async_stream, read_column_i64_async, read_column_i64_async_stream,
    AsyncParquetFile,
};
use ematix_parquet_codec::write::{
    write_f64_column_to_path, write_i32_column_to_path, write_i64_column_dict_to_path,
    write_i64_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use futures_util::stream::TryStreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;

fn fs_store_for(tmp: &std::path::Path) -> Arc<LocalFileSystem> {
    Arc::new(LocalFileSystem::new_with_prefix(tmp).unwrap())
}

#[tokio::test]
async fn i64_stream_concat_matches_async() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..7_777i64).map(|i| i * 5).collect();
    write_i64_column_to_path(&abs, "v", &values).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let full = read_column_i64_async(&aps, 0, 0).await.unwrap();

    for &batch in &[1usize, 100, 1024, 4096, 7777, 10_000] {
        let stream = read_column_i64_async_stream(&aps, 0, 0, batch);
        let batches: Vec<Vec<i64>> = stream.try_collect().await.unwrap();
        let n = batches.len();
        for (i, b) in batches.iter().enumerate() {
            if i + 1 < n {
                assert_eq!(b.len(), batch, "batch_size={batch} non-final");
            } else {
                assert!(!b.is_empty() && b.len() <= batch);
            }
        }
        let flat: Vec<i64> = batches.into_iter().flatten().collect();
        assert_eq!(flat, full, "batch_size={batch} concat mismatch");
    }
}

#[tokio::test]
async fn i64_stream_dict_concat_matches_async() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("d.parquet");
    let palette: [i64; 5] = [11, 22, 33, 44, 55];
    let values: Vec<i64> = (0..3_000).map(|i| palette[i % 5]).collect();
    write_i64_column_dict_to_path(&abs, "v", &values, CompressionCodec::Snappy).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("d.parquet"))
        .await
        .unwrap();
    let full = read_column_i64_async(&aps, 0, 0).await.unwrap();

    let stream = read_column_i64_async_stream(&aps, 0, 0, 256);
    let flat: Vec<i64> = stream.try_collect::<Vec<Vec<i64>>>().await.unwrap()
        .into_iter().flatten().collect();
    assert_eq!(flat, full);
}

#[tokio::test]
async fn i32_stream_works() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<i32> = (0..2_500i32).collect();
    write_i32_column_to_path(&abs, "v", &values).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let full = read_column_i32_async(&aps, 0, 0).await.unwrap();
    let stream = read_column_i32_async_stream(&aps, 0, 0, 333);
    let flat: Vec<i32> = stream.try_collect::<Vec<Vec<i32>>>().await.unwrap()
        .into_iter().flatten().collect();
    assert_eq!(flat, full);
}

#[tokio::test]
async fn f64_stream_works() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<f64> = (0..1_111).map(|i| i as f64 / 7.0).collect();
    write_f64_column_to_path(&abs, "v", &values).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let full = read_column_f64_async(&aps, 0, 0).await.unwrap();
    let stream = read_column_f64_async_stream(&aps, 0, 0, 500);
    let flat: Vec<f64> = stream.try_collect::<Vec<Vec<f64>>>().await.unwrap()
        .into_iter().flatten().collect();
    assert_eq!(flat, full);
}

#[tokio::test]
async fn batch_size_zero_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    write_i64_column_to_path(&abs, "v", &[0i64, 1, 2]).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let stream = read_column_i64_async_stream(&aps, 0, 0, 0);
    let r: ematix_parquet_async::Result<Vec<Vec<i64>>> = stream.try_collect().await;
    assert!(r.is_err(), "batch_size=0 must error");
}

#[tokio::test]
async fn batch_size_exceeds_chunk_yields_one_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("small.parquet");
    let values: Vec<i64> = (0..50i64).collect();
    write_i64_column_to_path(&abs, "v", &values).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("small.parquet"))
        .await
        .unwrap();
    let stream = read_column_i64_async_stream(&aps, 0, 0, 10_000);
    let batches: Vec<Vec<i64>> = stream.try_collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], values);
}
