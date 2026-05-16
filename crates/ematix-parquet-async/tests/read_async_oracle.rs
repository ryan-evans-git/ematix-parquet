//! Π.11b oracle: `read_column_*_async` parity vs the sync
//! `ematix_parquet_codec::read::read_column_*`. Same fixtures,
//! same column index, byte-identical output Vec<T>.
//!
//! The async path uses AsyncParquetFile (one GET per chunk via
//! object_store::local::LocalFileSystem) and dispatches into a
//! local mirror of the codec's sync chunk-walker. If parity holds,
//! callers can swap sync → async without any value drift.

use std::sync::Arc;

use ematix_parquet_async::{
    read_column_f64_async, read_column_f64_async_into, read_column_i32_async,
    read_column_i32_async_into, read_column_i64_async, read_column_i64_async_into,
    AsyncParquetFile,
};
use ematix_parquet_codec::read::{
    read_column_f64 as sync_f64, read_column_i32 as sync_i32, read_column_i64 as sync_i64,
};
use ematix_parquet_codec::write::{
    write_f64_column_dict_to_path, write_f64_column_to_path, write_i32_column_dict_to_path,
    write_i32_column_to_path, write_i64_column_dict_to_path, write_i64_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;

fn fs_store_for(tmp: &std::path::Path) -> Arc<LocalFileSystem> {
    Arc::new(LocalFileSystem::new_with_prefix(tmp).unwrap())
}

// ============================================================
// i64
// ============================================================

#[tokio::test]
async fn i64_plain_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| i * 13 - 1000).collect();
    write_i64_column_to_path(&abs, "v", &values).unwrap();

    let sync = sync_i64(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let async_out = read_column_i64_async(
        &AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
            .await
            .unwrap(),
        0,
        0,
    )
    .await
    .unwrap();
    assert_eq!(async_out, sync);
    assert_eq!(async_out, values);
}

#[tokio::test]
async fn i64_dict_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("dict.parquet");
    let palette: [i64; 6] = [10, 20, 30, 40, 50, 60];
    let values: Vec<i64> = (0..3_000).map(|i| palette[i % 6]).collect();
    write_i64_column_dict_to_path(&abs, "v", &values, CompressionCodec::Snappy).unwrap();

    let sync = sync_i64(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("dict.parquet"))
        .await
        .unwrap();
    let async_out = read_column_i64_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_out, sync);
}

#[tokio::test]
async fn i64_async_into_reuses_buffer() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("reuse.parquet");
    let values: Vec<i64> = (0..1_500i64).collect();
    write_i64_column_to_path(&abs, "v", &values).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("reuse.parquet"))
        .await
        .unwrap();
    let mut buf: Vec<i64> = Vec::new();
    read_column_i64_async_into(&aps, 0, 0, &mut buf)
        .await
        .unwrap();
    let cap_first = buf.capacity();
    for _ in 0..5 {
        read_column_i64_async_into(&aps, 0, 0, &mut buf)
            .await
            .unwrap();
    }
    assert_eq!(buf.capacity(), cap_first, "buffer capacity must not grow");
    assert_eq!(buf, values);
}

// ============================================================
// i32
// ============================================================

#[tokio::test]
async fn i32_plain_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<i32> = (0..2_500i32).map(|i| i * 7).collect();
    write_i32_column_to_path(&abs, "v", &values).unwrap();

    let sync = sync_i32(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let async_out = read_column_i32_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_out, sync);
}

#[tokio::test]
async fn i32_dict_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("d.parquet");
    let palette: [i32; 5] = [-1, 0, 1, 2, 3];
    let values: Vec<i32> = (0..2_000).map(|i| palette[i % 5]).collect();
    write_i32_column_dict_to_path(&abs, "v", &values, CompressionCodec::Snappy).unwrap();

    let sync = sync_i32(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("d.parquet"))
        .await
        .unwrap();
    let mut out = Vec::new();
    read_column_i32_async_into(&aps, 0, 0, &mut out)
        .await
        .unwrap();
    assert_eq!(out, sync);
}

// ============================================================
// f64
// ============================================================

#[tokio::test]
async fn f64_plain_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let values: Vec<f64> = (0..1_000).map(|i| i as f64 * 0.5 - 100.0).collect();
    write_f64_column_to_path(&abs, "v", &values).unwrap();

    let sync = sync_f64(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let async_out = read_column_f64_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_out, sync);
}

#[tokio::test]
async fn f64_dict_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("d.parquet");
    let palette: [f64; 4] = [1.1, 2.2, 3.3, 4.4];
    let values: Vec<f64> = (0..1_200).map(|i| palette[i % 4]).collect();
    write_f64_column_dict_to_path(&abs, "v", &values, CompressionCodec::Snappy).unwrap();

    let sync = sync_f64(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("d.parquet"))
        .await
        .unwrap();
    let async_out = read_column_f64_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_out, sync);
}

// ============================================================
// Multi-row-group sweep
// ============================================================

#[tokio::test]
async fn i64_multi_row_group_each_rg_independently() {
    use ematix_parquet_codec::write::{write_table_to_path_with_row_group_size, ColumnData};

    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("multi.parquet");
    let values: Vec<i64> = (0..3_000i64).collect();
    let cols = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&abs, &cols, CompressionCodec::Snappy, 1_000).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("multi.parquet"))
        .await
        .unwrap();
    let md = aps.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 3);

    let mut all: Vec<i64> = Vec::new();
    for rg in 0..md.row_groups.len() {
        let chunk = read_column_i64_async(&aps, rg, 0).await.unwrap();
        all.extend(chunk);
    }
    assert_eq!(all, values);
}
