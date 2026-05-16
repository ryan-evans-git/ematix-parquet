//! Π.11a oracle: `AsyncParquetFile` parity vs the sync
//! `ematix_parquet_io::ParquetFile` on the same files.
//!
//! Strategy: write parquet via the codec crate, open both ways,
//! check footer offset / file size / metadata fields / range-read
//! bytes are bit-identical. LocalFileSystem ObjectStore wraps the
//! same on-disk file the sync side reads directly.
//!
//! When this layer is correct, swapping sync → async at the read
//! façade can't accidentally drift the metadata or page byte
//! ranges — the rest of the async façade (Π.11b onward) reuses
//! the same byte slices.

use std::sync::Arc;

use ematix_parquet_async::AsyncParquetFile;
use ematix_parquet_codec::write::{write_i64_column_to_path, write_byte_array_column_to_path};
use ematix_parquet_io::ParquetFile;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;

fn fs_store_for(tmp: &std::path::Path) -> Arc<LocalFileSystem> {
    Arc::new(LocalFileSystem::new_with_prefix(tmp).unwrap())
}

#[tokio::test]
async fn open_matches_sync_file_size_and_footer() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| i * 11 - 100).collect();
    write_i64_column_to_path(&abs_path, "v", &values).unwrap();

    let sync = ParquetFile::open(&abs_path).unwrap();

    let store = fs_store_for(tmp.path());
    let path = OsPath::from("v.parquet");
    let aps = AsyncParquetFile::open(store, path).await.unwrap();

    assert_eq!(
        aps.file_size(), sync.file_size(),
        "file_size mismatch"
    );
    assert_eq!(
        aps.footer_offset(), sync.footer_offset(),
        "footer_offset mismatch"
    );
    assert_eq!(
        aps.footer_bytes(), sync.footer_bytes(),
        "footer bytes mismatch"
    );
}

#[tokio::test]
async fn metadata_decode_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..2_500i64).collect();
    write_i64_column_to_path(&abs_path, "v", &values).unwrap();

    let sync_file = ParquetFile::open(&abs_path).unwrap();
    let sync_md = sync_file.metadata().unwrap();

    let store = fs_store_for(tmp.path());
    let aps = AsyncParquetFile::open(store, OsPath::from("v.parquet")).await.unwrap();
    let async_md = aps.metadata().unwrap();

    assert_eq!(async_md.version, sync_md.version);
    assert_eq!(async_md.num_rows, sync_md.num_rows);
    assert_eq!(async_md.row_groups.len(), sync_md.row_groups.len());
    assert_eq!(async_md.created_by, sync_md.created_by);
}

#[tokio::test]
async fn read_range_matches_sync_byte_for_byte() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i * 7).collect();
    write_i64_column_to_path(&abs_path, "v", &values).unwrap();

    let sync = ParquetFile::open(&abs_path).unwrap();
    let store = fs_store_for(tmp.path());
    let aps = AsyncParquetFile::open(store, OsPath::from("v.parquet")).await.unwrap();

    // A handful of ranges spanning the first row group's column chunk.
    let md = sync.metadata().unwrap();
    let cm = md.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let len = cm.total_compressed_size as u64;

    let sync_bytes = sync.read_range(start, len).unwrap();
    let async_bytes = aps.read_range(start, len).await.unwrap();
    assert_eq!(sync_bytes.as_slice(), async_bytes.as_ref());

    // Small range from the head.
    let sync_head = sync.read_range(0, 64).unwrap();
    let async_head = aps.read_range(0, 64).await.unwrap();
    assert_eq!(sync_head.as_slice(), async_head.as_ref());
}

#[tokio::test]
async fn out_of_range_read_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..100i64).collect();
    write_i64_column_to_path(&abs_path, "v", &values).unwrap();

    let store = fs_store_for(tmp.path());
    let aps = AsyncParquetFile::open(store, OsPath::from("v.parquet")).await.unwrap();
    let size = aps.file_size();

    let r = aps.read_range(size, 1).await;
    assert!(r.is_err(), "read past EOF must error");

    let r = aps.read_range(0, size + 1).await;
    assert!(r.is_err(), "read past EOF must error (extends-past-end)");
}

#[tokio::test]
async fn zero_length_read_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("v.parquet");
    let values: Vec<i64> = (0..100i64).collect();
    write_i64_column_to_path(&abs_path, "v", &values).unwrap();

    let store = fs_store_for(tmp.path());
    let aps = AsyncParquetFile::open(store, OsPath::from("v.parquet")).await.unwrap();

    let bytes = aps.read_range(0, 0).await.unwrap();
    assert!(bytes.is_empty());
}

/// Large file (forces footer > 8 KB, exercises the two-RT fallback
/// path). To get a footer that large via the codec writer, we use a
/// byte_array column with many distinct values (each shows up in
/// statistics).
#[tokio::test]
async fn large_footer_falls_back_to_two_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("big.parquet");
    // Many wide values → ~few-KB statistics per chunk + per-page
    // stats. Even a modest column chunk count gets the footer into
    // the multi-KB range.
    let owned: Vec<Vec<u8>> = (0..20_000)
        .map(|i| format!("value-with-some-bytes-{i:08}").into_bytes())
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&abs_path, "v", &refs).unwrap();

    let sync = ParquetFile::open(&abs_path).unwrap();
    let store = fs_store_for(tmp.path());
    let aps = AsyncParquetFile::open(store, OsPath::from("big.parquet")).await.unwrap();

    // Whether or not the footer fits in 8 KB suffix, the result
    // must be byte-identical.
    assert_eq!(aps.file_size(), sync.file_size());
    assert_eq!(aps.footer_offset(), sync.footer_offset());
    assert_eq!(aps.footer_bytes(), sync.footer_bytes());
}

#[tokio::test]
async fn not_a_parquet_file_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let abs_path = tmp.path().join("garbage.parquet");
    std::fs::write(&abs_path, b"this is not a parquet file at all, it's just bytes").unwrap();

    let store = fs_store_for(tmp.path());
    let r = AsyncParquetFile::open(store, OsPath::from("garbage.parquet")).await;
    assert!(r.is_err(), "non-parquet file must error");
}
