//! Π.11c oracle: async byte_array façade parity with sync.
//! Both `Vec<Vec<u8>>` and Arrow-style `(bytes, offsets)` shapes.

use std::sync::Arc;

use ematix_parquet_async::{
    read_column_byte_array_async, read_column_byte_array_async_into,
    read_column_byte_array_offsets_async, read_column_byte_array_offsets_async_into,
    AsyncParquetFile,
};
use ematix_parquet_codec::read::{
    read_column_byte_array as sync_ba, read_column_byte_array_offsets as sync_ba_offsets,
};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;

fn fs_store_for(tmp: &std::path::Path) -> Arc<LocalFileSystem> {
    Arc::new(LocalFileSystem::new_with_prefix(tmp).unwrap())
}

#[tokio::test]
async fn byte_array_plain_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let owned: Vec<Vec<u8>> = (0..1_200)
        .map(|i| vec![(i % 26) as u8 + b'a'; (i % 5) + 1])
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&abs, "v", &refs).unwrap();

    let sync = sync_ba(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let async_out = read_column_byte_array_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_out, sync);
    assert_eq!(async_out, owned);
}

#[tokio::test]
async fn byte_array_dict_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("d.parquet");
    let palette: [&[u8]; 4] = [b"alpha", b"bravo", b"charlie", b"delta"];
    let owned: Vec<Vec<u8>> = (0..1_500).map(|i| palette[i % 4].to_vec()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_dict_to_path(&abs, "v", &refs, CompressionCodec::Snappy).unwrap();

    let sync = sync_ba(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("d.parquet"))
        .await
        .unwrap();
    let mut out = Vec::new();
    read_column_byte_array_async_into(&aps, 0, 0, &mut out).await.unwrap();
    assert_eq!(out, sync);
}

#[tokio::test]
async fn byte_array_offsets_plain_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("v.parquet");
    let owned: Vec<Vec<u8>> = (0..800)
        .map(|i| vec![(i % 26) as u8 + b'A'; (i % 9) + 1])
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&abs, "v", &refs).unwrap();

    let (sync_b, sync_o) = sync_ba_offsets(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("v.parquet"))
        .await
        .unwrap();
    let (async_b, async_o) = read_column_byte_array_offsets_async(&aps, 0, 0).await.unwrap();
    assert_eq!(async_b, sync_b);
    assert_eq!(async_o, sync_o);
    assert_eq!(async_o.len(), owned.len() + 1);
}

#[tokio::test]
async fn byte_array_offsets_dict_matches_sync() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("d.parquet");
    let palette: [&[u8]; 3] = [b"A", b"BB", b"CCC"];
    let owned: Vec<Vec<u8>> = (0..2_000).map(|i| palette[i % 3].to_vec()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_dict_to_path(&abs, "v", &refs, CompressionCodec::Uncompressed)
        .unwrap();

    let (sync_b, sync_o) = sync_ba_offsets(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("d.parquet"))
        .await
        .unwrap();
    let mut b = Vec::new();
    let mut o = Vec::new();
    read_column_byte_array_offsets_async_into(&aps, 0, 0, &mut b, &mut o).await.unwrap();
    assert_eq!(b, sync_b);
    assert_eq!(o, sync_o);
}

#[tokio::test]
async fn byte_array_offsets_async_into_clears_buffers() {
    let tmp = tempfile::tempdir().unwrap();
    let abs = tmp.path().join("reuse.parquet");
    let owned: Vec<Vec<u8>> = (0..200).map(|_| b"x".to_vec()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&abs, "v", &refs).unwrap();

    let aps = AsyncParquetFile::open(fs_store_for(tmp.path()), OsPath::from("reuse.parquet"))
        .await
        .unwrap();

    // Prefill: a second call must clear, not append.
    let mut bytes: Vec<u8> = vec![0xAB; 1024];
    let mut offsets: Vec<u32> = vec![999; 256];
    read_column_byte_array_offsets_async_into(&aps, 0, 0, &mut bytes, &mut offsets)
        .await
        .unwrap();
    assert_eq!(offsets.len(), owned.len() + 1, "offsets must be cleared on each call");
    let (want_b, want_o) = sync_ba_offsets(&ParquetFile::open(&abs).unwrap(), 0, 0).unwrap();
    assert_eq!(bytes, want_b);
    assert_eq!(offsets, want_o);
}
