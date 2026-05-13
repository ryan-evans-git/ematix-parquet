//! Unit coverage for the new `decompress_snappy_into` API.

use ematix_parquet_codec::compression::{
    decompress_snappy, decompress_snappy_into, decompress_zstd, decompress_zstd_into,
};

fn snappy_compress(input: &[u8]) -> Vec<u8> {
    let mut enc = snap::raw::Encoder::new();
    enc.compress_vec(input).unwrap()
}

fn zstd_compress(input: &[u8]) -> Vec<u8> {
    zstd::stream::encode_all(input, 3).unwrap()
}

#[test]
fn decompress_snappy_into_matches_decompress_snappy() {
    let original = b"hello hello hello hello world world world".repeat(10);
    let compressed = snappy_compress(&original);

    let owned = decompress_snappy(&compressed).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    decompress_snappy_into(&compressed, &mut buf).unwrap();
    assert_eq!(owned, buf);
    assert_eq!(buf, original);
}

#[test]
fn decompress_snappy_into_reuses_buffer_across_calls() {
    let a = b"the quick brown fox jumps over the lazy dog".repeat(5);
    let b = b"another payload of similar shape".repeat(5);
    let ca = snappy_compress(&a);
    let cb = snappy_compress(&b);

    let mut buf: Vec<u8> = Vec::new();
    decompress_snappy_into(&ca, &mut buf).unwrap();
    assert_eq!(buf, a);
    let cap_after_first = buf.capacity();

    // Second call with a smaller payload — capacity should not shrink.
    decompress_snappy_into(&cb, &mut buf).unwrap();
    assert_eq!(buf, b);
    assert!(
        buf.capacity() >= cap_after_first,
        "buffer capacity must not shrink between calls"
    );
}

#[test]
fn decompress_snappy_into_overwrites_prior_contents() {
    let original = vec![42u8; 1024];
    let compressed = snappy_compress(&original);

    let mut buf: Vec<u8> = vec![0xFFu8; 999];
    decompress_snappy_into(&compressed, &mut buf).unwrap();
    assert_eq!(buf, original);
}

// ---- ZSTD --------------------------------------------------------------

#[test]
fn decompress_zstd_roundtrip_owned() {
    let original = b"zstd payload zstd payload zstd payload".repeat(20);
    let compressed = zstd_compress(&original);
    let decoded = decompress_zstd(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn decompress_zstd_into_matches_decompress_zstd() {
    let original = b"the quick brown fox jumps over the lazy dog".repeat(50);
    let compressed = zstd_compress(&original);

    let owned = decompress_zstd(&compressed).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    decompress_zstd_into(&compressed, &mut buf).unwrap();
    assert_eq!(owned, buf);
    assert_eq!(buf, original);
}

#[test]
fn decompress_zstd_into_reuses_buffer_across_calls() {
    let a = vec![7u8; 4096];
    let b = vec![9u8; 1024];
    let ca = zstd_compress(&a);
    let cb = zstd_compress(&b);

    let mut buf: Vec<u8> = Vec::new();
    decompress_zstd_into(&ca, &mut buf).unwrap();
    assert_eq!(buf, a);
    let cap_after_first = buf.capacity();

    decompress_zstd_into(&cb, &mut buf).unwrap();
    assert_eq!(buf, b);
    assert!(
        buf.capacity() >= cap_after_first,
        "zstd buffer capacity must not shrink between calls"
    );
}

#[test]
fn decompress_zstd_into_overwrites_prior_contents() {
    let original = vec![13u8; 2048];
    let compressed = zstd_compress(&original);

    let mut buf: Vec<u8> = vec![0xAAu8; 777];
    decompress_zstd_into(&compressed, &mut buf).unwrap();
    assert_eq!(buf, original);
}

#[test]
fn decompress_zstd_error_on_garbage() {
    let bad = b"not a real zstd frame at all".to_vec();
    let mut buf = Vec::new();
    assert!(decompress_zstd(&bad).is_err());
    assert!(decompress_zstd_into(&bad, &mut buf).is_err());
}
