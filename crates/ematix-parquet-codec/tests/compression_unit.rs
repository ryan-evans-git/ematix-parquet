//! Unit coverage for the new `decompress_snappy_into` API.

use ematix_parquet_codec::compression::{decompress_snappy, decompress_snappy_into};

fn snappy_compress(input: &[u8]) -> Vec<u8> {
    let mut enc = snap::raw::Encoder::new();
    enc.compress_vec(input).unwrap()
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
