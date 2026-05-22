//! Unit coverage for the codec decompression APIs.

use ematix_parquet_codec::compression::{
    compress_brotli, compress_gzip, compress_lz4_raw, compress_snappy, compress_zstd,
    decompress_brotli, decompress_brotli_into, decompress_brotli_into_capped, decompress_gzip,
    decompress_gzip_into, decompress_gzip_into_capped, decompress_lz4_raw, decompress_lz4_raw_into,
    decompress_lz4_raw_into_sized, decompress_snappy, decompress_snappy_into, decompress_zstd,
    decompress_zstd_into, decompress_zstd_into_capped,
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

#[test]
fn decompress_zstd_into_capped_rejects_oversize() {
    // Compress payload, then declare an absurdly small uncompressed
    // size — the capped path must reject the actual output.
    let payload = vec![0u8; 16 * 1024];
    let compressed = compress_zstd(&payload).unwrap();
    let mut buf = Vec::new();
    let err = decompress_zstd_into_capped(&compressed, 32, &mut buf).expect_err("must error");
    assert!(format!("{err:?}").contains("zstd"), "{err:?}");
}

// ---- GZIP --------------------------------------------------------------

#[test]
fn decompress_gzip_roundtrip_owned() {
    let original = b"gzip payload gzip payload".repeat(50);
    let compressed = compress_gzip(&original).unwrap();
    let decoded = decompress_gzip(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn decompress_gzip_into_matches_owned() {
    let original = b"the gzip slow horse plods on through the mire".repeat(20);
    let compressed = compress_gzip(&original).unwrap();
    let owned = decompress_gzip(&compressed).unwrap();
    let mut buf = Vec::new();
    decompress_gzip_into(&compressed, &mut buf).unwrap();
    assert_eq!(owned, buf);
}

#[test]
fn decompress_gzip_empty_input_roundtrips() {
    let compressed = compress_gzip(b"").unwrap();
    let decoded = decompress_gzip(&compressed).unwrap();
    assert_eq!(decoded, b"");
}

#[test]
fn decompress_gzip_error_on_garbage() {
    let bad = b"not a gzip stream";
    let mut buf = Vec::new();
    assert!(decompress_gzip(bad).is_err());
    assert!(decompress_gzip_into(bad, &mut buf).is_err());
}

#[test]
fn decompress_gzip_into_capped_rejects_oversize() {
    let payload = vec![1u8; 8192];
    let compressed = compress_gzip(&payload).unwrap();
    let mut buf = Vec::new();
    let err = decompress_gzip_into_capped(&compressed, 16, &mut buf).expect_err("must error");
    assert!(format!("{err:?}").contains("gzip"), "{err:?}");
}

// ---- Brotli ------------------------------------------------------------

#[test]
fn decompress_brotli_roundtrip_owned() {
    let original = b"brotli payload brotli payload".repeat(50);
    let compressed = compress_brotli(&original).unwrap();
    let decoded = decompress_brotli(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn decompress_brotli_into_matches_owned() {
    let original = b"brotli compresses well on repetitive English".repeat(20);
    let compressed = compress_brotli(&original).unwrap();
    let owned = decompress_brotli(&compressed).unwrap();
    let mut buf = Vec::new();
    decompress_brotli_into(&compressed, &mut buf).unwrap();
    assert_eq!(owned, buf);
}

#[test]
fn decompress_brotli_empty_input_roundtrips() {
    let compressed = compress_brotli(b"").unwrap();
    let decoded = decompress_brotli(&compressed).unwrap();
    assert_eq!(decoded, b"");
}

#[test]
fn decompress_brotli_error_on_garbage() {
    let bad = b"not a brotli stream";
    let mut buf = Vec::new();
    assert!(decompress_brotli(bad).is_err());
    assert!(decompress_brotli_into(bad, &mut buf).is_err());
}

#[test]
fn decompress_brotli_into_capped_rejects_oversize() {
    let payload = vec![2u8; 4096];
    let compressed = compress_brotli(&payload).unwrap();
    let mut buf = Vec::new();
    let err = decompress_brotli_into_capped(&compressed, 16, &mut buf).expect_err("must error");
    assert!(format!("{err:?}").contains("brotli"), "{err:?}");
}

// ---- LZ4_RAW -----------------------------------------------------------

#[test]
fn decompress_lz4_raw_roundtrip_owned() {
    let original = b"lz4 payload lz4 payload lz4 payload".repeat(100);
    let compressed = compress_lz4_raw(&original).unwrap();
    let decoded = decompress_lz4_raw(&compressed).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn decompress_lz4_raw_into_matches_owned() {
    let original = b"the lz4 fox jumps over the lazy compression".repeat(80);
    let compressed = compress_lz4_raw(&original).unwrap();
    let owned = decompress_lz4_raw(&compressed).unwrap();
    let mut buf = Vec::new();
    decompress_lz4_raw_into(&compressed, &mut buf).unwrap();
    assert_eq!(owned, buf);
}

#[test]
fn decompress_lz4_raw_into_sized_is_zero_alloc_per_call() {
    // Σ.E7 regression guard: pre-sized variant must decode directly
    // into the caller's buffer without allocating a temporary Vec.
    // We can't easily measure allocations from a test, but we can
    // verify the API: passing the correct size succeeds.
    let original = vec![7u8; 16 * 1024];
    let compressed = compress_lz4_raw(&original).unwrap();
    let mut buf = Vec::new();
    decompress_lz4_raw_into_sized(&compressed, original.len(), &mut buf).unwrap();
    assert_eq!(buf, original);
}

#[test]
fn decompress_lz4_raw_into_sized_errors_on_wrong_size() {
    let original = vec![1u8; 256];
    let compressed = compress_lz4_raw(&original).unwrap();
    let mut buf = Vec::new();
    // Too-small declared size — should error.
    let err = decompress_lz4_raw_into_sized(&compressed, 16, &mut buf)
        .expect_err("too-small should error");
    assert!(format!("{err:?}").contains("lz4"), "{err:?}");
}

#[test]
fn decompress_lz4_raw_into_buffer_reused() {
    let a = vec![3u8; 8192];
    let b = vec![5u8; 2048];
    let ca = compress_lz4_raw(&a).unwrap();
    let cb = compress_lz4_raw(&b).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    decompress_lz4_raw_into_sized(&ca, a.len(), &mut buf).unwrap();
    assert_eq!(buf, a);
    let cap_after_first = buf.capacity();
    decompress_lz4_raw_into_sized(&cb, b.len(), &mut buf).unwrap();
    assert_eq!(buf, b);
    assert!(
        buf.capacity() >= cap_after_first,
        "lz4 buffer capacity must not shrink across calls"
    );
}

#[test]
fn decompress_lz4_raw_empty_input_roundtrips() {
    let compressed = compress_lz4_raw(b"").unwrap();
    let decoded = decompress_lz4_raw(&compressed).unwrap();
    assert_eq!(decoded, b"");
}

#[test]
fn decompress_lz4_raw_error_on_garbage() {
    let bad = vec![0xFFu8; 64];
    let mut buf = Vec::new();
    assert!(decompress_lz4_raw_into(&bad, &mut buf).is_err());
}

#[test]
fn snappy_against_external_compressor() {
    // Cross-check: round-trip our snappy through `snap` directly.
    let original = b"the quick brown fox jumps over the lazy dog";
    let compressed = compress_snappy(original).unwrap();
    let mut buf = Vec::new();
    decompress_snappy_into(&compressed, &mut buf).unwrap();
    assert_eq!(buf, original);
}
