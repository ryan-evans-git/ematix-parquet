//! Oracle for the hand-rolled Snappy decoder.
//!
//! `decompress_snappy_fast_into` must produce byte-for-byte identical
//! output to `decompress_snappy_into` (which wraps the snap crate)
//! across:
//!   - Tiny inputs (literal-only)
//!   - Heavy back-reference inputs (constant-fill patterns)
//!   - Real parquet-shaped inputs (~80KB random)
//!   - Inputs designed to exercise every tag class:
//!     literal (short ≤60), literal (60-byte run), literal (>60),
//!     copy-1 (1-byte offset, 4-11 byte length),
//!     copy-2 (2-byte offset, 1-64 byte length),
//!     copy-4 (4-byte offset, 1-64 byte length).
//!
//! If either path returns wrong bytes the test fails immediately.

use ematix_parquet_codec::compression::{decompress_snappy_fast_into, decompress_snappy_into};

fn snappy_encode(input: &[u8]) -> Vec<u8> {
    let mut enc = snap::raw::Encoder::new();
    enc.compress_vec(input).unwrap()
}

fn check_roundtrip(input: &[u8]) {
    let compressed = snappy_encode(input);
    let mut a = Vec::new();
    decompress_snappy_into(&compressed, &mut a).unwrap();
    let mut b = Vec::new();
    decompress_snappy_fast_into(&compressed, &mut b).unwrap();
    assert_eq!(a, b, "fast vs snap disagree");
    assert_eq!(b, input, "fast vs original disagree");
}

#[test]
fn empty_input() {
    check_roundtrip(b"");
}

#[test]
fn tiny_literal() {
    check_roundtrip(b"hello world");
}

#[test]
fn short_literal_60_bytes() {
    let v: Vec<u8> = (0..60u8).collect();
    check_roundtrip(&v);
}

#[test]
fn literal_61_bytes_forces_followup_length_byte() {
    let v: Vec<u8> = (0..61u8).collect();
    check_roundtrip(&v);
}

#[test]
fn literal_300_bytes() {
    let v: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
    check_roundtrip(&v);
}

#[test]
fn constant_run_forces_back_references() {
    let v = vec![42u8; 10_000];
    check_roundtrip(&v);
}

#[test]
fn repeating_pattern_forces_copies() {
    // "abc" × 1000 → snappy will encode the second occurrence onward
    // as back-references with small offset.
    let mut v = Vec::with_capacity(3000);
    for _ in 0..1000 {
        v.extend_from_slice(b"abc");
    }
    check_roundtrip(&v);
}

#[test]
fn long_pattern_exercises_copy2() {
    // 100-byte block × 50 → second + later occurrences need >64 byte
    // back-ref offsets, forcing copy-2 tags.
    let block: Vec<u8> = (0..100u8).collect();
    let mut v = Vec::with_capacity(5000);
    for _ in 0..50 {
        v.extend_from_slice(&block);
    }
    check_roundtrip(&v);
}

#[test]
fn realistic_random_data_80kb() {
    // Mimics a parquet page body — random bytes don't compress well,
    // mostly produces literals. Tests the literal hot path's bulk
    // memcpy at typical parquet page size.
    let n = 80 * 1024;
    let mut seed: u64 = 0xC0FFEECAFEDEADBE;
    let v: Vec<u8> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        })
        .collect();
    check_roundtrip(&v);
}

#[test]
fn alternating_run_with_some_compression() {
    // 256 distinct values then repeat, 32 KB.
    let mut v = Vec::with_capacity(32 * 1024);
    for cycle in 0..128 {
        for b in 0..=255u8 {
            v.push(b.wrapping_add(cycle as u8));
        }
    }
    check_roundtrip(&v);
}

#[test]
fn assorted_lengths() {
    for n in [1usize, 7, 16, 31, 32, 60, 61, 63, 64, 127, 128, 255, 256, 1000, 4095, 4096, 65535, 65536, 100_000] {
        let v: Vec<u8> = (0..n as u32).map(|i| (i % 251) as u8).collect();
        check_roundtrip(&v);
    }
}

#[test]
fn capacity_preserved_across_calls() {
    let a = vec![7u8; 4096];
    let b = vec![9u8; 1024];
    let ca = snappy_encode(&a);
    let cb = snappy_encode(&b);
    let mut buf: Vec<u8> = Vec::new();
    decompress_snappy_fast_into(&ca, &mut buf).unwrap();
    assert_eq!(buf, a);
    let cap_after_a = buf.capacity();
    decompress_snappy_fast_into(&cb, &mut buf).unwrap();
    assert_eq!(buf, b);
    assert!(
        buf.capacity() >= cap_after_a,
        "capacity must not shrink"
    );
}

#[test]
fn garbage_input_errors() {
    let bad = b"this is not snappy".to_vec();
    let mut buf = Vec::new();
    assert!(decompress_snappy_fast_into(&bad, &mut buf).is_err());
}
