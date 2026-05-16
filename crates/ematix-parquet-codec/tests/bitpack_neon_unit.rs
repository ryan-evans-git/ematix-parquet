//! NEON bit-unpacker correctness oracle.
//!
//! The NEON kernels must produce byte-for-byte identical output to
//! the scalar const-generic path. We pack a wide variety of inputs
//! at the SIMD-specialized bit widths, unpack with both paths, and
//! assert equality. If either path returns wrong values the test
//! fails immediately rather than masking the bug behind a benchmark.

#![cfg(target_arch = "aarch64")]

use ematix_parquet_codec::bitpack::unpack_indices_into;
use ematix_parquet_codec::bitpack_neon::{
    unpack_indices_into_neon_bw12, unpack_indices_into_neon_bw17,
};

/// Pack `n` u32 values of `bit_width` bits each (LSB-first) into a
/// fresh byte buffer. Mirrors the parquet bit-packed format and the
/// helper used in `bench_unpack`.
fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mask: u64 = if bit_width == 0 {
        0
    } else {
        (1u64 << bit_width) - 1
    };
    for (i, &v) in values.iter().enumerate() {
        let v = (v as u64) & mask;
        let start_bit = i * bit_width as usize;
        let mut byte_idx = start_bit / 8;
        let mut bit_in_byte = (start_bit % 8) as u32;
        let mut remaining = v;
        let mut remaining_bits = bit_width as u32;
        while remaining_bits > 0 {
            let space = 8 - bit_in_byte;
            let take = space.min(remaining_bits);
            let chunk = (remaining & ((1u64 << take) - 1)) as u8;
            out[byte_idx] |= chunk << bit_in_byte;
            remaining >>= take;
            remaining_bits -= take;
            byte_idx += 1;
            bit_in_byte = 0;
        }
    }
    out
}

fn unpack_scalar(packed: &[u8], n: usize, bw: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    unpack_indices_into(packed, n, bw, &mut out).unwrap();
    out
}

fn unpack_neon_bw12(packed: &[u8], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    unpack_indices_into_neon_bw12(packed, n, &mut out).unwrap();
    out
}

#[test]
fn neon_bw12_matches_scalar_known_pattern() {
    // 16 values: 0, 1, 2, ..., 15 — easy to read packed bytes by eye.
    let values: Vec<u32> = (0..16).collect();
    let packed = pack(&values, 12);
    let s = unpack_scalar(&packed, 16, 12);
    let n = unpack_neon_bw12(&packed, 16);
    assert_eq!(s, values);
    assert_eq!(n, s);
}

#[test]
fn neon_bw12_matches_scalar_full_width_values() {
    // Use the full 12-bit range — flushes out off-by-one mask errors.
    let values: Vec<u32> = (0..4096).collect();
    let packed = pack(&values, 12);
    let s = unpack_scalar(&packed, 4096, 12);
    let n = unpack_neon_bw12(&packed, 4096);
    assert_eq!(n, s);
}

#[test]
fn neon_bw12_matches_scalar_random_large() {
    // 1M pseudo-random 12-bit values. Catches loop-boundary bugs.
    let n: usize = 1_000_000;
    let mut seed: u32 = 0xDEADBEEF;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let packed = pack(&values, 12);
    let s = unpack_scalar(&packed, n, 12);
    let nv = unpack_neon_bw12(&packed, n);
    assert_eq!(nv.len(), n);
    assert_eq!(nv, s);
}

#[test]
fn neon_bw12_matches_scalar_short_lengths() {
    // Exercise every tail length 0..32, plus 33, 64, 96 (multi-chunk
    // boundary spots) — the NEON kernel processes 8 at a time, so 1..7
    // and 9..15 etc. all hit different scalar-tail mixes.
    let mut seed: u32 = 0xC0DEC0DE;
    let max_n = 200;
    let values: Vec<u32> = (0..max_n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let packed = pack(&values, 12);
    for n in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 128, 199,
    ] {
        let s = unpack_scalar(&packed, n, 12);
        let nv = unpack_neon_bw12(&packed, n);
        assert_eq!(nv, s, "mismatch at n={n}");
    }
}

fn unpack_neon_bw17(packed: &[u8], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    unpack_indices_into_neon_bw17(packed, n, &mut out).unwrap();
    out
}

#[test]
fn neon_bw17_matches_scalar_known_pattern() {
    // 16 values: 0..16, easy to verify by eye.
    let values: Vec<u32> = (0..16).collect();
    let packed = pack(&values, 17);
    let s = unpack_scalar(&packed, 16, 17);
    let n = unpack_neon_bw17(&packed, 16);
    assert_eq!(s, values);
    assert_eq!(n, s);
}

#[test]
fn neon_bw17_matches_scalar_full_width_values() {
    // Cover the full 17-bit range — flushes off-by-one mask errors.
    let n_total: usize = 1 << 17;
    let values: Vec<u32> = (0..n_total as u32).collect();
    let packed = pack(&values, 17);
    let s = unpack_scalar(&packed, n_total, 17);
    let nv = unpack_neon_bw17(&packed, n_total);
    assert_eq!(nv, s);
}

#[test]
fn neon_bw17_matches_scalar_random_large() {
    let n: usize = 1_000_000;
    let mut seed: u32 = 0xC0DEBEEF;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x1FFFF
        })
        .collect();
    let packed = pack(&values, 17);
    let s = unpack_scalar(&packed, n, 17);
    let nv = unpack_neon_bw17(&packed, n);
    assert_eq!(nv.len(), n);
    assert_eq!(nv, s);
}

#[test]
fn neon_bw17_matches_scalar_short_lengths() {
    let mut seed: u32 = 0xFADE;
    let max_n = 200;
    let values: Vec<u32> = (0..max_n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x1FFFF
        })
        .collect();
    let packed = pack(&values, 17);
    for n in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 128, 199,
    ] {
        let s = unpack_scalar(&packed, n, 17);
        let nv = unpack_neon_bw17(&packed, n);
        assert_eq!(nv, s, "mismatch at n={n}");
    }
}

#[test]
fn neon_bw17_matches_scalar_edge_values() {
    let cases: Vec<Vec<u32>> = vec![
        vec![0; 64],
        vec![0x1FFFF; 64],
        (0..64)
            .map(|i| if i % 2 == 0 { 0x1FFFF } else { 0 })
            .collect(),
        (0..64u32).collect(),
        (0..64u32).rev().collect(),
    ];
    for vals in &cases {
        let packed = pack(vals, 17);
        let s = unpack_scalar(&packed, vals.len(), 17);
        let nv = unpack_neon_bw17(&packed, vals.len());
        assert_eq!(nv, s, "mismatch for {vals:?}");
        assert_eq!(nv, *vals);
    }
}

#[test]
fn neon_bw12_matches_scalar_edge_values() {
    // All zeros, all ones (max 12-bit), alternating, ascending+descending.
    let cases: Vec<Vec<u32>> = vec![
        vec![0; 64],
        vec![0xFFF; 64],
        (0..64)
            .map(|i| if i % 2 == 0 { 0xFFF } else { 0 })
            .collect(),
        (0..64u32).collect(),
        (0..64u32).rev().collect(),
    ];
    for vals in &cases {
        let packed = pack(vals, 12);
        let s = unpack_scalar(&packed, vals.len(), 12);
        let nv = unpack_neon_bw12(&packed, vals.len());
        assert_eq!(nv, s, "mismatch for {vals:?}");
        assert_eq!(nv, *vals);
    }
}
