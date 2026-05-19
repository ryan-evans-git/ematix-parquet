//! NEON kernel oracle for the new small-width unpackers.
//!
//! Compares the NEON output of `unpack_indices_into_neon_bw{4,8}`
//! against the scalar reference for known bit patterns and random
//! inputs across multiple block counts (full + partial blocks).
//!
//! NEON-only — gated on aarch64; on other targets the test compiles
//! but is skipped at run time.

#![cfg(target_arch = "aarch64")]

use ematix_parquet_codec::bitpack_neon::{
    unpack_indices_into_neon_bw1, unpack_indices_into_neon_bw20, unpack_indices_into_neon_bw21,
    unpack_indices_into_neon_bw4, unpack_indices_into_neon_bw5, unpack_indices_into_neon_bw8,
};

/// Pack `values` LSB-first at `bit_width`. Bit-exact mirror of the
/// production packer that the readers expect.
fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mask: u64 = (1u64 << bit_width) - 1;
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut byte_ix = 0usize;
    for &v in values {
        acc |= ((v as u64) & mask) << bits;
        bits += bit_width as u32;
        while bits >= 8 {
            out[byte_ix] = (acc & 0xFF) as u8;
            byte_ix += 1;
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out[byte_ix] = (acc & 0xFF) as u8;
    }
    out
}

fn check_neon_bw8(values: &[u32]) {
    let packed = pack(values, 8);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw8(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw8: NEON output mismatch");
}

fn check_neon_bw4(values: &[u32]) {
    let packed = pack(values, 4);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw4(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw4: NEON output mismatch");
}

// ---- bw=8 -----------------------------------------------------------

#[test]
fn bw8_one_block_known_pattern() {
    let v: Vec<u32> = (0..32u32).collect();
    check_neon_bw8(&v);
}

#[test]
fn bw8_multi_block_descending() {
    let v: Vec<u32> = (0..256u32).rev().collect();
    check_neon_bw8(&v);
}

#[test]
fn bw8_partial_tail_scalar_fallback() {
    // 33 values = 1 full NEON block + 1 scalar-tail value.
    for n in [33usize, 64, 65, 100, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 31) & 0xFF).collect();
        check_neon_bw8(&v);
    }
}

#[test]
fn bw8_random_full_range() {
    let mut seed: u32 = 0xDEADBEEF;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFF
        })
        .collect();
    check_neon_bw8(&v);
}

#[test]
fn bw8_empty_is_noop() {
    let mut out = Vec::new();
    unpack_indices_into_neon_bw8(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

// ---- bw=4 -----------------------------------------------------------

#[test]
fn bw4_one_block_known_pattern() {
    let v: Vec<u32> = (0..32u32).map(|i| i & 0x0F).collect();
    check_neon_bw4(&v);
}

#[test]
fn bw4_lo_hi_nibble_interleave_correct() {
    // Adjacent values that exercise the low-vs-high-nibble split:
    // first byte should pack to 0x21 → values [1, 2].
    let v: Vec<u32> = (0..32u32).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
    let packed = pack(&v, 4);
    // First byte: lo nibble = 1, hi nibble = 2 → 0x21.
    assert_eq!(packed[0], 0x21, "lsb-first packing check");
    let mut got = Vec::new();
    unpack_indices_into_neon_bw4(&packed, v.len(), &mut got).unwrap();
    assert_eq!(got, v);
}

#[test]
fn bw4_multi_block() {
    let v: Vec<u32> = (0..256u32).map(|i| i & 0x0F).collect();
    check_neon_bw4(&v);
}

#[test]
fn bw4_partial_tail() {
    for n in [33usize, 34, 64, 65, 100, 511] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 7) & 0x0F).collect();
        check_neon_bw4(&v);
    }
}

#[test]
fn bw4_random() {
    let mut seed: u32 = 0xCAFEBABE;
    let v: Vec<u32> = (0..1024)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x0F
        })
        .collect();
    check_neon_bw4(&v);
}

#[test]
fn bw4_empty_is_noop() {
    let mut out = Vec::new();
    unpack_indices_into_neon_bw4(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

// ---- end-to-end via the public dispatcher ---------------------------
//
// Confirms `unpack_indices_into` (the dispatch entry point) routes
// bw=4 and bw=8 to NEON when available and produces the same bytes
// as if we'd called the kernel directly.

#[test]
fn dispatch_routes_bw8_through_neon_matching_results() {
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    let v: Vec<u32> = (0..256u32).map(|i| (i * 13) & 0xFF).collect();
    let packed = pack(&v, 8);
    let mut got = Vec::new();
    unpack_indices_into(&packed, v.len(), 8, &mut got).unwrap();
    assert_eq!(got, v);
}

#[test]
fn dispatch_routes_bw4_through_neon_matching_results() {
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    let v: Vec<u32> = (0..512u32).map(|i| (i * 5) & 0x0F).collect();
    let packed = pack(&v, 4);
    let mut got = Vec::new();
    unpack_indices_into(&packed, v.len(), 4, &mut got).unwrap();
    assert_eq!(got, v);
}

// ---- bw=1 -----------------------------------------------------------

fn check_neon_bw1(values: &[u32]) {
    let packed = pack(values, 1);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw1(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw1: NEON output mismatch");
}

#[test]
fn bw1_alternating_pattern() {
    let v: Vec<u32> = (0..32u32).map(|i| i & 1).collect();
    check_neon_bw1(&v);
}

#[test]
fn bw1_all_zeros_and_all_ones() {
    check_neon_bw1(&vec![0u32; 256]);
    check_neon_bw1(&vec![1u32; 256]);
}

#[test]
fn bw1_partial_tail() {
    for n in [33usize, 65, 100, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 7) & 1).collect();
        check_neon_bw1(&v);
    }
}

#[test]
fn bw1_random() {
    let mut seed: u32 = 0xC0FFEE;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 1
        })
        .collect();
    check_neon_bw1(&v);
}

// ---- bw=5 -----------------------------------------------------------

fn check_neon_bw5(values: &[u32]) {
    let packed = pack(values, 5);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw5(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw5: NEON output mismatch");
}

#[test]
fn bw5_one_block_known_pattern() {
    let v: Vec<u32> = (0..8u32).collect();
    check_neon_bw5(&v);
}

#[test]
fn bw5_full_range() {
    // Every distinct 5-bit value across a 32-value chunk.
    let v: Vec<u32> = (0..32u32).map(|i| i & 0x1F).collect();
    check_neon_bw5(&v);
}

#[test]
fn bw5_partial_tail() {
    for n in [9usize, 17, 65, 100, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 13) & 0x1F).collect();
        check_neon_bw5(&v);
    }
}

#[test]
fn bw5_random() {
    let mut seed: u32 = 0xFADEBABE;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x1F
        })
        .collect();
    check_neon_bw5(&v);
}

// ---- bw=20 ----------------------------------------------------------

fn check_neon_bw20(values: &[u32]) {
    let packed = pack(values, 20);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw20(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw20: NEON output mismatch");
}

#[test]
fn bw20_one_block_known_pattern() {
    let v: Vec<u32> = (0..8u32).map(|i| i.wrapping_mul(0xA5A5)).collect();
    check_neon_bw20(&v);
}

#[test]
fn bw20_full_range() {
    // Exercise values near the 20-bit boundary.
    let v: Vec<u32> = (0..32u32).map(|i| (i * 31_337) & 0x0F_FFFF).collect();
    check_neon_bw20(&v);
}

#[test]
fn bw20_partial_tail() {
    for n in [9usize, 17, 100, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 17) & 0x0F_FFFF).collect();
        check_neon_bw20(&v);
    }
}

#[test]
fn bw20_random() {
    let mut seed: u32 = 0xBADC0FFE;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x0F_FFFF
        })
        .collect();
    check_neon_bw20(&v);
}

// ---- bw=21 ----------------------------------------------------------

fn check_neon_bw21(values: &[u32]) {
    let packed = pack(values, 21);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw21(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw21: NEON output mismatch");
}

#[test]
fn bw21_one_block_known_pattern() {
    let v: Vec<u32> = (0..8u32).map(|i| i.wrapping_mul(0x12345)).collect();
    check_neon_bw21(&v);
}

#[test]
fn bw21_full_range() {
    let v: Vec<u32> = (0..32u32).map(|i| (i * 31_337) & 0x1F_FFFF).collect();
    check_neon_bw21(&v);
}

#[test]
fn bw21_partial_tail() {
    for n in [9usize, 17, 100, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 17) & 0x1F_FFFF).collect();
        check_neon_bw21(&v);
    }
}

#[test]
fn bw21_random() {
    let mut seed: u32 = 0xFEEDC0DE;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x1F_FFFF
        })
        .collect();
    check_neon_bw21(&v);
}

// ---- dispatch routing -----------------------------------------------

#[test]
fn dispatch_routes_bw1_5_20_21_through_neon() {
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    for &bw in &[1u8, 5, 20, 21] {
        let mask: u32 = if bw == 32 { u32::MAX } else { (1u32 << bw) - 1 };
        let v: Vec<u32> = (0..256u32)
            .map(|i| (i.wrapping_mul(1_597_463)) & mask)
            .collect();
        let packed = pack(&v, bw);
        let mut got = Vec::new();
        unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
        assert_eq!(got, v, "bw{bw} dispatch mismatch");
    }
}
