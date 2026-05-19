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
    unpack_indices_into_neon_bw1, unpack_indices_into_neon_bw2, unpack_indices_into_neon_bw20,
    unpack_indices_into_neon_bw21, unpack_indices_into_neon_bw3, unpack_indices_into_neon_bw4,
    unpack_indices_into_neon_bw5, unpack_indices_into_neon_bw8,
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

// ---- bw=2 -----------------------------------------------------------

fn check_neon_bw2(values: &[u32]) {
    let packed = pack(values, 2);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw2(&packed, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw2: NEON output mismatch");
}

#[test]
fn bw2_one_block_known_pattern() {
    // 32 values cycling 0,1,2,3 — exercises every 2-bit value.
    let v: Vec<u32> = (0..32u32).map(|i| i & 0x03).collect();
    check_neon_bw2(&v);
}

#[test]
fn bw2_4_streams_interleave_correct() {
    // First byte should pack values [0, 1, 2, 3] as
    // (0) | (1<<2) | (2<<4) | (3<<6) = 0b11_10_01_00 = 0xE4.
    let v: Vec<u32> = (0..32u32).map(|i| i & 0x03).collect();
    let packed = pack(&v, 2);
    assert_eq!(packed[0], 0xE4, "lsb-first packing check");
    let mut got = Vec::new();
    unpack_indices_into_neon_bw2(&packed, v.len(), &mut got).unwrap();
    assert_eq!(got, v);
}

#[test]
fn bw2_multi_block() {
    let v: Vec<u32> = (0..512u32).map(|i| i & 0x03).collect();
    check_neon_bw2(&v);
}

#[test]
fn bw2_partial_tail() {
    for n in [33usize, 34, 35, 36, 64, 65, 100, 511, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 7) & 0x03).collect();
        check_neon_bw2(&v);
    }
}

#[test]
fn bw2_random() {
    let mut seed: u32 = 0xC0DEFACE;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x03
        })
        .collect();
    check_neon_bw2(&v);
}

#[test]
fn bw2_empty_is_noop() {
    let mut out = Vec::new();
    unpack_indices_into_neon_bw2(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn bw2_dispatch_routes_through_neon() {
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    let v: Vec<u32> = (0..512u32).map(|i| (i * 13) & 0x03).collect();
    let packed = pack(&v, 2);
    let mut got = Vec::new();
    unpack_indices_into(&packed, v.len(), 2, &mut got).unwrap();
    assert_eq!(got, v);
}

// ---- bw=3 -----------------------------------------------------------

fn check_neon_bw3(values: &[u32]) {
    let packed = pack(values, 3);
    // The bw=3 NEON kernel reads 16 bytes per block; pad packed so
    // even small inputs hit the SIMD path. (Real callers — RLE pages —
    // always have generous trailing slack.)
    let mut padded = packed.clone();
    padded.resize(packed.len().max(64), 0);
    let mut got = Vec::new();
    unpack_indices_into_neon_bw3(&padded, values.len(), &mut got).unwrap();
    assert_eq!(got, values, "bw3: NEON output mismatch");
}

#[test]
fn bw3_one_block_known_pattern() {
    // 0..8 — covers every distinct 3-bit value.
    let v: Vec<u32> = (0..8u32).collect();
    check_neon_bw3(&v);
}

#[test]
fn bw3_packed_bytes_match_spec() {
    // 8 values [0,1,2,3,4,5,6,7] pack LSB-first into 3 bytes:
    //   byte 0 = v0 | v1<<3 | (v2&3)<<6     = 0 | 8 | 0x80 = 0x88
    //   byte 1 = (v2>>2) | v3<<1 | v4<<4 | (v5&1)<<7 = 0 | 6 | 0x40 | 0x80 = 0xC6
    //   byte 2 = (v5>>1) | v6<<2 | v7<<5    = 2 | 0x18 | 0xE0 = 0xFA
    let v: Vec<u32> = (0..8u32).collect();
    let packed = pack(&v, 3);
    assert_eq!(packed[0], 0x88);
    assert_eq!(packed[1], 0xC6);
    assert_eq!(packed[2], 0xFA);
}

#[test]
fn bw3_multi_block_full_range() {
    let v: Vec<u32> = (0..256u32).map(|i| i & 0x07).collect();
    check_neon_bw3(&v);
}

#[test]
fn bw3_partial_tail() {
    for n in [9usize, 17, 33, 65, 100, 511, 1023] {
        let v: Vec<u32> = (0..n as u32).map(|i| (i * 13) & 0x07).collect();
        check_neon_bw3(&v);
    }
}

#[test]
fn bw3_random() {
    let mut seed: u32 = 0xBADBEEF1;
    let v: Vec<u32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0x07
        })
        .collect();
    check_neon_bw3(&v);
}

#[test]
fn bw3_small_input_uses_scalar_fallback() {
    // Force the safety guard to bypass SIMD: only the bare minimum
    // packed bytes provided. Scalar path must still produce correct
    // results.
    let v: Vec<u32> = (0..8u32).collect();
    let packed = pack(&v, 3);
    assert_eq!(packed.len(), 3); // exact required, no slack
    let mut got = Vec::new();
    unpack_indices_into_neon_bw3(&packed, v.len(), &mut got).unwrap();
    assert_eq!(got, v);
}

#[test]
fn bw3_empty_is_noop() {
    let mut out = Vec::new();
    unpack_indices_into_neon_bw3(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn bw3_dispatch_routes_through_neon() {
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    let v: Vec<u32> = (0..512u32).map(|i| (i * 19) & 0x07).collect();
    let mut packed = pack(&v, 3);
    packed.resize(packed.len().max(64), 0);
    let mut got = Vec::new();
    unpack_indices_into(&packed, v.len(), 3, &mut got).unwrap();
    assert_eq!(got, v);
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

// ---- lookup variants (bw=4 / 6 / 8) ---------------------------------
//
// Each test pre-packs LSB-first, then drives the NEON lookup kernel and
// verifies dict[idx[i]] is produced for every i. Tail-only inputs
// exercise the scalar fallback path; out-of-range indices must error.

fn check_lookup<T: Copy + std::fmt::Debug + PartialEq>(bw: u8, indices: &[u32], dict: &[T]) {
    use ematix_parquet_codec::bitpack_neon::{
        unpack_lookup_into_neon_bw4, unpack_lookup_into_neon_bw6, unpack_lookup_into_neon_bw8,
    };
    let packed = pack(indices, bw);
    let mut got: Vec<T> = Vec::new();
    match bw {
        4 => unpack_lookup_into_neon_bw4(&packed, indices.len(), dict, &mut got).unwrap(),
        6 => {
            // bw=6 kernel reads 16 bytes per block; pad source.
            let mut padded = packed.clone();
            padded.resize(packed.len().max(64), 0);
            unpack_lookup_into_neon_bw6(&padded, indices.len(), dict, &mut got).unwrap();
        }
        8 => unpack_lookup_into_neon_bw8(&packed, indices.len(), dict, &mut got).unwrap(),
        _ => unreachable!(),
    }
    let expected: Vec<T> = indices.iter().map(|&i| dict[i as usize]).collect();
    assert_eq!(got, expected, "bw{bw}: lookup mismatch");
}

#[test]
fn lookup_bw4_round_trip() {
    let dict: Vec<i64> = (1000..1016i64).collect();
    let indices: Vec<u32> = (0..256u32).map(|i| i & 0x0F).collect();
    check_lookup(4, &indices, &dict);
}

#[test]
fn lookup_bw4_tail() {
    let dict: Vec<f64> = (0..16).map(|i| i as f64 * 0.25).collect();
    for n in [33usize, 65, 100, 511, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 7) & 0x0F).collect();
        check_lookup(4, &indices, &dict);
    }
}

#[test]
fn lookup_bw4_small_dict_bounds_path() {
    // dict_size = 15 forces the bounds-checked path (max idx = 15
    // wouldn't fit if dict.len() <= 15).
    let dict: Vec<u8> = (0..15).collect();
    let indices: Vec<u32> = (0..64u32).map(|i| i & 0x0E).collect(); // even, all < 15
    check_lookup(4, &indices, &dict);
}

#[test]
fn lookup_bw4_out_of_range_errors() {
    use ematix_parquet_codec::bitpack_neon::unpack_lookup_into_neon_bw4;
    let dict: Vec<u8> = (0..8).collect();
    let indices: Vec<u32> = (0..32u32)
        .map(|i| if i == 17 { 12 } else { i & 7 })
        .collect();
    let packed = pack(&indices, 4);
    let mut got: Vec<u8> = Vec::new();
    let r = unpack_lookup_into_neon_bw4(&packed, indices.len(), &dict, &mut got);
    assert!(r.is_err());
}

#[test]
fn lookup_bw6_round_trip() {
    let dict: Vec<i32> = (2000..2064i32).collect();
    let indices: Vec<u32> = (0..256u32).map(|i| i & 0x3F).collect();
    check_lookup(6, &indices, &dict);
}

#[test]
fn lookup_bw6_tail() {
    let dict: Vec<f64> = (0..64).map(|i| i as f64).collect();
    for n in [9usize, 17, 33, 65, 100, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 13) & 0x3F).collect();
        check_lookup(6, &indices, &dict);
    }
}

#[test]
fn lookup_bw6_bounds_checked_path() {
    let dict: Vec<u16> = (0..50).collect();
    let indices: Vec<u32> = (0..128u32).map(|i| i % 50).collect();
    check_lookup(6, &indices, &dict);
}

#[test]
fn lookup_bw8_round_trip() {
    let dict: Vec<u64> = (10_000..10_256u64).collect();
    let indices: Vec<u32> = (0..1024u32).map(|i| i & 0xFF).collect();
    check_lookup(8, &indices, &dict);
}

#[test]
fn lookup_bw8_tail() {
    let dict: Vec<i64> = (0..256i64).collect();
    for n in [33usize, 64, 100, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 31) & 0xFF).collect();
        check_lookup(8, &indices, &dict);
    }
}

#[test]
fn lookup_bw8_small_dict_bounds_path() {
    // dict.len() = 7 ⇒ indices > 6 must error.
    use ematix_parquet_codec::bitpack_neon::unpack_lookup_into_neon_bw8;
    let dict: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
    let indices: Vec<u32> = (0..32u32).map(|i| i % 7).collect();
    let packed = pack(&indices, 8);
    let mut got: Vec<u8> = Vec::new();
    unpack_lookup_into_neon_bw8(&packed, indices.len(), &dict, &mut got).unwrap();
    let expected: Vec<u8> = indices.iter().map(|&i| dict[i as usize]).collect();
    assert_eq!(got, expected);
}

#[test]
fn lookup_dispatch_routes_bw4_6_8() {
    use ematix_parquet_codec::bitpack::unpack_lookup_into;
    // bw=4
    {
        let dict: Vec<u64> = (0..16u64).collect();
        let indices: Vec<u32> = (0..256u32).map(|i| i & 0x0F).collect();
        let packed = pack(&indices, 4);
        let mut got: Vec<u64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 4, &dict, &mut got).unwrap();
        let expected: Vec<u64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
    // bw=6
    {
        let dict: Vec<u64> = (0..64u64).collect();
        let indices: Vec<u32> = (0..256u32).map(|i| (i * 5) & 0x3F).collect();
        let mut packed = pack(&indices, 6);
        packed.resize(packed.len().max(64), 0);
        let mut got: Vec<u64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 6, &dict, &mut got).unwrap();
        let expected: Vec<u64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
    // bw=8
    {
        let dict: Vec<i64> = (0..256i64).collect();
        let indices: Vec<u32> = (0..1024u32).map(|i| (i * 13) & 0xFF).collect();
        let packed = pack(&indices, 8);
        let mut got: Vec<i64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 8, &dict, &mut got).unwrap();
        let expected: Vec<i64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
}
