//! Π.13 Tier 2 — AVX2 fused predicate-bitmap kernel correctness.
//!
//! For each bit-width that has a hand-written AVX2 fused decoder
//! (`bitpack_avx2::decode_predicate_bitmap_avx2_bw{12,14,15,16,17,18}`),
//! generate deterministic test data, run both the AVX2 decoder and a
//! scalar reference, assert bit-identical output bitmaps.
//!
//! Only compiled + run on x86_64 with AVX2.  On aarch64 / non-AVX2 x86
//! these tests are cfg'd out — the NEON + portable Rust paths cover
//! those targets.
//!
//! Reference implementation: unpack indices via the portable
//! `bitpack::unpack_indices_into` (which itself dispatches to AVX2 or
//! NEON or scalar for the unpacking step, all already-validated by the
//! existing oracle suites), then do an 8-lane scalar gather + OR-fold
//! exactly mirroring the body of `dict::fused_bitmap_chunk`'s fallback.

#![cfg(target_arch = "x86_64")]

use ematix_parquet_codec::bitpack::unpack_indices_into;
use ematix_parquet_codec::bitpack_avx2::{
    decode_predicate_bitmap_avx2_bw12, decode_predicate_bitmap_avx2_bw14,
    decode_predicate_bitmap_avx2_bw15, decode_predicate_bitmap_avx2_bw16,
    decode_predicate_bitmap_avx2_bw17, decode_predicate_bitmap_avx2_bw18,
};

fn skip_if_no_avx2() -> bool {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("test skipped: CPU lacks AVX2");
        return true;
    }
    false
}

/// Generic LSB-first bit-packer matching Parquet's RLE_DICTIONARY
/// bit-pack wire format.  Round-trip-validates against the existing
/// `bitpack::unpack_indices_into` family.
fn pack_bw_n(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mut bit_offset = 0usize;
    for &v in values {
        for i in 0..bit_width {
            if v & (1 << i) != 0 {
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                out[byte_idx] |= 1 << bit_idx;
            }
            bit_offset += 1;
        }
    }
    out
}

/// Scalar reference: unpack `num_values` indices via the portable
/// path, then 8-lane gather + OR-fold one byte at a time.  Identical
/// shape to `dict::fused_bitmap_chunk`'s fallback body but inlined
/// here so we don't depend on a private function.
fn scalar_decode_predicate_bitmap(
    packed: &[u8],
    num_values: usize,
    bit_width: u8,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) {
    let mut idxs: Vec<u32> = Vec::with_capacity(num_values);
    unpack_indices_into(packed, num_values, bit_width, &mut idxs).unwrap();
    let bytes = num_values.div_ceil(8);
    let out_start = out.len();
    out.resize(out_start + bytes, 0);
    for (i, &idx) in idxs.iter().enumerate() {
        let bit = dict_mask[idx as usize];
        out[out_start + i / 8] |= bit << (i % 8);
    }
}

/// LCG so the test is deterministic + dependency-free.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Drive one width: generate `n` random indices in [0, 1<<bw), pack
/// them, decode via AVX2 + via scalar reference, assert bit-identical
/// bitmap.
fn check_width(bit_width: u8, n: usize, mask_density: u32) {
    let mod_val = 1u32 << bit_width;
    let mut rng = Lcg::new(0x00C0_FFEE_DEAD_BEEF_u64 ^ (bit_width as u64) ^ (n as u64));

    let values: Vec<u32> = (0..n).map(|_| rng.next_u32() % mod_val).collect();
    let packed = pack_bw_n(&values, bit_width);

    let dict_size = 1usize << bit_width;
    let dict_mask: Vec<u8> = (0..dict_size)
        .map(|_| {
            // mask_density out of 100 → 1, else 0
            if rng.next_u32() % 100 < mask_density {
                1
            } else {
                0
            }
        })
        .collect();

    let mut avx2_out: Vec<u8> = Vec::new();
    let mut scalar_out: Vec<u8> = Vec::new();
    match bit_width {
        12 => decode_predicate_bitmap_avx2_bw12(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        14 => decode_predicate_bitmap_avx2_bw14(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        15 => decode_predicate_bitmap_avx2_bw15(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        16 => decode_predicate_bitmap_avx2_bw16(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        17 => decode_predicate_bitmap_avx2_bw17(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        18 => decode_predicate_bitmap_avx2_bw18(&packed, n, &dict_mask, &mut avx2_out).unwrap(),
        _ => panic!("unsupported width"),
    }
    scalar_decode_predicate_bitmap(&packed, n, bit_width, &dict_mask, &mut scalar_out);
    assert_eq!(
        avx2_out, scalar_out,
        "bitmap mismatch at bw={bit_width}, n={n}, density={mask_density}",
    );
}

#[test]
fn bw12_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    // Aligned-to-8, with-tail, dense-mask, sparse-mask.
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(12, n, density);
        }
    }
}

#[test]
fn bw14_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(14, n, density);
        }
    }
}

#[test]
fn bw15_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(15, n, density);
        }
    }
}

#[test]
fn bw16_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(16, n, density);
        }
    }
}

#[test]
fn bw17_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(17, n, density);
        }
    }
}

#[test]
fn bw18_matches_scalar() {
    if skip_if_no_avx2() {
        return;
    }
    for n in [8, 16, 64, 256, 1024, 7, 13, 35, 1003] {
        for density in [10u32, 50, 90] {
            check_width(18, n, density);
        }
    }
}

/// Edge-case: empty input.
#[test]
fn empty_input_returns_empty() {
    if skip_if_no_avx2() {
        return;
    }
    for &bw in &[12u8, 14, 15, 16, 17, 18] {
        let dict_mask = vec![0u8; 1 << bw];
        let mut out: Vec<u8> = Vec::new();
        match bw {
            12 => decode_predicate_bitmap_avx2_bw12(&[], 0, &dict_mask, &mut out).unwrap(),
            14 => decode_predicate_bitmap_avx2_bw14(&[], 0, &dict_mask, &mut out).unwrap(),
            15 => decode_predicate_bitmap_avx2_bw15(&[], 0, &dict_mask, &mut out).unwrap(),
            16 => decode_predicate_bitmap_avx2_bw16(&[], 0, &dict_mask, &mut out).unwrap(),
            17 => decode_predicate_bitmap_avx2_bw17(&[], 0, &dict_mask, &mut out).unwrap(),
            18 => decode_predicate_bitmap_avx2_bw18(&[], 0, &dict_mask, &mut out).unwrap(),
            _ => unreachable!(),
        }
        assert!(out.is_empty(), "expected empty bitmap for bw={bw}");
    }
}
