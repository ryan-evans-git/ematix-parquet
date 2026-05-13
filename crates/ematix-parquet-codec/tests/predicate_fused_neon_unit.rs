//! Predicate-fused NEON decode oracle (bw=12).
//!
//! Pipeline under test:
//!   bit-packed indices (bw=12)  +  dict_mask (u8: 1 = match, 0 = miss)
//!   →  packed row bitmap (1 byte per 8 rows; bit i of byte k = row 8k+i)
//!
//! For each row we conceptually do `bitmap[row] = dict_mask[idx[row]]`,
//! but the NEON kernel fuses the bit-unpack and the gather + bit-pack
//! into one tight loop with no intermediate Vec<u32> / Vec<bool>.
//!
//! Correctness: compare to a hand-written scalar reference that does
//! the same logical operation on a Vec<u32> of decoded indices.

#![cfg(target_arch = "aarch64")]

use ematix_parquet_codec::bitpack::unpack_indices_into;
use ematix_parquet_codec::bitpack_neon::decode_predicate_bitmap_neon_bw12;

fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let bw_mask: u64 = (1u64 << bit_width) - 1;
    for (i, &v) in values.iter().enumerate() {
        let v = (v as u64) & bw_mask;
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

/// Pad to ≥ 4096 entries — the NEON kernel requires this so per-lane
/// gathers can use 12-bit indices without bounds checks.
fn padded_dict_mask(matches: &[u32]) -> Vec<u8> {
    let mut m = vec![0u8; 4096];
    for &idx in matches {
        m[idx as usize] = 1;
    }
    m
}

/// Reference: decode indices via scalar unpack, then per-row gather
/// dict_mask, then pack bitmap.
fn reference_bitmap(packed: &[u8], n: usize, dict_mask: &[u8]) -> Vec<u8> {
    let mut indices: Vec<u32> = Vec::with_capacity(n);
    unpack_indices_into(packed, n, 12, &mut indices).unwrap();
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    for (row, idx) in indices.into_iter().enumerate() {
        let bit = dict_mask[idx as usize];
        bitmap[row / 8] |= bit << (row % 8);
    }
    bitmap
}

#[test]
fn fused_matches_reference_known_pattern() {
    // 8 values [3, 7, 3, 0, 7, 1, 3, 7]. Match: dict_mask[3]=1, dict_mask[7]=1.
    let values: Vec<u32> = vec![3, 7, 3, 0, 7, 1, 3, 7];
    let packed = pack(&values, 12);
    let dict_mask = padded_dict_mask(&[3, 7]);

    let mut bitmap = Vec::new();
    decode_predicate_bitmap_neon_bw12(&packed, 8, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&packed, 8, &dict_mask);
    assert_eq!(bitmap, expected);
    // Sanity: row 0 = 1, row 1 = 1, row 2 = 1, row 3 = 0, row 4 = 1,
    // row 5 = 0, row 6 = 1, row 7 = 1 → byte = 0b1101_0111 = 0xD7.
    assert_eq!(bitmap, vec![0xD7u8]);
}

#[test]
fn fused_matches_reference_multi_byte() {
    // 24 values across 3 bitmap bytes.
    let values: Vec<u32> = (0..24).map(|i| (i * 7) % 4096).collect();
    let packed = pack(&values, 12);
    // Match indices that are even.
    let matches: Vec<u32> = (0..4096).filter(|i| i % 2 == 0).collect();
    let dict_mask = padded_dict_mask(&matches);

    let mut bitmap = Vec::new();
    decode_predicate_bitmap_neon_bw12(&packed, 24, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&packed, 24, &dict_mask);
    assert_eq!(bitmap, expected);
    assert_eq!(bitmap.len(), 3);
}

#[test]
fn fused_matches_reference_random_million() {
    let n: usize = 1_000_000;
    let mut seed: u32 = 0xC0DEFACE;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let packed = pack(&values, 12);
    // Q14-shape: match ~1% of dict entries (one narrow window).
    let matches: Vec<u32> = (1000..1040).collect();
    let dict_mask = padded_dict_mask(&matches);

    let mut bitmap = Vec::new();
    decode_predicate_bitmap_neon_bw12(&packed, n, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&packed, n, &dict_mask);
    assert_eq!(bitmap.len(), n.div_ceil(8));
    assert_eq!(bitmap, expected);

    // Also sanity-check the match count vs naive scan.
    let match_count: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
    let expected_count = values
        .iter()
        .filter(|v| **v >= 1000 && **v < 1040)
        .count();
    assert_eq!(match_count, expected_count);
}

#[test]
fn fused_matches_reference_various_lengths() {
    let mut seed: u32 = 0xFADEC0DE;
    let max_n = 300;
    let values: Vec<u32> = (0..max_n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let packed = pack(&values, 12);
    let dict_mask = padded_dict_mask(&(0..4096).filter(|i| i % 3 == 0).collect::<Vec<_>>());

    for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 200, 299] {
        let mut bitmap = Vec::new();
        decode_predicate_bitmap_neon_bw12(&packed, n, &dict_mask, &mut bitmap).unwrap();
        let expected = reference_bitmap(&packed, n, &dict_mask);
        assert_eq!(bitmap, expected, "mismatch at n={n}");
    }
}

#[test]
fn fused_all_match_returns_all_ones() {
    let n = 64;
    let values: Vec<u32> = (0..n as u32).map(|i| i & 0xFFF).collect();
    let packed = pack(&values, 12);
    let dict_mask = vec![1u8; 4096];

    let mut bitmap = Vec::new();
    decode_predicate_bitmap_neon_bw12(&packed, n, &dict_mask, &mut bitmap).unwrap();
    assert_eq!(bitmap, vec![0xFFu8; 8]);
}

#[test]
fn fused_no_match_returns_all_zeros() {
    let n = 64;
    let values: Vec<u32> = (0..n as u32).map(|i| i & 0xFFF).collect();
    let packed = pack(&values, 12);
    let dict_mask = vec![0u8; 4096];

    let mut bitmap = Vec::new();
    decode_predicate_bitmap_neon_bw12(&packed, n, &dict_mask, &mut bitmap).unwrap();
    assert_eq!(bitmap, vec![0u8; 8]);
}

#[test]
fn fused_requires_padded_dict_mask() {
    let values: Vec<u32> = vec![3, 7, 3, 0];
    let packed = pack(&values, 12);
    let short = vec![1u8; 10]; // way too small
    let mut bitmap = Vec::new();
    let r = decode_predicate_bitmap_neon_bw12(&packed, 4, &short, &mut bitmap);
    assert!(r.is_err());
}
