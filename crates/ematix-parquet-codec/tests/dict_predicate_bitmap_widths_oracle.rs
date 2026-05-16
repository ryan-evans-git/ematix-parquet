//! Π.9b oracle: width-generic predicate-fused decode.
//!
//! `decode_rle_dictionary_predicate_bitmap` must return a packed
//! match-bitmap that is byte-for-byte identical to the
//! materialise-then-filter reference path, across every NEON-
//! specialized width (12, 14, 15, 16, 17, 18) plus a non-NEON width
//! (13, scalar fallback) and the degenerate bw=0 width.
//!
//! Body construction uses `encode_rle_bit_packed_single_run` from
//! the codec, prepended with a bit_width byte — the exact format
//! `decode_rle_dictionary_predicate_bitmap` expects in `body[0..]`.

use ematix_parquet_codec::dict::{
    build_dict_predicate_mask, decode_rle_dictionary_indices,
    decode_rle_dictionary_predicate_bitmap,
};
use ematix_parquet_codec::rle::encode_rle_bit_packed_single_run;

/// Build a dict-page body: `bit_width` byte + RLE/bit-packed indices.
fn build_body(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut body = vec![bit_width];
    body.extend(encode_rle_bit_packed_single_run(indices, bit_width));
    body
}

/// Reference: decode indices via the existing scalar path, then map
/// each through `dict_mask` and pack into a bitmap. The fused decoder
/// must agree byte-for-byte with this.
fn reference_bitmap(body: &[u8], num_values: usize, dict_mask: &[u8]) -> Vec<u8> {
    let idxs = decode_rle_dictionary_indices(body, num_values).unwrap();
    let mut bitmap = vec![0u8; num_values.div_ceil(8)];
    for (row, idx) in idxs.iter().enumerate() {
        let bit = dict_mask[*idx as usize];
        bitmap[row / 8] |= bit << (row % 8);
    }
    bitmap
}

fn check_width(bit_width: u8, num_values: usize, dict_size: usize, mask_seed: u64) {
    assert!(dict_size <= (1usize << bit_width));
    // Indices: deterministic round-robin over the dict.
    let indices: Vec<u32> = (0..num_values).map(|i| (i % dict_size) as u32).collect();
    // dict_mask: pseudo-random bit per dict slot, padded to 1<<bw.
    let mask_len = 1usize << bit_width;
    let mut dict_mask = vec![0u8; mask_len];
    for i in 0..dict_size {
        let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ mask_seed;
        dict_mask[i] = (h >> 32) as u8 & 1;
    }

    let body = build_body(&indices, bit_width);
    let want = reference_bitmap(&body, num_values, &dict_mask);

    let mut got = Vec::new();
    decode_rle_dictionary_predicate_bitmap(&body, num_values, &dict_mask, &mut got).unwrap();
    assert_eq!(
        got, want,
        "width={bit_width} n={num_values} dict_size={dict_size}: bitmap mismatch"
    );
}

#[test]
fn bw12_matches_reference() {
    check_width(12, 4096, 2525, 0xDEAD_BEEF);
}
#[test]
fn bw14_matches_reference() {
    check_width(14, 4096, 12000, 0xCAFE_F00D);
}
#[test]
fn bw15_matches_reference() {
    check_width(15, 4096, 30000, 0xF00D_BEEF);
}
#[test]
fn bw16_matches_reference() {
    check_width(16, 4096, 60000, 0xBABE_FACE);
}
#[test]
fn bw17_matches_reference() {
    check_width(17, 4096, 120_000, 0xFACE_BABE);
}
#[test]
fn bw18_matches_reference() {
    check_width(18, 4096, 250_000, 0xDEAD_F00D);
}

/// Scalar fallback path (no NEON kernel for bw=13).
#[test]
fn bw13_scalar_fallback_matches_reference() {
    check_width(13, 4096, 8000, 0xBEEF_CAFE);
}

/// Tail < 8 rows still works.
#[test]
fn unaligned_tail_widths() {
    for &bw in &[12u8, 14, 15, 16, 17, 18] {
        check_width(bw, 4096 + 5, 1000, 0x1234);
    }
}

/// Predicate selectivity: when no dict slot matches, every bit must
/// be 0; when every slot matches, every bit must be 1.
#[test]
fn all_zero_and_all_one_masks() {
    for &bw in &[12u8, 14, 15, 16, 17, 18] {
        let n = 1024;
        let dict_size = 100;
        let indices: Vec<u32> = (0..n).map(|i| (i % dict_size) as u32).collect();
        let body = build_body(&indices, bw);

        // All-zero mask.
        let mask = vec![0u8; 1usize << bw];
        let mut got = Vec::new();
        decode_rle_dictionary_predicate_bitmap(&body, n, &mask, &mut got).unwrap();
        assert!(
            got.iter().all(|b| *b == 0),
            "bw={bw} all-zero mask: {got:?}"
        );

        // All-one mask within dict (rest stays zero — never indexed).
        let mut mask = vec![0u8; 1usize << bw];
        for slot in mask.iter_mut().take(dict_size) {
            *slot = 1;
        }
        let mut got = Vec::new();
        decode_rle_dictionary_predicate_bitmap(&body, n, &mask, &mut got).unwrap();
        // Every bit set: full bytes 0xFF, partial-tail covers remainder.
        let full_bytes = n / 8;
        for &b in got.iter().take(full_bytes) {
            assert_eq!(b, 0xFF, "bw={bw} all-one mask, byte not 0xFF");
        }
    }
}

/// `build_dict_predicate_mask` builds a mask of exactly 1<<bw bytes
/// with bit set iff predicate matches; padding past dict.len() is 0.
#[test]
fn build_mask_basic() {
    let dict: Vec<i32> = vec![1, 5, 9, 13, 17, 21, 25];
    let mask = build_dict_predicate_mask(&dict, 14, |v| *v % 2 == 1).unwrap();
    assert_eq!(mask.len(), 1 << 14);
    for (i, v) in dict.iter().enumerate() {
        assert_eq!(mask[i], if *v % 2 == 1 { 1 } else { 0 });
    }
    // Padding past dict.len() is zero.
    for slot in mask.iter().skip(dict.len()) {
        assert_eq!(*slot, 0);
    }
}

#[test]
fn build_mask_rejects_oversize_dict() {
    // 1<<2 = 4 — a 5-entry dict can't fit.
    let dict: Vec<i32> = vec![0, 1, 2, 3, 4];
    let r = build_dict_predicate_mask(&dict, 2, |_| true);
    assert!(r.is_err(), "expected error for dict.len() > 1<<bit_width");
}

#[test]
fn build_mask_rejects_zero_bit_width() {
    let dict: Vec<i32> = vec![1];
    let r = build_dict_predicate_mask(&dict, 0, |_| true);
    assert!(r.is_err());
}

/// End-to-end happy path: build the mask via the helper, decode, and
/// confirm the bitmap reproduces the materialise-then-filter result.
#[test]
fn end_to_end_dict_then_filter() {
    let bit_width: u8 = 17;
    let dict: Vec<i64> = (0..100_000).collect();
    let n: usize = 4096;
    let indices: Vec<u32> = (0..n).map(|i| (i * 13 % dict.len()) as u32).collect();
    let body = build_body(&indices, bit_width);

    let pred = |v: &i64| *v % 7 == 0;
    let mask = build_dict_predicate_mask(&dict, bit_width, pred).unwrap();

    let mut got = Vec::new();
    decode_rle_dictionary_predicate_bitmap(&body, n, &mask, &mut got).unwrap();

    // Reference: materialise values then evaluate predicate per row.
    let mut want = vec![0u8; n.div_ceil(8)];
    for (row, &idx) in indices.iter().enumerate() {
        let v = dict[idx as usize];
        if pred(&v) {
            want[row / 8] |= 1 << (row % 8);
        }
    }
    assert_eq!(got, want);
}

/// dict_mask too small for the bit_width is rejected.
#[test]
fn rejects_undersized_dict_mask() {
    let bit_width: u8 = 16;
    let indices: Vec<u32> = (0..64).map(|i| i as u32).collect();
    let body = build_body(&indices, bit_width);
    // Need ≥ 65536; supply 1024.
    let dict_mask = vec![0u8; 1024];
    let mut out = Vec::new();
    let r = decode_rle_dictionary_predicate_bitmap(&body, 64, &dict_mask, &mut out);
    assert!(r.is_err());
}

/// Bit_width = 0 (single-distinct-value page) — every bit equals
/// dict_mask[0]; no body bytes needed past the bit_width prefix.
#[test]
fn bit_width_zero_emits_uniform_bitmap() {
    let body = vec![0u8]; // bw=0, no run bytes
    let n = 100;

    let mask_zero = vec![0u8; 1];
    let mut got = Vec::new();
    decode_rle_dictionary_predicate_bitmap(&body, n, &mask_zero, &mut got).unwrap();
    assert!(got.iter().all(|b| *b == 0));

    let mask_one = vec![1u8; 1];
    let mut got = Vec::new();
    decode_rle_dictionary_predicate_bitmap(&body, n, &mask_one, &mut got).unwrap();
    let full = n / 8;
    let tail = n % 8;
    for &b in got.iter().take(full) {
        assert_eq!(b, 0xFF);
    }
    if tail > 0 {
        let expected_tail = (1u8 << tail) - 1;
        assert_eq!(got[full], expected_tail);
    }
}
