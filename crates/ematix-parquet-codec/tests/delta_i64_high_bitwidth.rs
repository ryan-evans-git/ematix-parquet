//! DELTA_BINARY_PACKED INT64 with bit_width > 32 — exercises the
//! u64-output unpacker added for #69.
//!
//! Builds DELTA streams via parquet-rs's `DeltaBitPackEncoder` whose
//! values force the writer into mini-blocks with bit_width in the
//! 33..=64 range, then confirms our decoder reproduces them exactly.
//! Prior to #69 this path errored with "bit_width > 32 not yet
//! supported".
//!
//! Also includes a direct unit test of `unpack_indices64_into` for
//! bit_width = 64 specifically — the corner case where the mask
//! becomes `u64::MAX` and the per-value accumulator needs to span
//! 9 source bytes.

use ematix_parquet_codec::bitpack::unpack_indices64_into;
use ematix_parquet_codec::delta::decode_delta_i64;
use parquet::data_type::Int64Type;
use parquet::encodings::encoding::{DeltaBitPackEncoder, Encoder};

fn pr_encode(values: &[i64]) -> Vec<u8> {
    let mut enc = DeltaBitPackEncoder::<Int64Type>::new();
    enc.put(values).unwrap();
    enc.flush_buffer().unwrap().to_vec()
}

fn roundtrip(values: &[i64]) {
    let bytes = pr_encode(values);
    let decoded = decode_delta_i64(&bytes).unwrap();
    assert_eq!(decoded.len(), values.len());
    assert_eq!(decoded, values, "decoded vs input mismatch");
}

#[test]
fn i64_wide_deltas_force_bit_width_above_32() {
    // Alternating between 0 and ~2^40 → per-pair delta magnitude
    // ~2^40, forcing the writer into bit_width >= 40. Long enough to
    // span multiple mini-blocks.
    let big: i64 = 1 << 40;
    let v: Vec<i64> = (0..512).map(|i| if i % 2 == 0 { 0 } else { big }).collect();
    roundtrip(&v);
}

#[test]
fn i64_max_magnitude_deltas() {
    // Pair of swings near the i64 limits — produces the largest
    // possible zigzag delta value (~2^63), which the encoder
    // represents in a mini-block with bit_width up to 64.
    let v: Vec<i64> = vec![0, i64::MAX / 2, -(i64::MAX / 2), i64::MAX / 2, 0];
    roundtrip(&v);
}

#[test]
fn i64_random_full_range() {
    // Pseudo-random i64s across the full signed range — the writer
    // will hit a mix of bit_widths per mini-block, including the
    // > 32 ones. Catches accumulator / mask / span-9-bytes bugs.
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let v: Vec<i64> = (0..2048)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed as i64
        })
        .collect();
    roundtrip(&v);
}

#[test]
fn i64_clustered_then_huge_jump() {
    // First half is monotonic small (bit_width ~ low), second half
    // jumps to high values (bit_width forced wide). Exercises the
    // mini-block bit_width switching mid-stream.
    let mut v: Vec<i64> = (0..128i64).collect();
    let base: i64 = 1 << 50;
    v.extend((0..128).map(|i| base + (i as i64) * 17));
    roundtrip(&v);
}

// ---- Direct unpacker corner cases ----------------------------------

#[test]
fn unpack_indices64_bw64_roundtrip() {
    // Build 32 known u64 values, pack them at bit_width = 64
    // (= byte-aligned, 8 bytes per value, no mask trickery), then
    // confirm the unpacker reproduces them.
    let values: Vec<u64> = (0..32)
        .map(|i| 0xFEEDFACE_DEADBEEFu64.wrapping_add(i as u64 * 0x0123456789ABCDEF))
        .collect();
    let mut packed = Vec::with_capacity(values.len() * 8);
    for v in &values {
        packed.extend_from_slice(&v.to_le_bytes());
    }
    let mut out = Vec::new();
    unpack_indices64_into(&packed, 32, 64, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn unpack_indices64_bw0_is_all_zeros() {
    let mut out = Vec::new();
    unpack_indices64_into(&[], 64, 0, &mut out).unwrap();
    assert_eq!(out, vec![0u64; 64]);
}

#[test]
fn unpack_indices64_rejects_bw_above_64() {
    let mut out = Vec::new();
    let r = unpack_indices64_into(&[0; 16], 1, 65, &mut out);
    assert!(r.is_err(), "bit_width > 64 must error");
}

#[test]
fn unpack_indices64_bw57_to_63_span_nine_bytes() {
    // The 58..=63 bit-width range is where a single value crosses
    // 9 source bytes (because start_bit % 8 can be up to 7, and 7 +
    // 57 = 64 which fits in u64, but 7 + 58 = 65 which doesn't —
    // hence the u128 accumulator).
    //
    // Build a known sequence at each width via the same pattern the
    // production code uses (LSB-first bit packing), unpack, and
    // compare. Use 32 values so we hit a full chunk.
    for bw in 57u8..=63 {
        let mask: u64 = if bw == 64 { u64::MAX } else { (1u64 << bw) - 1 };
        // Distinct values within the bit_width's representable range,
        // picked deterministically.
        let values: Vec<u64> = (0..32u64)
            .map(|i| (i.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xCAFEBABE) & mask)
            .collect();
        // Pack using a u128 accumulator (mirrors the unpacker's
        // own math) — straight bit-shift append.
        let total_bits = 32 * bw as usize;
        let total_bytes = total_bits.div_ceil(8);
        let mut packed = vec![0u8; total_bytes];
        let mut acc: u128 = 0;
        let mut bits: u32 = 0;
        let mut byte_ix = 0usize;
        for &v in &values {
            acc |= (v as u128) << bits;
            bits += bw as u32;
            while bits >= 8 {
                packed[byte_ix] = (acc & 0xFF) as u8;
                byte_ix += 1;
                acc >>= 8;
                bits -= 8;
            }
        }
        if bits > 0 {
            packed[byte_ix] = (acc & 0xFF) as u8;
        }

        let mut out = Vec::new();
        unpack_indices64_into(&packed, 32, bw, &mut out).unwrap();
        assert_eq!(out, values, "bw {bw} unpack mismatch");
    }
}
