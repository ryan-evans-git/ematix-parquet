//! Parquet's RLE / bit-packed hybrid encoding.
//!
//! Spec: https://github.com/apache/parquet-format/blob/master/Encodings.md
//!
//! Wire format: a sequence of runs, each prefixed by a single
//! unsigned varint header. LSB of the header chooses between:
//!
//!   header = (run_len << 1)          — RLE run; one repeated value
//!   header = (n_groups_of_8 << 1)|1  — bit-packed run; 8·n_groups values
//!
//! RLE values are stored as `ceil(bit_width / 8)` bytes, little-endian.
//! Bit-packed values are stored LSB-first within each byte, contiguous,
//! `bit_width` bits per value. Writers always pack multiples of 8
//! values, padding the last group; readers stop at `num_values`.
//!
//! `bit_width = 0` is the degenerate case (only one possible value, 0);
//! no value bytes are written.

use ematix_parquet_format::compact::{read_uvarint, Cursor};

use crate::error::{CodecError, Result};

/// Smallest `bit_width` that holds every value in `0..dict_len`.
/// `dict_len <= 1` → 0 (only the value 0 is possible).
pub fn min_bit_width_for_dict(dict_len: usize) -> u8 {
    if dict_len <= 1 {
        0
    } else {
        // ceil(log2(dict_len)) for dict_len >= 2.
        (32 - ((dict_len as u32) - 1).leading_zeros()) as u8
    }
}

/// Encode `indices` as a single bit-packed run of the RLE/bit-pack
/// hybrid format. Padding to a multiple of 8 values is handled here;
/// the reader stops at `num_values` (carried by the data-page header)
/// and never sees the padding.
///
/// This is the minimum-viable encoder — it does not coalesce repeated
/// values into RLE runs. The decoder happily accepts a single
/// bit-packed run, and parquet-rs reads it back identically. RLE
/// coalescing is a size optimisation we can layer on later when
/// benchmarks show value.
///
/// Returns the body bytes without the leading bit-width byte —
/// callers prepend that when building the data-page body. For the
/// `bit_width == 0` degenerate case, the body is a single uvarint
/// `(N << 1) | 0` (one RLE run of zeros) with no value bytes.
pub fn encode_rle_bit_packed_single_run(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut out = Vec::new();
    if indices.is_empty() {
        return out;
    }
    if bit_width == 0 {
        // RLE run of `len` zeros; value_bytes = 0.
        write_uvarint(&mut out, (indices.len() as u64) << 1);
        return out;
    }
    if bit_width > 32 {
        // Dictionary indices fit in u32; widths > 32 would mean
        // we miscomputed. Cap and let the test layer catch it.
        return out;
    }

    let num_groups = indices.len().div_ceil(8);
    let header: u64 = ((num_groups as u64) << 1) | 1;
    write_uvarint(&mut out, header);

    // Pad to multiple of 8 with zeros, then bit-pack via the
    // shared LSB-first packer.
    let mut padded = Vec::with_capacity(num_groups * 8);
    padded.extend_from_slice(indices);
    padded.resize(num_groups * 8, 0);
    pack_lsb_first(&mut out, &padded, bit_width);
    out
}

/// LEB128 (uvarint) write used by the RLE/bit-pack encoder for run
/// headers. Matches what `read_uvarint` consumes on the decode side.
fn write_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Pack `indices.len()` values (must be a multiple of 8) into `out`,
/// LSB-first, `bit_width` bits per value. Used by both the
/// single-run encoder and the smart run-coalescing encoder.
fn pack_lsb_first(out: &mut Vec<u8>, indices: &[u32], bit_width: u8) {
    debug_assert_eq!(indices.len() % 8, 0);
    let body_len = indices.len() / 8 * bit_width as usize;
    let body_start = out.len();
    out.resize(body_start + body_len, 0);

    let bw = bit_width as usize;
    for (i, &idx) in indices.iter().enumerate() {
        let bit_pos = i * bw;
        let mut byte_ix = body_start + (bit_pos / 8);
        let mut shift = bit_pos % 8;
        let mut remaining = bw;
        let mut value = idx as u64;
        while remaining > 0 {
            let take = (8 - shift).min(remaining);
            let take_mask = (1u64 << take) - 1;
            let chunk = (value & take_mask) as u8;
            out[byte_ix] |= chunk << shift;
            value >>= take;
            remaining -= take;
            byte_ix += 1;
            shift = 0;
        }
    }
}

/// Smart RLE/bit-pack encoder: emits RLE runs for stretches of
/// repeated indices, bit-packed runs for the rest. Output is the
/// run-bytes (no leading bit-width byte — caller prepends that
/// when building the data-page body, matching
/// `encode_rle_bit_packed_single_run`).
///
/// **Threshold.** A value run of length ≥ 8 always wins over
/// bit-packing the same stretch (RLE costs 1 + ceil(bw/8) bytes vs
/// bit-pack's bw bytes per group of 8, and the ratio gets
/// dramatically better as run length grows). Shorter runs land in
/// the bit-pack accumulator.
///
/// **Alignment.** Bit-packed runs must be multiples of 8 values
/// per the spec. When an RLE-worthy value run interrupts a
/// non-aligned bit-pack accumulator (e.g., 5 values pending),
/// we borrow `8 - 5 = 3` values from the RLE run to align the
/// bit-pack, then RLE the remaining `run_len - 3` values. The
/// borrow only happens when the borrowed run is *still* RLE-worthy
/// after the steal — otherwise the values join the bit-pack
/// accumulator instead. Only the final bit-pack run in the stream
/// is allowed to be zero-padded (since the decoder stops at
/// `num_values` and never sees the padding).
///
/// For all-bit-packed input (every run < 8) this produces the same
/// bytes as `encode_rle_bit_packed_single_run`. For input with
/// long runs, it can be 5-100× smaller — e.g., a single value
/// repeated 10000 times encodes as ~3 bytes (1-byte uvarint header
/// + ceil(bw/8) value bytes) regardless of bit_width.
pub fn encode_rle_bit_packed(indices: &[u32], bit_width: u8) -> Vec<u8> {
    const RLE_THRESHOLD: usize = 8;
    let mut out = Vec::new();
    if indices.is_empty() {
        return out;
    }
    if bit_width == 0 {
        // Degenerate width — every value is 0 — single RLE run.
        write_uvarint(&mut out, (indices.len() as u64) << 1);
        return out;
    }
    if bit_width > 32 {
        return out;
    }

    let value_bytes = (bit_width as usize + 7) / 8;
    let mut pending: Vec<u32> = Vec::new();

    let mut i = 0usize;
    while i < indices.len() {
        let v = indices[i];
        let mut j = i + 1;
        while j < indices.len() && indices[j] == v {
            j += 1;
        }
        let run_len = j - i;

        // To emit RLE we need pending.len() to be a multiple of 8.
        // If it isn't, we can borrow `borrow` values from this run
        // to align — but only if the residual run is still at
        // least RLE_THRESHOLD long.
        let pending_mod = pending.len() % 8;
        let borrow = if pending_mod == 0 { 0 } else { 8 - pending_mod };

        if run_len >= RLE_THRESHOLD + borrow {
            // Borrow → flush aligned bit-pack → emit RLE for the
            // residual.
            for _ in 0..borrow {
                pending.push(v);
            }
            let residual = run_len - borrow;

            if !pending.is_empty() {
                let num_groups = pending.len() / 8; // already aligned
                write_uvarint(&mut out, ((num_groups as u64) << 1) | 1);
                pack_lsb_first(&mut out, &pending, bit_width);
                pending.clear();
            }

            write_uvarint(&mut out, (residual as u64) << 1);
            let value = v as u64;
            for b in 0..value_bytes {
                out.push((value >> (b * 8)) as u8);
            }
        } else {
            // Either the run is short, or after borrowing alignment
            // wouldn't leave enough to RLE — append the whole run
            // to the bit-pack accumulator.
            pending.reserve(run_len);
            for _ in 0..run_len {
                pending.push(v);
            }
        }
        i = j;
    }

    // Final flush — this is the only place padding is allowed,
    // because the decoder stops at num_values and never sees the
    // padding bytes.
    if !pending.is_empty() {
        let num_groups = pending.len().div_ceil(8);
        write_uvarint(&mut out, ((num_groups as u64) << 1) | 1);
        pending.resize(num_groups * 8, 0);
        pack_lsb_first(&mut out, &pending, bit_width);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(indices: &[u32], bit_width: u8) {
        let body = encode_rle_bit_packed_single_run(indices, bit_width);
        let decoded = decode_rle_bit_packed(&body, bit_width, indices.len()).unwrap();
        let want: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
        assert_eq!(
            decoded,
            want,
            "bit_width={bit_width}, len={}",
            indices.len()
        );
    }

    #[test]
    fn empty_indices_yield_empty_body() {
        let body = encode_rle_bit_packed_single_run(&[], 8);
        assert!(body.is_empty());
    }

    #[test]
    fn bit_width_0_rle_run_of_zeros() {
        // dict_len = 1 → bit_width = 0 → every index is 0. The body
        // is just the run header; no value bytes.
        let indices = vec![0u32; 13];
        roundtrip(&indices, 0);
    }

    #[test]
    fn bit_width_1_alternating() {
        let indices: Vec<u32> = (0..16).map(|i| i & 1).collect();
        roundtrip(&indices, 1);
    }

    #[test]
    fn bit_width_3_padding_partial_group() {
        // 5 indices forces padding up to 8 — exercises the
        // partial-group write path.
        let indices = vec![5u32, 0, 7, 3, 1];
        roundtrip(&indices, 3);
    }

    #[test]
    fn bit_width_4_full_groups() {
        let indices: Vec<u32> = (0..32).map(|i| i & 0xF).collect();
        roundtrip(&indices, 4);
    }

    #[test]
    fn bit_width_8_byte_aligned() {
        let indices: Vec<u32> = (0..200).map(|i| (i * 17) as u32 & 0xFF).collect();
        roundtrip(&indices, 8);
    }

    #[test]
    fn bit_width_12_spans_bytes() {
        let indices: Vec<u32> = (0..150).map(|i| (i * 31) as u32 & 0xFFF).collect();
        roundtrip(&indices, 12);
    }

    #[test]
    fn bit_width_17_unaligned() {
        let indices: Vec<u32> = (0..73).map(|i| (i * 1009) as u32 & 0x1FFFF).collect();
        roundtrip(&indices, 17);
    }

    #[test]
    fn bit_width_24_wide() {
        let indices: Vec<u32> = (0..50).map(|i| (i * 7919) as u32 & 0xFF_FFFF).collect();
        roundtrip(&indices, 24);
    }

    #[test]
    fn bit_width_32_max() {
        let indices: Vec<u32> = vec![0, u32::MAX, 1234567890, 0xDEAD_BEEF, 1];
        roundtrip(&indices, 32);
    }

    fn smart_roundtrip(indices: &[u32], bit_width: u8) {
        let body = encode_rle_bit_packed(indices, bit_width);
        let decoded = decode_rle_bit_packed(&body, bit_width, indices.len()).unwrap();
        let want: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
        assert_eq!(
            decoded,
            want,
            "smart roundtrip failed: bit_width={bit_width}, len={}",
            indices.len()
        );
    }

    #[test]
    fn smart_empty_yields_empty() {
        let body = encode_rle_bit_packed(&[], 8);
        assert!(body.is_empty());
    }

    #[test]
    fn smart_bit_width_0_zeros() {
        let indices = vec![0u32; 100];
        smart_roundtrip(&indices, 0);
    }

    #[test]
    fn smart_all_same_uses_rle() {
        // 10000 copies of value 7 at bw=12 — should be very small.
        let indices = vec![7u32; 10_000];
        let body = encode_rle_bit_packed(&indices, 12);
        smart_roundtrip(&indices, 12);
        // RLE encoding: 1-2 byte uvarint header + 2 value bytes ≈ 3-4 bytes total.
        assert!(
            body.len() < 8,
            "all-same RLE should be tiny, got {} bytes",
            body.len()
        );
    }

    #[test]
    fn smart_alternating_uses_bitpack() {
        // No repetition → all bit-packed (same as single-run encoder).
        let indices: Vec<u32> = (0..1000).map(|i| (i & 0x7FF) as u32).collect();
        smart_roundtrip(&indices, 11);

        let smart = encode_rle_bit_packed(&indices, 11);
        let single = encode_rle_bit_packed_single_run(&indices, 11);
        assert_eq!(
            smart.len(),
            single.len(),
            "no-repetition input should match single-run encoder size"
        );
    }

    #[test]
    fn smart_mixed_runs_and_singletons() {
        // [a×20, b, c, d, e, f, g, h, i, a×100, j, k, l, m, n, o, p, q, a×50]
        // Three RLE runs interleaved with bit-pack groups.
        let mut indices: Vec<u32> = Vec::new();
        indices.extend(std::iter::repeat(5).take(20));
        for v in 0..8 {
            indices.push(v);
        }
        indices.extend(std::iter::repeat(5).take(100));
        for v in 8..16 {
            indices.push(v);
        }
        indices.extend(std::iter::repeat(5).take(50));

        smart_roundtrip(&indices, 12);

        let smart = encode_rle_bit_packed(&indices, 12);
        let single = encode_rle_bit_packed_single_run(&indices, 12);
        assert!(
            smart.len() < single.len(),
            "mixed input with long runs should compress better: smart {} vs single {}",
            smart.len(),
            single.len()
        );
    }

    #[test]
    fn smart_long_runs_at_every_bit_width() {
        // For each bit_width 1..=32, build a sequence with three
        // long runs of distinct values + some singletons. Verify
        // round-trip correctness.
        for bw in 1u8..=32 {
            let mask: u32 = if bw == 32 { u32::MAX } else { (1u32 << bw) - 1 };
            let v1 = 0u32;
            let v2 = mask / 3 & mask;
            let v3 = mask / 2 & mask;

            let mut indices: Vec<u32> = Vec::new();
            indices.extend(std::iter::repeat(v1).take(64));
            for i in 0..7 {
                indices.push((i & mask as usize) as u32 & mask);
            }
            indices.extend(std::iter::repeat(v2).take(32));
            indices.extend(std::iter::repeat(v3).take(13));
            for i in 0..3 {
                indices.push((i & mask as usize) as u32 & mask);
            }
            smart_roundtrip(&indices, bw);
        }
    }

    #[test]
    fn smart_short_runs_below_threshold_stay_in_bitpack() {
        // Runs of length 7 (just below the threshold) should stay
        // in bit-pack — no RLE emitted. Smart should match single-run.
        let mut indices: Vec<u32> = Vec::new();
        for v in [3u32, 5, 7, 11, 13] {
            for _ in 0..7 {
                indices.push(v);
            }
        }
        smart_roundtrip(&indices, 5);

        let smart = encode_rle_bit_packed(&indices, 5);
        let single = encode_rle_bit_packed_single_run(&indices, 5);
        assert_eq!(
            smart.len(),
            single.len(),
            "all runs below threshold should match single-run size"
        );
    }

    #[test]
    fn smart_starts_with_long_run() {
        // Long run at the very start — pending bit-pack is empty
        // when the RLE flush happens. Exercises the empty-pending
        // branch.
        let mut indices: Vec<u32> = std::iter::repeat(42u32).take(50).collect();
        for v in 0..8 {
            indices.push(v);
        }
        smart_roundtrip(&indices, 8);
    }

    #[test]
    fn smart_ends_with_long_run() {
        // Long run at the end — final flush only emits RLE.
        let mut indices: Vec<u32> = (0..15).collect();
        indices.extend(std::iter::repeat(99u32).take(40));
        smart_roundtrip(&indices, 7);
    }

    #[test]
    fn min_bit_width_table() {
        assert_eq!(min_bit_width_for_dict(0), 0);
        assert_eq!(min_bit_width_for_dict(1), 0);
        assert_eq!(min_bit_width_for_dict(2), 1);
        assert_eq!(min_bit_width_for_dict(3), 2);
        assert_eq!(min_bit_width_for_dict(4), 2);
        assert_eq!(min_bit_width_for_dict(5), 3);
        assert_eq!(min_bit_width_for_dict(255), 8);
        assert_eq!(min_bit_width_for_dict(256), 8);
        assert_eq!(min_bit_width_for_dict(257), 9);
        assert_eq!(min_bit_width_for_dict(4096), 12);
        assert_eq!(min_bit_width_for_dict(4097), 13);
    }
}

/// Decode `num_values` values from a RLE/bit-packed hybrid stream.
/// Values come out as `u64` regardless of declared bit_width — the
/// caller narrows to u32/u16/u8 as needed.
pub fn decode_rle_bit_packed(bytes: &[u8], bit_width: u8, num_values: usize) -> Result<Vec<u64>> {
    if bit_width > 64 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(Vec::new());
    }
    if bit_width == 0 {
        return Ok(vec![0u64; num_values]);
    }

    let mut out = Vec::with_capacity(num_values);
    let value_bytes = (bit_width as usize + 7) / 8;
    let mask: u64 = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    let mut cur = Cursor::new(bytes);

    while out.len() < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let needed = (total * bit_width as usize + 7) / 8;
            let chunk = cur.take(needed)?;
            let to_emit = (num_values - out.len()).min(total);
            for i in 0..to_emit {
                let bit_offset = i * bit_width as usize;
                let byte_offset = bit_offset / 8;
                let bit_in_byte = bit_offset % 8;
                let bits_needed = bit_in_byte + bit_width as usize;
                let bytes_needed = (bits_needed + 7) / 8;
                let mut acc: u64 = 0;
                for j in 0..bytes_needed {
                    acc |= (chunk[byte_offset + j] as u64) << (j * 8);
                }
                out.push((acc >> bit_in_byte) & mask);
            }
        } else {
            // RLE run.
            let value_chunk = cur.take(value_bytes)?;
            let mut val: u64 = 0;
            for j in 0..value_bytes {
                val |= (value_chunk[j] as u64) << (j * 8);
            }
            let to_emit = (num_values - out.len()).min(count);
            for _ in 0..to_emit {
                out.push(val);
            }
        }
    }
    Ok(out)
}
