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

    // Pack LSB-first into `num_groups * bit_width` bytes; pad the
    // final partial group with zero indices so the byte count matches
    // what the decoder expects.
    let body_len = num_groups * bit_width as usize;
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
            let take_mask = ((1u64 << take) - 1) as u64;
            let chunk = (value & take_mask) as u8;
            out[byte_ix] |= chunk << shift;
            value >>= take;
            remaining -= take;
            byte_ix += 1;
            shift = 0;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(indices: &[u32], bit_width: u8) {
        let body = encode_rle_bit_packed_single_run(indices, bit_width);
        let decoded = decode_rle_bit_packed(&body, bit_width, indices.len()).unwrap();
        let want: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
        assert_eq!(decoded, want, "bit_width={bit_width}, len={}", indices.len());
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
pub fn decode_rle_bit_packed(
    bytes: &[u8],
    bit_width: u8,
    num_values: usize,
) -> Result<Vec<u64>> {
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
