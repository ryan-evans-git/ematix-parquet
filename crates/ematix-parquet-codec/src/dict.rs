//! RLE_DICTIONARY data-page encoding.
//!
//! When a column has a dictionary page, downstream data pages carry
//! RLE/bit-packed-encoded indices into that dictionary, not the
//! values themselves. The data-page body layout (after decompression,
//! and after any rep/def level prefixes for nested columns) is:
//!
//!   byte 0     : bit_width of the index stream (u8)
//!   bytes 1..  : RLE/bit-packed encoded indices, no length prefix
//!
//! The dictionary itself is PLAIN-encoded — decoded by `plain::*`
//! for the matching physical type.

use ematix_parquet_format::compact::{read_uvarint, Cursor};

use crate::bitpack::{unpack_indices_into, unpack_lookup_into};
#[cfg(target_arch = "aarch64")]
use crate::bitpack_neon::decode_predicate_bitmap_neon_bw12;
use crate::error::{CodecError, Result};

/// Decode `num_values` u32 indices from a data-page body that uses
/// RLE_DICTIONARY (or the legacy PLAIN_DICTIONARY) encoding.
///
/// Uses the const-generic per-bit-width unpacker for bit-packed runs;
/// RLE runs emit a single value repeated. This is the path used by
/// `DictColumnChunk` construction.
pub fn decode_rle_dictionary_indices(body: &[u8], num_values: usize) -> Result<Vec<u32>> {
    if body.is_empty() {
        return Err(CodecError::EmptyDictPageBody);
    }
    let bit_width = body[0];
    if bit_width > 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let mut out: Vec<u32> = Vec::with_capacity(num_values);
    if bit_width == 0 {
        out.resize(num_values, 0);
        return Ok(out);
    }

    let value_bytes = (bit_width as usize + 7) / 8;
    let mut cur = Cursor::new(&body[1..]);
    let mut emitted = 0usize;

    while emitted < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let needed = (total * bit_width as usize + 7) / 8;
            let chunk = cur.take(needed)?;
            let to_emit = (num_values - emitted).min(total);
            unpack_indices_into(chunk, to_emit, bit_width, &mut out)?;
            emitted += to_emit;
        } else {
            let value_chunk = cur.take(value_bytes)?;
            let mut idx: u32 = 0;
            for j in 0..value_bytes {
                idx |= (value_chunk[j] as u32) << (j * 8);
            }
            let to_emit = (num_values - emitted).min(count);
            for _ in 0..to_emit {
                out.push(idx);
            }
            emitted += to_emit;
        }
    }
    Ok(out)
}

/// Fused dict-decode: walk the RLE/bit-packed index stream and look
/// each index up in `dict`, appending values directly to `out`.
///
/// Skips the `Vec<u32>` index intermediate that
/// `decode_rle_dictionary_indices` + `lookup_dict` would allocate per
/// page. For lineitem `l_shipdate` (the Q14 filter column) this is a
/// ~3× speedup at the page-decode level.
///
/// Dict indices are spec-limited to 32 bits, so the bit-unpack
/// accumulator stays in `u32` instead of widening to `u64`.
pub fn decode_rle_dictionary_into<T: Copy>(
    body: &[u8],
    dict: &[T],
    num_values: usize,
    out: &mut Vec<T>,
) -> Result<()> {
    if body.is_empty() {
        return Err(CodecError::EmptyDictPageBody);
    }
    let bit_width = body[0];
    if bit_width > 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(());
    }

    out.reserve(num_values);
    let dict_size = dict.len();

    if bit_width == 0 {
        // Degenerate dict (1 distinct value); every index is 0.
        if dict_size == 0 {
            return Err(CodecError::DictIndexOutOfRange {
                index: 0,
                dict_size: 0,
            });
        }
        let v = dict[0];
        for _ in 0..num_values {
            out.push(v);
        }
        return Ok(());
    }

    let value_bytes = (bit_width as usize + 7) / 8;
    let mut cur = Cursor::new(&body[1..]);
    let mut emitted = 0usize;

    while emitted < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let needed = (total * bit_width as usize + 7) / 8;
            let chunk = cur.take(needed)?;
            let to_emit = (num_values - emitted).min(total);
            // Hand off to the const-generic unpacker. For typical
            // pages (full-page bit-packed run, no truncation needed),
            // `to_emit == total` and the unpacker processes the
            // entire run in 32-value chunks with NUM_BITS const-known.
            unpack_lookup_into(chunk, to_emit, bit_width, dict, out)?;
            emitted += to_emit;
        } else {
            // RLE run: one index repeated `count` times.
            let value_chunk = cur.take(value_bytes)?;
            let mut idx_u32: u32 = 0;
            for j in 0..value_bytes {
                idx_u32 |= (value_chunk[j] as u32) << (j * 8);
            }
            let idx = idx_u32 as usize;
            if idx >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx_u32,
                    dict_size,
                });
            }
            let v = dict[idx];
            let to_emit = (num_values - emitted).min(count);
            for _ in 0..to_emit {
                out.push(v);
            }
            emitted += to_emit;
        }
    }
    Ok(())
}

/// Look up `indices` in `dict` to produce the column's actual values.
/// Bounds-checked: an out-of-range index returns
/// `CodecError::DictIndexOutOfRange`.
///
/// Generic over the dict value type — `T: Copy` covers all parquet
/// physical types we materialize (Int32, Int64, Float32, Float64).
/// Walk an RLE_DICTIONARY data-page body (bit_width = 12) and emit
/// a packed predicate bitmap directly. Bit `i` of byte `k` represents
/// row `8k + i`; its value is `dict_mask[idx[8k + i]]` (zero-padded
/// for any unused tail bits in the last byte).
///
/// Fuses three logical passes into one — bit-unpack, dict-mask
/// gather, bitmap pack — with no intermediate `Vec<u32>` or
/// `Vec<bool>`. On bit-packed runs the inner loop runs the NEON
/// kernel `decode_predicate_bitmap_neon_bw12` (8 rows per ~7 NEON
/// ops + 8 byte loads). On RLE runs the kernel falls back to a
/// scalar "splat a dict_mask bit across N bitmap positions".
///
/// `dict_mask` must be ≥ 4096 bytes (zero-padded). Caller fills it
/// once per page by applying its predicate to each PLAIN-decoded
/// dict value.
///
/// Intended for the Q14-shape critical path: l_shipdate / shipdate-
/// like columns where bw=12 + ~1% selectivity dominate. Errors out
/// if the page's bit_width prefix is not 12.
pub fn decode_rle_dictionary_predicate_bitmap_bw12(
    body: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if body.is_empty() {
        return Err(CodecError::EmptyDictPageBody);
    }
    let bit_width = body[0];
    if bit_width != 12 {
        return Err(CodecError::Decompress(format!(
            "decode_rle_dictionary_predicate_bitmap_bw12: expected bit_width=12, got {bit_width}"
        )));
    }
    if dict_mask.len() < 4096 {
        return Err(CodecError::Decompress(format!(
            "dict_mask must be ≥ 4096 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }

    let bitmap_bytes = num_values.div_ceil(8);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let mut cur = Cursor::new(&body[1..]);
    let mut emitted: usize = 0;

    while emitted < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let to_emit = (num_values - emitted).min(total);
            let needed = (total * 12 + 7) / 8;
            let chunk = cur.take(needed)?;

            // Two cases:
            //   (a) `emitted % 8 == 0` AND `to_emit == total` (or % 8 == 0):
            //       The NEON kernel can write directly into the bitmap
            //       starting at byte `emitted / 8` — no row-shift work.
            //   (b) `emitted % 8 != 0` OR partial run:
            //       Use a temporary bitmap from a fresh `Vec<u8>` then
            //       OR-merge into `out` with the right row offset.
            //
            // In real lineitem pages emitted always grows by multiples
            // of 8 (page sizes are 20480 rows and RLE runs are
            // multiples of 8), so (a) is the common path.
            if emitted % 8 == 0 && to_emit % 8 == 0 {
                let dst_byte = out_start + emitted / 8;
                let dst_bytes = to_emit / 8;
                let mut tmp: Vec<u8> = Vec::with_capacity(dst_bytes);
                fused_bitmap_chunk(chunk, to_emit, dict_mask, &mut tmp)?;
                // `fused_bitmap_chunk` produces exactly `dst_bytes`
                // bytes since `to_emit % 8 == 0`.
                debug_assert_eq!(tmp.len(), dst_bytes);
                out[dst_byte..dst_byte + dst_bytes].copy_from_slice(&tmp);
            } else {
                // Slow path: produce a temp bitmap and OR-merge bit-
                // by-bit (rare).
                let mut tmp: Vec<u8> = Vec::with_capacity(to_emit.div_ceil(8));
                fused_bitmap_chunk(chunk, to_emit, dict_mask, &mut tmp)?;
                for i in 0..to_emit {
                    let src_bit = (tmp[i / 8] >> (i % 8)) & 1;
                    let row = emitted + i;
                    out[out_start + row / 8] |= src_bit << (row % 8);
                }
            }

            emitted += to_emit;
        } else {
            // RLE run: one index repeated `count` times.
            let value_chunk = cur.take(2)?; // bw=12 → 2 bytes
            let idx = ((value_chunk[0] as u32) | ((value_chunk[1] as u32) << 8)) as usize;
            if idx >= 4096 {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx as u32,
                    dict_size: 4096,
                });
            }
            // SAFETY: idx < 4096 ≤ dict_mask.len().
            let bit = unsafe { *dict_mask.get_unchecked(idx) };
            let to_emit = (num_values - emitted).min(count);
            if bit == 0 {
                // Zero already; just advance.
            } else {
                let start_row = emitted;
                let end_row = emitted + to_emit;
                splat_ones(out, out_start, start_row, end_row);
            }
            emitted += to_emit;
        }
    }
    Ok(())
}

/// Set bits `[start_row, end_row)` of the bitmap stored in
/// `out[out_start..]` to 1. Handles partial-byte heads and tails.
#[inline]
fn splat_ones(out: &mut [u8], out_start: usize, start_row: usize, end_row: usize) {
    if start_row >= end_row {
        return;
    }
    let mut row = start_row;
    // Partial head byte.
    while row < end_row && row % 8 != 0 {
        out[out_start + row / 8] |= 1u8 << (row % 8);
        row += 1;
    }
    // Whole bytes.
    while row + 8 <= end_row {
        out[out_start + row / 8] = 0xFF;
        row += 8;
    }
    // Partial tail.
    while row < end_row {
        out[out_start + row / 8] |= 1u8 << (row % 8);
        row += 1;
    }
}

/// Dispatch helper: bit-packed chunk → bitmap bytes. NEON on
/// aarch64, scalar fallback elsewhere.
#[inline]
fn fused_bitmap_chunk(
    chunk: &[u8],
    n: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    #[cfg(target_arch = "aarch64")]
    {
        return decode_predicate_bitmap_neon_bw12(chunk, n, dict_mask, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Scalar fallback: unpack, gather, pack.
        let mut idxs: Vec<u32> = Vec::with_capacity(n);
        crate::bitpack::unpack_indices_into(chunk, n, 12, &mut idxs)?;
        let bytes = n.div_ceil(8);
        let out_start = out.len();
        out.resize(out_start + bytes, 0);
        for (row, idx) in idxs.into_iter().enumerate() {
            let i = idx as usize;
            if i >= dict_mask.len() {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx,
                    dict_size: dict_mask.len(),
                });
            }
            let bit = dict_mask[i];
            out[out_start + row / 8] |= bit << (row % 8);
        }
        Ok(())
    }
}

pub fn lookup_dict<T: Copy>(dict: &[T], indices: &[u32]) -> Result<Vec<T>> {
    let n = dict.len();
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let i = idx as usize;
        if i >= n {
            return Err(CodecError::DictIndexOutOfRange {
                index: idx,
                dict_size: n,
            });
        }
        out.push(dict[i]);
    }
    Ok(out)
}
