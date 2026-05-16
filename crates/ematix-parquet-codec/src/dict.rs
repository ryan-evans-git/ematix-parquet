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
use crate::bitpack_neon::{
    decode_predicate_bitmap_neon_bw12, decode_predicate_bitmap_neon_bw14,
    decode_predicate_bitmap_neon_bw15, decode_predicate_bitmap_neon_bw16,
    decode_predicate_bitmap_neon_bw17, decode_predicate_bitmap_neon_bw18,
};
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
    decode_rle_dictionary_predicate_bitmap(body, num_values, dict_mask, out)
}

/// Width-generic predicate-fused decode. Mirror of
/// `decode_rle_dictionary_predicate_bitmap_bw12` but supports any
/// bit_width: NEON-fused for bw ∈ {12, 14, 15, 16, 17, 18}, scalar
/// fallback otherwise.
///
/// `dict_mask.len()` must be ≥ `1 << bit_width` so the NEON gather
/// can safely read every possible index without per-lane bounds
/// checks. Bits ≥ `dict.len()` should be zero-padded by the caller
/// (see `build_dict_predicate_mask`).
///
/// Produces a packed bitmap (`num_values.div_ceil(8)` bytes appended
/// to `out`). Bit `i` of byte `k` is `dict_mask[idx[8k+i]]`.
pub fn decode_rle_dictionary_predicate_bitmap(
    body: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if body.is_empty() {
        return Err(CodecError::EmptyDictPageBody);
    }
    let bit_width = body[0];
    if bit_width > 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    let required_mask = if bit_width == 0 {
        1
    } else if bit_width >= 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    } else {
        1usize << bit_width
    };
    if dict_mask.len() < required_mask {
        return Err(CodecError::Decompress(format!(
            "dict_mask must be ≥ {required_mask} entries for bw={bit_width} (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }

    let bitmap_bytes = num_values.div_ceil(8);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    if bit_width == 0 {
        // Degenerate dict: every index is 0, so every bit is dict_mask[0].
        if dict_mask[0] != 0 {
            splat_ones(out, out_start, 0, num_values);
        }
        return Ok(());
    }

    let value_bytes = (bit_width as usize + 7) / 8;
    let mut cur = Cursor::new(&body[1..]);
    let mut emitted: usize = 0;

    while emitted < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let to_emit = (num_values - emitted).min(total);
            let needed = (total * bit_width as usize + 7) / 8;
            let chunk = cur.take(needed)?;

            // Common path: row-aligned + emit-aligned. NEON kernel
            // writes directly into the bitmap.
            if emitted % 8 == 0 && to_emit % 8 == 0 {
                let dst_byte = out_start + emitted / 8;
                let dst_bytes = to_emit / 8;
                let mut tmp: Vec<u8> = Vec::with_capacity(dst_bytes);
                fused_bitmap_chunk(chunk, bit_width, to_emit, dict_mask, &mut tmp)?;
                debug_assert_eq!(tmp.len(), dst_bytes);
                out[dst_byte..dst_byte + dst_bytes].copy_from_slice(&tmp);
            } else {
                // Slow path: produce temp bitmap, OR-merge bit-by-bit.
                let mut tmp: Vec<u8> = Vec::with_capacity(to_emit.div_ceil(8));
                fused_bitmap_chunk(chunk, bit_width, to_emit, dict_mask, &mut tmp)?;
                for i in 0..to_emit {
                    let src_bit = (tmp[i / 8] >> (i % 8)) & 1;
                    let row = emitted + i;
                    out[out_start + row / 8] |= src_bit << (row % 8);
                }
            }

            emitted += to_emit;
        } else {
            // RLE run: one index repeated `count` times.
            let value_chunk = cur.take(value_bytes)?;
            let mut idx_u32: u32 = 0;
            for j in 0..value_bytes {
                idx_u32 |= (value_chunk[j] as u32) << (j * 8);
            }
            let idx = idx_u32 as usize;
            if idx >= required_mask {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx_u32,
                    dict_size: required_mask,
                });
            }
            // SAFETY: idx < required_mask ≤ dict_mask.len().
            let bit = unsafe { *dict_mask.get_unchecked(idx) };
            let to_emit = (num_values - emitted).min(count);
            if bit != 0 {
                let start_row = emitted;
                let end_row = emitted + to_emit;
                splat_ones(out, out_start, start_row, end_row);
            }
            emitted += to_emit;
        }
    }
    Ok(())
}

/// Build a `dict_mask` suitable for `decode_rle_dictionary_predicate_bitmap`
/// from a decoded dictionary and a predicate. Output length is
/// `1 << bit_width` (padded with zeros for index slots beyond
/// `dict.len()`).
///
/// Caller must pass the bit_width that matches the data pages — this
/// is the value in `body[0]` of any dict-encoded page in the chunk.
pub fn build_dict_predicate_mask<T>(
    dict: &[T],
    bit_width: u8,
    pred: impl Fn(&T) -> bool,
) -> Result<Vec<u8>> {
    if bit_width > 32 || bit_width == 0 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    let len = 1usize << bit_width;
    if dict.len() > len {
        return Err(CodecError::Decompress(format!(
            "dict has {} entries, exceeds bit_width={bit_width} addressable space ({len})",
            dict.len()
        )));
    }
    let mut mask = vec![0u8; len];
    for (i, v) in dict.iter().enumerate() {
        if pred(v) {
            mask[i] = 1;
        }
    }
    Ok(mask)
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

/// Dispatch helper: bit-packed chunk → bitmap bytes. NEON-fused for
/// bw ∈ {12, 14, 15, 16, 17, 18}; scalar fallback otherwise. Caller
/// has already verified `dict_mask.len() ≥ 1 << bit_width`.
#[inline]
fn fused_bitmap_chunk(
    chunk: &[u8],
    bit_width: u8,
    n: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    #[cfg(target_arch = "aarch64")]
    {
        match bit_width {
            12 => return decode_predicate_bitmap_neon_bw12(chunk, n, dict_mask, out),
            14 => return decode_predicate_bitmap_neon_bw14(chunk, n, dict_mask, out),
            15 => return decode_predicate_bitmap_neon_bw15(chunk, n, dict_mask, out),
            16 => return decode_predicate_bitmap_neon_bw16(chunk, n, dict_mask, out),
            17 => return decode_predicate_bitmap_neon_bw17(chunk, n, dict_mask, out),
            18 => return decode_predicate_bitmap_neon_bw18(chunk, n, dict_mask, out),
            _ => {}
        }
    }
    // Scalar fallback: unpack, gather, pack.
    let mut idxs: Vec<u32> = Vec::with_capacity(n);
    unpack_indices_into(chunk, n, bit_width, &mut idxs)?;
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

/// Bitmap-driven sparse gather from a dict-encoded data page.
///
/// Walks the RLE/bit-packed body, and for each row where the global
/// row-mask `bitmap[bitmap_offset + row]` is 1, pushes `dict[idx[row]]`
/// into `out`. Rows whose bitmap bit is 0 are skipped (bit stream
/// still advances).
///
/// Use case: after a Phase-5-style filter produces a column bitmap,
/// the aggregate columns (l_partkey, l_extendedprice, l_discount in
/// Q14) are decoded sparsely — only the matching rows materialize.
/// At typical Q14 selectivity (~1%) this collapses the per-page
/// gather from ~20K writes to ~250 writes. The unpack still walks
/// the full stream (parquet's bit-packing is positional), but the
/// allocator + cache traffic drops 100×.
///
/// `bitmap_offset` is the global row index of `num_values=0`. Used
/// when the same bitmap covers multiple pages of a row group.
pub fn gather_dict_at_bitmap_into<T: Clone>(
    body: &[u8],
    num_values: usize,
    bitmap: &[u8],
    bitmap_offset: usize,
    dict: &[T],
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
    let dict_size = dict.len();

    if bit_width == 0 {
        if dict_size == 0 {
            return Err(CodecError::DictIndexOutOfRange {
                index: 0,
                dict_size: 0,
            });
        }
        let v = &dict[0];
        for row in 0..num_values {
            let bit_pos = bitmap_offset + row;
            let bit = (bitmap[bit_pos / 8] >> (bit_pos % 8)) & 1;
            if bit == 1 {
                out.push(v.clone());
            }
        }
        return Ok(());
    }

    let value_bytes = (bit_width as usize + 7) / 8;
    let mut cur = Cursor::new(&body[1..]);
    let mut row: usize = 0;

    while row < num_values {
        let header = read_uvarint(&mut cur)?;
        let is_bit_packed = (header & 1) == 1;
        let count = (header >> 1) as usize;

        if is_bit_packed {
            let total = count * 8;
            let to_consume = (num_values - row).min(total);
            let needed = (total * bit_width as usize + 7) / 8;
            let chunk = cur.take(needed)?;

            let bytes_per_8 = bit_width as usize;
            debug_assert_eq!(8 * bit_width as usize, bytes_per_8 * 8);
            let mut byte_cursor = 0usize;
            let mut block_row = 0usize;
            let mut scratch: [u32; 8] = [0; 8];

            // Head peel: the byte-wise fast path below assumes
            // `bitmap_offset + row + block_row` is a multiple of 8 so a
            // single `bitmap[..]` byte covers exactly the next 8 rows.
            // When `bitmap_offset` is not 8-aligned (which happens for
            // any data page whose predecessor pages didn't sum to a
            // multiple of 8 — every byte-array Utf8 column in TPC-H
            // l_returnflag-shaped data), we peel rows until aligned,
            // per-row scalar.
            let head_unaligned = (bitmap_offset + row) % 8;
            if head_unaligned != 0 {
                let head_rows = (8 - head_unaligned).min(to_consume);
                let head_bytes = (head_rows * bit_width as usize + 7) / 8;
                let mut head_idxs: Vec<u32> = Vec::with_capacity(head_rows);
                unpack_indices_into(
                    &chunk[byte_cursor..byte_cursor + head_bytes],
                    head_rows,
                    bit_width,
                    &mut head_idxs,
                )?;
                for (i, idx) in head_idxs.into_iter().enumerate() {
                    let bit_pos = bitmap_offset + row + i;
                    let bit = (bitmap[bit_pos / 8] >> (bit_pos % 8)) & 1;
                    if bit == 1 {
                        let idx_u = idx as usize;
                        if idx_u >= dict_size {
                            return Err(CodecError::DictIndexOutOfRange {
                                index: idx,
                                dict_size,
                            });
                        }
                        out.push(dict[idx_u].clone());
                    }
                }
                // The bit-packed input is positional — each row
                // consumes exactly `bit_width` bits, regardless of
                // whether it's part of an 8-row block. So advancing
                // the byte cursor by `head_rows * bit_width / 8` is
                // only safe when that product is a whole number of
                // bytes. `head_rows` is at most 7; pick the smallest
                // multiple of 8 we can re-enter the fast path at by
                // peeling rows up to the next 8-row boundary in the
                // *page* (block_row % 8 == 0), which happens when
                // `head_rows + block_row` is a multiple of 8. Since
                // block_row starts at 0 and head_rows ∈ [1, 7], that
                // means we peel `head_rows` and then add (8 -
                // head_rows) more to reach the next 8-row block.
                // Simpler approach: don't try to re-enter the fast
                // path mid-page when misaligned — just scalar-process
                // the whole run. Performance regression is bounded:
                // misalignment only happens on Utf8 / byte-array
                // columns where the page layout doesn't pre-align.
                let total_bytes = (to_consume * bit_width as usize + 7) / 8;
                let mut all_idxs: Vec<u32> = Vec::with_capacity(to_consume);
                unpack_indices_into(
                    &chunk[byte_cursor..byte_cursor + total_bytes],
                    to_consume,
                    bit_width,
                    &mut all_idxs,
                )?;
                for (i, idx) in all_idxs.into_iter().enumerate().skip(head_rows) {
                    let bit_pos = bitmap_offset + row + i;
                    let bit = (bitmap[bit_pos / 8] >> (bit_pos % 8)) & 1;
                    if bit == 1 {
                        let idx_u = idx as usize;
                        if idx_u >= dict_size {
                            return Err(CodecError::DictIndexOutOfRange {
                                index: idx,
                                dict_size,
                            });
                        }
                        out.push(dict[idx_u].clone());
                    }
                }
                row += to_consume;
                continue;
            }

            // Aligned fast path: 8 rows at a time. Each 8-row block
            // consumes `bit_width` bytes (= 8 × bit_width bits), so
            // we can skip a block by advancing the byte cursor by
            // `bit_width` without unpacking, *iff* the bitmap byte
            // for that block is 0. At Q14-shape selectivity (~1%)
            // almost all bitmap bytes are 0, so most blocks skip
            // entirely.
            while block_row + 8 <= to_consume {
                let bit_pos_base = bitmap_offset + row + block_row;
                debug_assert_eq!(bit_pos_base % 8, 0, "fast path requires byte-aligned offset");
                let mask_byte = bitmap[bit_pos_base / 8];
                if mask_byte != 0 {
                    unpack_8_indices(&chunk[byte_cursor..byte_cursor + bytes_per_8], bit_width, &mut scratch);
                    for lane in 0..8 {
                        if (mask_byte >> lane) & 1 == 1 {
                            let idx_u = scratch[lane] as usize;
                            if idx_u >= dict_size {
                                return Err(CodecError::DictIndexOutOfRange {
                                    index: scratch[lane],
                                    dict_size,
                                });
                            }
                            out.push(dict[idx_u].clone());
                        }
                    }
                }
                byte_cursor += bytes_per_8;
                block_row += 8;
            }
            // Tail (< 8 rows) at the end of the run: scalar per-row.
            if block_row < to_consume {
                let tail = to_consume - block_row;
                let tail_bytes = (tail * bit_width as usize + 7) / 8;
                let mut tail_idxs: Vec<u32> = Vec::with_capacity(tail);
                unpack_indices_into(
                    &chunk[byte_cursor..byte_cursor + tail_bytes],
                    tail,
                    bit_width,
                    &mut tail_idxs,
                )?;
                for (i, idx) in tail_idxs.into_iter().enumerate() {
                    let bit_pos = bitmap_offset + row + block_row + i;
                    let bit = (bitmap[bit_pos / 8] >> (bit_pos % 8)) & 1;
                    if bit == 1 {
                        let idx_u = idx as usize;
                        if idx_u >= dict_size {
                            return Err(CodecError::DictIndexOutOfRange {
                                index: idx,
                                dict_size,
                            });
                        }
                        out.push(dict[idx_u].clone());
                    }
                }
            }
            row += to_consume;
        } else {
            // RLE run: one index repeated `count` times. Bitmap tells
            // us how many matching rows to emit, all with the same value.
            let value_chunk = cur.take(value_bytes)?;
            let mut idx_u32: u32 = 0;
            for j in 0..value_bytes {
                idx_u32 |= (value_chunk[j] as u32) << (j * 8);
            }
            let idx_u = idx_u32 as usize;
            if idx_u >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx_u32,
                    dict_size,
                });
            }
            let v = &dict[idx_u];
            let to_consume = (num_values - row).min(count);
            // Count set bits in `bitmap[bitmap_offset + row .. + to_consume]`
            // and push `v` that many times. Faster than per-row test
            // for long RLE runs since `v` is constant.
            let matched = popcount_range(bitmap, bitmap_offset + row, bitmap_offset + row + to_consume);
            for _ in 0..matched {
                out.push(v.clone());
            }
            row += to_consume;
        }
    }
    Ok(())
}

/// Streaming bit-unpack of exactly 8 values at `bit_width` bits each.
/// Input must be exactly `bit_width` bytes (= 8 values' worth).
/// Used by `gather_dict_at_bitmap_into` to unpack one 8-row block
/// only when the bitmap byte indicates at least one match.
#[inline]
fn unpack_8_indices(packed: &[u8], bit_width: u8, out: &mut [u32; 8]) {
    debug_assert_eq!(packed.len(), bit_width as usize);
    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bit_width) - 1
    };
    let mut buf: u64 = 0;
    let mut bits: u32 = 0;
    let mut byte_idx = 0usize;
    for lane in 0..8 {
        while bits < bit_width as u32 {
            buf |= (packed[byte_idx] as u64) << bits;
            byte_idx += 1;
            bits += 8;
        }
        out[lane] = (buf & mask) as u32;
        buf >>= bit_width;
        bits -= bit_width as u32;
    }
}

/// Count the number of set bits in `bitmap[start_bit..end_bit]`.
/// Fast paths whole-byte aligned cases via `count_ones()`; partial
/// head/tail bytes via masked count.
fn popcount_range(bitmap: &[u8], start_bit: usize, end_bit: usize) -> usize {
    if start_bit >= end_bit {
        return 0;
    }
    let mut bit = start_bit;
    let mut total: usize = 0;
    // Partial head.
    while bit < end_bit && bit % 8 != 0 {
        let b = bitmap[bit / 8];
        if (b >> (bit % 8)) & 1 == 1 {
            total += 1;
        }
        bit += 1;
    }
    // Whole bytes.
    while bit + 8 <= end_bit {
        total += bitmap[bit / 8].count_ones() as usize;
        bit += 8;
    }
    // Partial tail.
    while bit < end_bit {
        let b = bitmap[bit / 8];
        if (b >> (bit % 8)) & 1 == 1 {
            total += 1;
        }
        bit += 1;
    }
    total
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
