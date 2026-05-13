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
