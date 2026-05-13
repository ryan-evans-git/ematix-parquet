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

use crate::error::{CodecError, Result};
use crate::rle::decode_rle_bit_packed;

/// Decode `num_values` u32 indices from a data-page body that uses
/// RLE_DICTIONARY (or the legacy PLAIN_DICTIONARY) encoding.
pub fn decode_rle_dictionary_indices(body: &[u8], num_values: usize) -> Result<Vec<u32>> {
    if body.is_empty() {
        return Err(CodecError::EmptyDictPageBody);
    }
    let bit_width = body[0];
    let decoded = decode_rle_bit_packed(&body[1..], bit_width, num_values)?;
    // Spec caps dict indices at 32 bits. Truncating from u64 here is
    // safe; the caller chose i64 only because the underlying RLE
    // primitive emits u64 to stay general.
    Ok(decoded.into_iter().map(|v| v as u32).collect())
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
    let mask: u32 = if bit_width == 32 {
        u32::MAX
    } else {
        (1u32 << bit_width) - 1
    };
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

            // Streaming bit buffer: each input byte is read exactly
            // once, then drained value-by-value via shift+mask. Tight
            // inner loop, no per-value byte_offset recomputation.
            let bw = bit_width as u32;
            let mask64 = mask as u64;
            let mut buf: u64 = 0;
            let mut bits: u32 = 0;
            let mut byte_idx: usize = 0;

            // When the dict covers the full index range (`dict_size >=
            // 2^bit_width`), the bit mask itself guarantees every
            // extracted index is in-bounds. Hoist the bounds check
            // out of the hot loop in that case.
            let bounds_safe = (mask as usize) < dict_size;
            if bounds_safe {
                for _ in 0..to_emit {
                    while bits < bw {
                        buf |= (chunk[byte_idx] as u64) << bits;
                        byte_idx += 1;
                        bits += 8;
                    }
                    let idx = (buf & mask64) as usize;
                    buf >>= bw;
                    bits -= bw;
                    out.push(dict[idx]);
                }
            } else {
                for _ in 0..to_emit {
                    while bits < bw {
                        buf |= (chunk[byte_idx] as u64) << bits;
                        byte_idx += 1;
                        bits += 8;
                    }
                    let idx = (buf & mask64) as usize;
                    buf >>= bw;
                    bits -= bw;
                    if idx >= dict_size {
                        return Err(CodecError::DictIndexOutOfRange {
                            index: idx as u32,
                            dict_size,
                        });
                    }
                    out.push(dict[idx]);
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
