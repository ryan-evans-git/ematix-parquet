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

/// Look up `indices` in `dict` to produce the column's actual values.
/// Bounds-checked: an out-of-range index returns
/// `CodecError::DictIndexOutOfRange`.
pub fn lookup_dict_i64(dict: &[i64], indices: &[u32]) -> Result<Vec<i64>> {
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
