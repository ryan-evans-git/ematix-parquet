//! Definition + repetition level decoding for v1 data pages.
//!
//! Parquet records nullability and nesting at the page-body level via
//! two parallel RLE/bit-packed streams that precede the value bytes:
//!
//!   v1 page body = [rep_levels][def_levels][values]
//!
//! Each level stream — when present — is prefixed by a 4-byte LE
//! `u32` byte length and encoded with the
//! [`crate::rle::decode_rle_bit_packed`] primitive at
//! `bit_width = ceil(log2(max_level + 1))`. The stream is OMITTED
//! entirely (no length prefix, no bytes) when its max level is 0.
//!
//! For REQUIRED non-nested columns (max_def_level = 0, max_rep_level = 0)
//! both streams are absent and the page body IS the value bytes —
//! which is what lineitem (TPC-H) hits in every column.
//!
//! For nullable scalar columns (max_def_level = 1, max_rep_level = 0):
//!   - rep stream omitted
//!   - def stream present, bit_width = 1
//!     a row's value is present iff def_level == 1
//!
//! v2 data pages encode the levels uncompressed in fixed-length
//! regions ahead of the (possibly compressed) values, with the lengths
//! stored on the page header itself. Not handled here yet.

use crate::error::{CodecError, Result};
use crate::rle::decode_rle_bit_packed;

/// Bit width needed to carry `0..=max_level` values.
///
///   bit_width_for(0) = 0          (level always 0; nothing on the wire)
///   bit_width_for(1) = 1          (typical nullable scalar)
///   bit_width_for(2..=3) = 2
///   bit_width_for(4..=7) = 3
///   ...
///
/// Equivalent to `ceil(log2(max_level + 1))` for `max_level >= 1`.
pub fn bit_width_for(max_level: u16) -> u8 {
    if max_level == 0 {
        0
    } else {
        (32 - (max_level as u32).leading_zeros()) as u8
    }
}

/// Decode a level stream out of the start of a v1 data-page body.
///
/// Returns `(levels, bytes_consumed)`. When `bit_width == 0` the spec
/// says the stream is omitted on the wire; we synthesize `num_values`
/// zeros and return `bytes_consumed = 0`.
///
/// Otherwise: reads a 4-byte LE length prefix, then decodes the
/// RLE/bit-packed stream of that length at the given bit width.
pub fn decode_levels(
    body: &[u8],
    bit_width: u8,
    num_values: usize,
) -> Result<(Vec<u16>, usize)> {
    if bit_width == 0 {
        return Ok((vec![0u16; num_values], 0));
    }
    if body.len() < 4 {
        return Err(CodecError::Decompress(format!(
            "level stream: need 4-byte length prefix, have {}",
            body.len()
        )));
    }
    let len = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
    let end = 4 + len;
    if body.len() < end {
        return Err(CodecError::Decompress(format!(
            "level stream: prefix says {len} body bytes, have {}",
            body.len() - 4
        )));
    }
    let level_bytes = &body[4..end];
    let levels: Vec<u16> = decode_rle_bit_packed(level_bytes, bit_width, num_values)?
        .into_iter()
        .map(|v| v as u16)
        .collect();
    Ok((levels, end))
}

/// Slice the rep + def level streams off the front of a v1 data-page
/// body and return them along with the remaining values-section
/// bytes. `max_rep_level`/`max_def_level` come from the column-chunk
/// schema (computed via `bit_width_for`).
///
/// Order on the wire is `rep` then `def` (rep levels come first).
pub fn parse_v1_data_page_body<'a>(
    body: &'a [u8],
    max_rep_level: u16,
    max_def_level: u16,
    num_values: usize,
) -> Result<(Vec<u16>, Vec<u16>, &'a [u8])> {
    let rep_bw = bit_width_for(max_rep_level);
    let def_bw = bit_width_for(max_def_level);

    let mut off = 0;
    let (rep, rep_consumed) = decode_levels(&body[off..], rep_bw, num_values)?;
    off += rep_consumed;
    let (def, def_consumed) = decode_levels(&body[off..], def_bw, num_values)?;
    off += def_consumed;

    Ok((rep, def, &body[off..]))
}
