//! PLAIN-encoding decoders, one per Parquet physical type.
//!
//! For fixed-width numeric types (INT32, INT64, FLOAT, DOUBLE),
//! PLAIN is just a contiguous run of little-endian values:
//!
//!   INT64 values: [b0 b1 b2 b3 b4 b5 b6 b7] [b0 b1 ... b7] ...
//!
//! No length prefix, no per-value framing. The number of values is
//! known from the page header (`num_values`); the byte count is
//! `num_values * sizeof(T)`.
//!
//! Dictionary pages also use PLAIN for the dict values themselves;
//! the data pages that reference them use RLE_DICTIONARY (indices),
//! which lands in `dict.rs` once that module exists.

use crate::error::{CodecError, Result};

/// Decode a PLAIN-encoded byte buffer as i64 values. The buffer length
/// must be an exact multiple of 8; partial trailing bytes are a
/// wire-format violation, not silently dropped.
pub fn decode_plain_i64(bytes: &[u8]) -> Result<Vec<i64>> {
    if bytes.len() % 8 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// Same as `decode_plain_i64` but caps the number of values consumed.
/// Useful when a buffer may carry trailing rep/def-level bytes or
/// other data the caller wants to read separately.
pub fn decode_plain_i64_n(bytes: &[u8], n: usize) -> Result<Vec<i64>> {
    let needed = n * 8;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
            requested_values: n,
        });
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes[..needed].chunks_exact(8) {
        out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}
