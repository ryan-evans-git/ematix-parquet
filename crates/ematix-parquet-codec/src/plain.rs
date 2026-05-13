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

use ematix_parquet_format::compact::Cursor;

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

/// PLAIN-encoded INT32 — 4-byte little-endian per value.
pub fn decode_plain_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    if bytes.len() % 4 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn decode_plain_i32_n(bytes: &[u8], n: usize) -> Result<Vec<i32>> {
    let needed = n * 4;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
            requested_values: n,
        });
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes[..needed].chunks_exact(4) {
        out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// PLAIN-encoded FLOAT (IEEE-754 single, 4 bytes LE).
pub fn decode_plain_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn decode_plain_f32_n(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let needed = n * 4;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
            requested_values: n,
        });
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes[..needed].chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// PLAIN-encoded DOUBLE (IEEE-754 double, 8 bytes LE).
pub fn decode_plain_f64(bytes: &[u8]) -> Result<Vec<f64>> {
    if bytes.len() % 8 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn decode_plain_f64_n(bytes: &[u8], n: usize) -> Result<Vec<f64>> {
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
        out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// PLAIN-encoded BYTE_ARRAY — each value is a `u32-LE` length prefix
/// followed by that many raw bytes. Zero-copy: the returned slices
/// borrow from `bytes`. Consumes the whole buffer.
pub fn decode_plain_byte_array<'a>(bytes: &'a [u8]) -> Result<Vec<&'a [u8]>> {
    let mut out = Vec::new();
    let mut cur = Cursor::new(bytes);
    while !cur.is_empty() {
        let len = read_u32_le(&mut cur)? as usize;
        let value = cur.take(len)?;
        out.push(value);
    }
    Ok(out)
}

/// Same as `decode_plain_byte_array` but stops after `n` values. Any
/// trailing bytes are left untouched — useful when the buffer carries
/// more than just the values (e.g. dict-followed-by-padding).
pub fn decode_plain_byte_array_n<'a>(bytes: &'a [u8], n: usize) -> Result<Vec<&'a [u8]>> {
    let mut out = Vec::with_capacity(n);
    let mut cur = Cursor::new(bytes);
    for _ in 0..n {
        let len = read_u32_le(&mut cur)? as usize;
        let value = cur.take(len)?;
        out.push(value);
    }
    Ok(out)
}

fn read_u32_le(cur: &mut Cursor<'_>) -> Result<u32> {
    let bytes = cur.take(4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
