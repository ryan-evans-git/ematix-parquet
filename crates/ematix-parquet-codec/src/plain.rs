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

/// Fast path for fixed-width PLAIN decode on little-endian targets:
/// raw memcpy from `bytes` into a pre-sized `Vec<T>`. Parquet's
/// PLAIN encoding stores values in little-endian byte order with no
/// per-value framing, so on LE platforms the byte semantics match
/// the in-memory representation of `T` exactly — no per-value
/// `from_le_bytes` needed.
///
/// `WIDTH` must equal `size_of::<T>()`. `bytes.len()` must be a
/// multiple of `WIDTH` (caller checks).
///
/// SAFETY of the unsafe block: the destination buffer is sized
/// `bytes.len()`, the source is `bytes.len()` bytes long, and they
/// cannot overlap (fresh allocation vs caller's input).
#[cfg(target_endian = "little")]
#[inline]
fn plain_memcpy<T: Copy>(bytes: &[u8], width: usize) -> Vec<T> {
    debug_assert_eq!(width, std::mem::size_of::<T>());
    debug_assert_eq!(bytes.len() % width, 0);
    let n = bytes.len() / width;
    let mut out: Vec<T> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(n);
    }
    out
}

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
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<i64>(bytes, 8));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / 8;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(8) {
            out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
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
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<i64>(&bytes[..needed], 8));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(n);
        for chunk in bytes[..needed].chunks_exact(8) {
            out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}

/// PLAIN-encoded INT32 — 4-byte little-endian per value.
pub fn decode_plain_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    if bytes.len() % 4 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
        });
    }
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<i32>(bytes, 4));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
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
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<i32>(&bytes[..needed], 4));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(n);
        for chunk in bytes[..needed].chunks_exact(4) {
            out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}

/// PLAIN-encoded FLOAT (IEEE-754 single, 4 bytes LE).
pub fn decode_plain_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
        });
    }
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<f32>(bytes, 4));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
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
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<f32>(&bytes[..needed], 4));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(n);
        for chunk in bytes[..needed].chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}

/// PLAIN-encoded DOUBLE (IEEE-754 double, 8 bytes LE).
pub fn decode_plain_f64(bytes: &[u8]) -> Result<Vec<f64>> {
    if bytes.len() % 8 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
        });
    }
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<f64>(bytes, 8));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / 8;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(8) {
            out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
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
    #[cfg(target_endian = "little")]
    {
        return Ok(plain_memcpy::<f64>(&bytes[..needed], 8));
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(n);
        for chunk in bytes[..needed].chunks_exact(8) {
            out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}

/// Sparse PLAIN decode for INT64. Reads only rows where the
/// corresponding bit in `mask` (interpreted at `bitmap_offset + row`)
/// is set. Appends matched values to `out` in row order.
///
/// Used by `read_column_i64_masked_into` on PLAIN data pages. The
/// 8-row block-skip mirrors `gather_dict_at_bitmap_into`: when the
/// mask byte covering 8 consecutive rows is 0, skip the whole block
/// (no value reads, one byte test). Pays off on selective scans.
///
/// `bitmap_offset` is the global row index of `bytes`'s row 0 within
/// the chunk's row-mask address space; required to be byte-aligned
/// (debug assertion). Real callers ensure this because parquet page
/// boundaries are always multiples of 8 (page sizes are typically
/// 20480 and dict-encoded chunks emit 8-row groups).
pub fn plain_sparse_decode_i64_into(
    bytes: &[u8],
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out: &mut Vec<i64>,
) -> Result<()> {
    let needed = num_values * 8;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
            requested_values: num_values,
        });
    }
    let mut row = 0usize;
    while row + 8 <= num_values {
        let bit_pos_base = bitmap_offset + row;
        debug_assert_eq!(bit_pos_base % 8, 0, "bitmap_offset must be byte-aligned");
        let mask_byte = mask[bit_pos_base / 8];
        if mask_byte != 0 {
            for lane in 0..8 {
                if (mask_byte >> lane) & 1 == 1 {
                    let off = (row + lane) * 8;
                    let v = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                    out.push(v);
                }
            }
        }
        row += 8;
    }
    while row < num_values {
        let bit_pos = bitmap_offset + row;
        let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            let off = row * 8;
            let v = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            out.push(v);
        }
        row += 1;
    }
    Ok(())
}

/// Sparse PLAIN decode for INT32. Same shape as the i64 variant; 4
/// bytes per value.
pub fn plain_sparse_decode_i32_into(
    bytes: &[u8],
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out: &mut Vec<i32>,
) -> Result<()> {
    let needed = num_values * 4;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 4,
            buffer_len: bytes.len(),
            requested_values: num_values,
        });
    }
    let mut row = 0usize;
    while row + 8 <= num_values {
        let bit_pos_base = bitmap_offset + row;
        debug_assert_eq!(bit_pos_base % 8, 0, "bitmap_offset must be byte-aligned");
        let mask_byte = mask[bit_pos_base / 8];
        if mask_byte != 0 {
            for lane in 0..8 {
                if (mask_byte >> lane) & 1 == 1 {
                    let off = (row + lane) * 4;
                    let v = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    out.push(v);
                }
            }
        }
        row += 8;
    }
    while row < num_values {
        let bit_pos = bitmap_offset + row;
        let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            let off = row * 4;
            let v = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            out.push(v);
        }
        row += 1;
    }
    Ok(())
}

/// Sparse PLAIN decode for DOUBLE. Same shape as the i64 variant.
pub fn plain_sparse_decode_f64_into(
    bytes: &[u8],
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out: &mut Vec<f64>,
) -> Result<()> {
    let needed = num_values * 8;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
            requested_values: num_values,
        });
    }
    let mut row = 0usize;
    while row + 8 <= num_values {
        let bit_pos_base = bitmap_offset + row;
        debug_assert_eq!(bit_pos_base % 8, 0, "bitmap_offset must be byte-aligned");
        let mask_byte = mask[bit_pos_base / 8];
        if mask_byte != 0 {
            for lane in 0..8 {
                if (mask_byte >> lane) & 1 == 1 {
                    let off = (row + lane) * 8;
                    let v = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                    out.push(v);
                }
            }
        }
        row += 8;
    }
    while row < num_values {
        let bit_pos = bitmap_offset + row;
        let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            let off = row * 8;
            let v = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            out.push(v);
        }
        row += 1;
    }
    Ok(())
}

/// PLAIN-encoded BOOLEAN — 1 bit per value, LSB-first within each
/// byte. The wire form holds `ceil(num_values / 8)` bytes; any
/// trailing bits in the last byte are padding and must be ignored.
///
/// Unlike the fixed-width numeric types, the count is needed
/// up-front because the padding makes the buffer ambiguous on its
/// own (5 values pack into the same 1 byte as 8 values).
pub fn decode_plain_bool(bytes: &[u8], num_values: usize) -> Result<Vec<bool>> {
    let needed = (num_values + 7) / 8;
    if bytes.len() < needed {
        return Err(CodecError::UnderflowingPlainBuffer {
            value_width: 1,
            buffer_len: bytes.len(),
            requested_values: num_values,
        });
    }
    let mut out = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let byte = bytes[i / 8];
        out.push((byte >> (i % 8)) & 1 == 1);
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

/// Sparse PLAIN decode for BYTE_ARRAY. Walks length-prefixes
/// sequentially (we can't skip ahead without parsing each prefix)
/// and copies only mask-set values into `out`.
///
/// Two output shapes — caller picks one:
///   - `plain_sparse_decode_byte_array_into(..., out: &mut Vec<Vec<u8>>)`
///     — one allocation per matched value (Arrow's BinaryArray is
///       a flatter layout; prefer the offsets variant when possible).
///   - `plain_sparse_decode_byte_array_offsets_into(..., bytes, offsets)`
///     — Arrow-style flat bytes + N+1 offsets (zero-malloc per row).
///
/// `bitmap_offset` is the global row index of the page's row 0
/// within the chunk's mask address space.
pub fn plain_sparse_decode_byte_array_into(
    bytes: &[u8],
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let mut cur = Cursor::new(bytes);
    for row in 0..num_values {
        let len = read_u32_le(&mut cur)? as usize;
        let value = cur.take(len)?;
        let bit_pos = bitmap_offset + row;
        let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            out.push(value.to_vec());
        }
    }
    Ok(())
}

/// Arrow-style sparse PLAIN decode for BYTE_ARRAY. Appends matched
/// values' bytes to `out_bytes` and pushes a running offset to
/// `out_offsets`.
///
/// On entry, if `out_offsets` is empty, the function pushes the
/// initial `0` offset; otherwise it continues from the existing
/// trailing offset (so multiple chunks can be concatenated into the
/// same `(out_bytes, out_offsets)` pair). After return,
/// `out_offsets.len() = previous_len + matched_in_this_call`
/// (i.e. one new offset per matched value).
pub fn plain_sparse_decode_byte_array_offsets_into(
    bytes: &[u8],
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out_bytes: &mut Vec<u8>,
    out_offsets: &mut Vec<u32>,
) -> Result<()> {
    if out_offsets.is_empty() {
        out_offsets.push(0);
    }
    let mut cur = Cursor::new(bytes);
    let mut running = *out_offsets.last().unwrap();
    for row in 0..num_values {
        let len = read_u32_le(&mut cur)? as usize;
        let value = cur.take(len)?;
        let bit_pos = bitmap_offset + row;
        let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            out_bytes.extend_from_slice(value);
            running = running.checked_add(len as u32).ok_or_else(|| {
                CodecError::InvalidInput(
                    "byte_array sparse-decode: offset overflow > u32::MAX".into(),
                )
            })?;
            out_offsets.push(running);
        }
    }
    Ok(())
}

fn read_u32_le(cur: &mut Cursor<'_>) -> Result<u32> {
    let bytes = cur.take(4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

// ---- INT96 ---------------------------------------------------------

/// 96-bit value. Legacy Parquet timestamp encoding (deprecated in
/// favour of INT64 micros / nanos, but still common in pre-2018 Hive
/// output). Wire form: three little-endian `u32`s, 12 bytes total.
///
/// We mirror parquet-rs's accessor shape (`data() -> [u32; 3]`) so
/// consumers can copy-paste timestamp-conversion logic between the
/// two libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Int96(pub [u32; 3]);

impl Int96 {
    pub fn new(a: u32, b: u32, c: u32) -> Self {
        Self([a, b, c])
    }
    pub fn data(&self) -> [u32; 3] {
        self.0
    }
}

/// PLAIN-encoded INT96 — 12 bytes per value, three little-endian u32s.
pub fn decode_plain_int96(bytes: &[u8]) -> Result<Vec<Int96>> {
    if bytes.len() % 12 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 12,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 12;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(12) {
        let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let c = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
        out.push(Int96([a, b, c]));
    }
    Ok(out)
}

// ---- FIXED_LEN_BYTE_ARRAY ------------------------------------------

/// PLAIN-encoded FIXED_LEN_BYTE_ARRAY — `type_length` raw bytes per
/// value, no length prefix. Zero-copy: returned slices borrow from
/// `bytes`. The caller knows `type_length` from the schema element.
///
/// Common usage: UUIDs (`type_length` = 16), DECIMAL(N, S) (binary
/// two's-complement of fixed byte width), arbitrary opaque BLOBs.
pub fn decode_plain_fixed_len_byte_array<'a>(
    bytes: &'a [u8],
    type_length: i32,
) -> Result<Vec<&'a [u8]>> {
    if type_length <= 0 {
        return Err(CodecError::InvalidInput(format!(
            "FIXED_LEN_BYTE_ARRAY requires type_length > 0, got {type_length}"
        )));
    }
    let stride = type_length as usize;
    if bytes.len() % stride != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: stride,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / stride;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * stride;
        out.push(&bytes[start..start + stride]);
    }
    Ok(out)
}
