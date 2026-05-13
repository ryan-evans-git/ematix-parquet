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
