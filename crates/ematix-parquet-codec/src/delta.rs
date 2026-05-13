//! DELTA_BINARY_PACKED encoding (Parquet v2 default for integer
//! columns from Arrow-rs, modern Spark, and many other writers).
//!
//! Wire format (per parquet-format Encodings.md):
//!
//!   <block_size: uvarint>
//!   <mini_blocks_per_block: uvarint>
//!   <num_values: uvarint>          (total, including the first value)
//!   <first_value: zigzag varint>
//!
//!   for each block of (block_size) deltas:
//!     <min_delta_in_block: zigzag varint>
//!     <bit_widths: mini_blocks_per_block bytes>
//!     <packed_deltas: bit-packed at the corresponding bit_width per
//!                     mini-block; each mini-block holds
//!                     block_size / mini_blocks_per_block deltas>
//!
//! Constraints from the spec:
//!   - `block_size` must be a multiple of 128
//!   - `block_size / mini_blocks_per_block` must be a multiple of 32
//!     (i.e. mini-block size aligns with our 32-value chunked
//!     unpacker)
//!
//! Decoding:
//!   - Emit `first_value`.
//!   - For each delta read from the bit-packed mini-blocks,
//!         next_value = prev_value + min_delta + delta
//!     and emit it.
//!   - Stop after `num_values` total emissions; any trailing
//!     padding deltas in the last mini-block are discarded.
//!
//! The last block in a stream may not be full: the writer pads
//! the bit-packed values to a full mini-block and we ignore the
//! extras.

use ematix_parquet_format::compact::{read_uvarint, read_zigzag_i32, read_zigzag_i64, Cursor};

use crate::bitpack::unpack_indices_into;
use crate::error::{CodecError, Result};

/// Decode a DELTA_BINARY_PACKED INT32 stream.
///
/// Internally accumulates in i64 to keep partial sums correct even
/// when intermediate `min_delta + delta` arithmetic exceeds i32
/// range; downcast back to i32 on emit.
pub fn decode_delta_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    let mut cur = Cursor::new(bytes);
    let block_size = read_uvarint(&mut cur)? as usize;
    let mini_blocks_per_block = read_uvarint(&mut cur)? as usize;
    let num_values = read_uvarint(&mut cur)? as usize;
    let first_value = read_zigzag_i32(&mut cur)?;

    if num_values == 0 {
        return Ok(Vec::new());
    }
    if mini_blocks_per_block == 0 {
        return Err(delta_err("mini_blocks_per_block must be > 0"));
    }
    if !block_size.is_multiple_of(mini_blocks_per_block) {
        return Err(delta_err(
            "block_size must be a multiple of mini_blocks_per_block",
        ));
    }
    let mini_block_size = block_size / mini_blocks_per_block;

    let mut out = Vec::with_capacity(num_values);
    out.push(first_value);
    let mut prev: i64 = first_value as i64;
    let mut remaining = num_values - 1;

    let mut unpack_buf: Vec<u32> = Vec::with_capacity(mini_block_size);

    while remaining > 0 {
        let block_min_delta = read_zigzag_i64(&mut cur)?;

        // Read mini_blocks_per_block bit_width bytes up front; readers
        // need them all to know how far to jump per mini-block.
        let mut bit_widths: [u8; 64] = [0u8; 64]; // mini_blocks_per_block is small
        if mini_blocks_per_block > bit_widths.len() {
            return Err(delta_err("mini_blocks_per_block too large"));
        }
        for i in 0..mini_blocks_per_block {
            bit_widths[i] = cur.read_u8()?;
        }

        for &bit_width in &bit_widths[..mini_blocks_per_block] {
            if remaining == 0 {
                break;
            }
            // Bytes needed = mini_block_size * bit_width / 8.
            // block_size is a multiple of 128 and mini_block_size of
            // 32, so this product is always a whole byte count.
            let body_bytes = mini_block_size * bit_width as usize / 8;
            let chunk = cur.take(body_bytes)?;

            unpack_buf.clear();
            unpack_indices_into(chunk, mini_block_size, bit_width, &mut unpack_buf)?;

            for &delta_u in unpack_buf.iter() {
                if remaining == 0 {
                    break;
                }
                prev = prev.wrapping_add(block_min_delta).wrapping_add(delta_u as i64);
                out.push(prev as i32);
                remaining -= 1;
            }
        }
    }
    Ok(out)
}

/// Decode a DELTA_BINARY_PACKED INT64 stream.
///
/// Uses an i128 accumulator so the partial sums can't overflow on
/// any valid input. Caps mini-block `bit_width` at 32 because our
/// const-generic unpacker only specializes up to 32 bits per value;
/// real-world DELTA-i64 data almost never crosses that threshold
/// (the writer would emit PLAIN in that case). Larger bit_widths
/// would need a u64-output unpacker (TODO).
pub fn decode_delta_i64(bytes: &[u8]) -> Result<Vec<i64>> {
    let mut cur = Cursor::new(bytes);
    let block_size = read_uvarint(&mut cur)? as usize;
    let mini_blocks_per_block = read_uvarint(&mut cur)? as usize;
    let num_values = read_uvarint(&mut cur)? as usize;
    let first_value = read_zigzag_i64(&mut cur)?;

    if num_values == 0 {
        return Ok(Vec::new());
    }
    if mini_blocks_per_block == 0 {
        return Err(delta_err("mini_blocks_per_block must be > 0"));
    }
    if !block_size.is_multiple_of(mini_blocks_per_block) {
        return Err(delta_err(
            "block_size must be a multiple of mini_blocks_per_block",
        ));
    }
    let mini_block_size = block_size / mini_blocks_per_block;

    let mut out = Vec::with_capacity(num_values);
    out.push(first_value);
    let mut prev: i128 = first_value as i128;
    let mut remaining = num_values - 1;

    let mut unpack_buf: Vec<u32> = Vec::with_capacity(mini_block_size);

    while remaining > 0 {
        let block_min_delta = read_zigzag_i64(&mut cur)? as i128;

        let mut bit_widths: [u8; 64] = [0u8; 64];
        if mini_blocks_per_block > bit_widths.len() {
            return Err(delta_err("mini_blocks_per_block too large"));
        }
        for i in 0..mini_blocks_per_block {
            bit_widths[i] = cur.read_u8()?;
        }

        for &bit_width in &bit_widths[..mini_blocks_per_block] {
            if remaining == 0 {
                break;
            }
            if bit_width > 32 {
                return Err(delta_err(
                    "i64 DELTA with bit_width > 32 not yet supported \
                     (unusual in practice; writer would normally emit PLAIN)",
                ));
            }
            let body_bytes = mini_block_size * bit_width as usize / 8;
            let chunk = cur.take(body_bytes)?;

            unpack_buf.clear();
            unpack_indices_into(chunk, mini_block_size, bit_width, &mut unpack_buf)?;

            for &delta_u in unpack_buf.iter() {
                if remaining == 0 {
                    break;
                }
                prev = prev
                    .wrapping_add(block_min_delta)
                    .wrapping_add(delta_u as i128);
                out.push(prev as i64);
                remaining -= 1;
            }
        }
    }
    Ok(out)
}

fn delta_err(msg: &str) -> CodecError {
    CodecError::Decompress(format!("DELTA_BINARY_PACKED: {msg}"))
}
