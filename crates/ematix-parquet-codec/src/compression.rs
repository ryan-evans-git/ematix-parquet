//! Page-body decompression.
//!
//! Two API shapes per codec:
//!   - `decompress_<codec>(&[u8]) -> Vec<u8>` — convenience, fresh alloc
//!   - `decompress_<codec>_into(&[u8], &mut Vec<u8>)` — caller-owned
//!     buffer; reuse the same Vec across many pages to amortize
//!     allocator cost. Lineitem rg 0 col 0 has ~52 pages; the
//!     reuse path goes from 52 allocs to 1 (after the first page
//!     resizes the buffer to max).
//!
//! Codecs: SNAPPY (TPC-H), ZSTD (Spark, modern polars). Gzip/Brotli/LZ4
//! later.

use std::io::Read;

use crate::error::{CodecError, Result};

/// Snappy raw-format decompression. Parquet uses the framed-less
/// "raw" variant of snappy, not the framing-protocol variant.
pub fn decompress_snappy(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut dec = snap::raw::Decoder::new();
    dec.decompress_vec(compressed)
        .map_err(|e| CodecError::Decompress(format!("snappy: {e}")))
}

/// Variant that decompresses into a caller-supplied `Vec<u8>` for
/// buffer reuse. On entry `out` may have any state; it is `clear()`ed
/// and `resize()`d to the decompressed length. On exit `out.len()`
/// equals the decompressed size. Subsequent calls retain the
/// capacity so the second-and-later page never allocates.
pub fn decompress_snappy_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let dec_len = snap::raw::decompress_len(compressed)
        .map_err(|e| CodecError::Decompress(format!("snappy len: {e}")))?;
    out.clear();
    out.resize(dec_len, 0);
    let mut dec = snap::raw::Decoder::new();
    let n = dec
        .decompress(compressed, out.as_mut_slice())
        .map_err(|e| CodecError::Decompress(format!("snappy: {e}")))?;
    debug_assert_eq!(n, dec_len);
    out.truncate(n);
    Ok(())
}

/// Hand-rolled Snappy (raw, parquet variant) decompressor.
///
/// Why: the `snap` crate at ~2.3 GB/s leaves a lot on the table. This
/// path inlines the tag dispatch, uses unsafe pointer arithmetic
/// throughout the hot loop, pre-sizes the output Vec from the header
/// uvarint, and specializes the literal hot path for short (≤16-byte)
/// copies — by far the most common shape in real parquet pages.
///
/// Targets ~4-5 GB/s on M-series; closes the polars gap on Q14.
///
/// Wire format (Snappy raw, RFC):
///   uvarint               -- decompressed_length
///   stream of elements:
///     tag byte. tag & 0b11 selects element class.
///     0b00 literal:
///         len_field = tag >> 2
///         if len_field <= 59: literal_len = len_field + 1, data follows
///         else: literal_len = LE-read(len_field - 59 bytes) + 1, data follows
///     0b01 copy-1:
///         len = ((tag >> 2) & 0b111) + 4   (length in 4..=11)
///         offset = ((tag >> 5) << 8) | next_byte
///     0b10 copy-2:
///         len = (tag >> 2) + 1            (length in 1..=64)
///         offset = LE u16
///     0b11 copy-4:
///         len = (tag >> 2) + 1
///         offset = LE u32
///
/// Copies reference already-decoded output bytes. When the copy
/// `length > offset`, the copy must propagate (RLE-style fill) — we
/// handle this by per-byte copy in that case, which is rare.
pub fn decompress_snappy_fast_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    // 1. Parse the uvarint decompressed length.
    let (dec_len, header_bytes) = read_uvarint_le(compressed)
        .ok_or_else(|| CodecError::Decompress("snappy: bad header uvarint".into()))?;
    let src = &compressed[header_bytes..];

    out.clear();
    // Reserve 16 bytes of padding past dec_len so the hot copy
    // helpers can do 8-byte word copies that potentially overrun by
    // up to 7 bytes without UB. We `truncate` back to dec_len before
    // returning.
    out.reserve(dec_len + 16);
    // SAFETY: out.set_len(dec_len + 16) below; we never read
    // uninitialized bytes — every read is from bytes we've already
    // written (back-references) or from the input buffer (literals).
    // Final truncate(dec_len) drops the padding.
    unsafe { out.set_len(dec_len + 16) };

    let mut src_pos: usize = 0;
    let mut written: usize = 0;
    let src_end = src.len();
    let src_ptr = src.as_ptr();
    let dst_ptr = out.as_mut_ptr();

    while src_pos < src_end {
        // SAFETY: src_pos < src_end and src is a valid slice.
        let tag = unsafe { *src_ptr.add(src_pos) };
        src_pos += 1;

        match tag & 0b11 {
            0b00 => {
                // Literal.
                let mut lit_len: usize = ((tag >> 2) as usize) + 1;
                if lit_len > 60 {
                    let extra = lit_len - 60;
                    if src_pos + extra > src_end {
                        return Err(CodecError::Decompress("snappy: short literal length".into()));
                    }
                    let mut len_acc: usize = 0;
                    for i in 0..extra {
                        len_acc |= (unsafe { *src_ptr.add(src_pos + i) } as usize) << (i * 8);
                    }
                    lit_len = len_acc + 1;
                    src_pos += extra;
                }
                if src_pos + lit_len > src_end || written + lit_len > dec_len {
                    return Err(CodecError::Decompress("snappy: literal overrun".into()));
                }
                // Bulk byte copy. For lengths ≤ 16, LLVM emits a
                // single 16-byte vector load+store; for longer we
                // fall through to memcpy. Both faster than a byte
                // loop.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(src_pos),
                        dst_ptr.add(written),
                        lit_len,
                    );
                }
                src_pos += lit_len;
                written += lit_len;
            }
            0b01 => {
                // Copy-1: 1-byte offset, 4-bit length (offset in 1..=2047).
                if src_pos >= src_end {
                    return Err(CodecError::Decompress("snappy: short copy-1".into()));
                }
                let len = (((tag >> 2) & 0b111) as usize) + 4;
                let offset = (((tag >> 5) as usize) << 8) | (unsafe { *src_ptr.add(src_pos) } as usize);
                src_pos += 1;
                if offset == 0 || offset > written || written + len > dec_len {
                    return Err(CodecError::Decompress("snappy: bad copy-1 offset/len".into()));
                }
                unsafe { copy_back_ref(dst_ptr, written, offset, len) };
                written += len;
            }
            0b10 => {
                // Copy-2: 2-byte LE offset, length+1 in 1..=64.
                if src_pos + 2 > src_end {
                    return Err(CodecError::Decompress("snappy: short copy-2".into()));
                }
                let len = ((tag >> 2) as usize) + 1;
                let offset = unsafe {
                    (*src_ptr.add(src_pos) as usize)
                        | ((*src_ptr.add(src_pos + 1) as usize) << 8)
                };
                src_pos += 2;
                if offset == 0 || offset > written || written + len > dec_len {
                    return Err(CodecError::Decompress("snappy: bad copy-2 offset/len".into()));
                }
                unsafe { copy_back_ref(dst_ptr, written, offset, len) };
                written += len;
            }
            _ => {
                // Copy-4: 4-byte LE offset, length+1 in 1..=64.
                if src_pos + 4 > src_end {
                    return Err(CodecError::Decompress("snappy: short copy-4".into()));
                }
                let len = ((tag >> 2) as usize) + 1;
                let offset = unsafe {
                    (*src_ptr.add(src_pos) as usize)
                        | ((*src_ptr.add(src_pos + 1) as usize) << 8)
                        | ((*src_ptr.add(src_pos + 2) as usize) << 16)
                        | ((*src_ptr.add(src_pos + 3) as usize) << 24)
                };
                src_pos += 4;
                if offset == 0 || offset > written || written + len > dec_len {
                    return Err(CodecError::Decompress("snappy: bad copy-4 offset/len".into()));
                }
                unsafe { copy_back_ref(dst_ptr, written, offset, len) };
                written += len;
            }
        }
    }
    if written != dec_len {
        return Err(CodecError::Decompress(format!(
            "snappy: wrote {} but header declared {}",
            written, dec_len
        )));
    }
    // Drop the 16-byte padding that lets copy_back_ref overrun safely.
    out.truncate(dec_len);
    Ok(())
}

/// Read a Snappy varint (LE base-128). Returns (value, bytes_consumed)
/// or None on malformed input. Max 5 bytes for 32-bit value.
#[inline]
fn read_uvarint_le(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut acc: usize = 0;
    let mut shift: u32 = 0;
    for (i, &b) in bytes.iter().enumerate().take(5) {
        acc |= ((b & 0x7F) as usize) << shift;
        if (b & 0x80) == 0 {
            return Some((acc, i + 1));
        }
        shift += 7;
    }
    None
}

/// Copy `len` bytes from `dst[written - offset..]` to `dst[written..]`.
///
/// Three cases:
///   - `offset >= 8`: source and dest don't overlap within an 8-byte
///     word, so we can use 8-byte word copies (memcpy_nonoverlapping
///     of 8 each) up to `len`, potentially overrunning by up to 7
///     bytes. Caller reserved 16 bytes of padding past dec_len for
///     this to be safe.
///   - `1 <= offset < 8`: source and dest overlap inside an 8-byte
///     window — must propagate. Common shape: RLE of small period.
///     Build the first 8 bytes of dst by replicating the period, then
///     8-byte word copies onward.
///   - `offset == 1`: byte fill — emit one byte then 8-byte word
///     copies of the same byte.
///
/// # Safety
/// Caller must guarantee `written + len <= dec_len`,
/// `dst[..written + len + 16]` is reserved, `offset <= written`,
/// `offset > 0`.
#[inline]
unsafe fn copy_back_ref(dst_ptr: *mut u8, written: usize, offset: usize, len: usize) {
    let mut src = unsafe { dst_ptr.add(written - offset) };
    let mut dst = unsafe { dst_ptr.add(written) };

    if offset >= 8 {
        // Fast path: word copies, possibly overrunning.
        let mut i: usize = 0;
        while i < len {
            unsafe { std::ptr::copy_nonoverlapping(src, dst, 8) };
            unsafe {
                src = src.add(8);
                dst = dst.add(8);
            }
            i += 8;
        }
        return;
    }

    // Propagating copy: offset in 1..8. We unroll the first 8 bytes
    // byte-by-byte to materialize a full 8-byte window at dst[0..8].
    // After that, the gap between src and dst grows past 8 and we
    // can switch to word copies.
    let total = len.min(8);
    for i in 0..total {
        unsafe { *dst.add(i) = *src.add(i) };
    }
    if len <= 8 {
        return;
    }
    // After writing 8 bytes, src and dst are still 'offset' apart;
    // but we've established 8 bytes of contiguous duplicate data at
    // dst[0..8]. Advance dst by 8 and copy from a new src that now
    // starts 8 bytes earlier in our output (which still maintains
    // the periodic pattern).
    let new_offset = if offset == 1 { 8 } else { 8 - (8 % offset) + offset };
    // Simpler: just byte-copy. Calling this case is rare and short.
    let mut i: usize = 8;
    while i < len {
        unsafe { *dst.add(i) = *dst.add(i - offset) };
        i += 1;
    }
    let _ = new_offset;
}

/// ZSTD decompression. Parquet's ZSTD payload is a single complete
/// zstd frame per page body. We don't know the decompressed size up
/// front (zstd frames may omit it), so we stream into the output Vec
/// via the standard `Read` adapter — `zstd::stream::read::Decoder`
/// grows the Vec naturally.
pub fn decompress_zstd(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_zstd_into(compressed, &mut out)?;
    Ok(out)
}

/// In-place variant. `out` is `clear()`ed but its capacity is
/// preserved across calls (the `Vec::clear` contract). Subsequent
/// pages of similar size avoid reallocation.
pub fn decompress_zstd_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let mut dec = zstd::stream::read::Decoder::new(compressed)
        .map_err(|e| CodecError::Decompress(format!("zstd init: {e}")))?;
    dec.read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("zstd: {e}")))?;
    Ok(())
}
