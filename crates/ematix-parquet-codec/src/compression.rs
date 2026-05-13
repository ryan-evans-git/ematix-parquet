//! Page-body decompression.
//!
//! Two API shapes:
//!   - `decompress_snappy(&[u8]) -> Vec<u8>` — convenience, fresh alloc
//!   - `decompress_snappy_into(&[u8], &mut Vec<u8>)` — caller-owned
//!     buffer; reuse the same Vec across many pages to amortize
//!     allocator cost. Lineitem rg 0 col 0 has ~52 pages; the
//!     reuse path goes from 52 allocs to 1 (after the first page
//!     resizes the buffer to max).

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
