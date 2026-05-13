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
