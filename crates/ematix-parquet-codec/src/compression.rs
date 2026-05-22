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
//! For LZ4_RAW the parquet wire format omits the uncompressed length, so
//! the `_into` variant additionally requires the caller pass the size
//! (carried in PageHeader.uncompressed_page_size). Brotli/Gzip/ZSTD have
//! no embedded size limit either, so their `_into` variants take a cap
//! to guard against DoS via malformed input that expands to many GB.

use std::io::Read;

use crate::error::{CodecError, Result};

/// Safety margin applied to `uncompressed_page_size` from the page
/// header when capping read_to_end-style decompression. Honest writers
/// declare exact sizes; we allow a small overshoot for off-by-one
/// quirks but anything beyond rejects as malformed input.
const DECOMPRESS_CAP_SLACK: usize = 64;

fn cap_with_slack(declared: usize) -> usize {
    declared.saturating_add(DECOMPRESS_CAP_SLACK)
}

fn check_cap(actual: usize, cap: usize, codec: &str) -> Result<()> {
    if actual > cap {
        return Err(CodecError::Decompress(format!(
            "{codec}: decompressed {actual} bytes exceeds cap {cap}"
        )));
    }
    Ok(())
}

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
///
/// Unbounded: grows `out` until the zstd stream ends. For trusted
/// input. Use `decompress_zstd_into_capped` when the caller knows the
/// expected size (e.g. the parquet page header).
pub fn decompress_zstd_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let mut dec = zstd::stream::read::Decoder::new(compressed)
        .map_err(|e| CodecError::Decompress(format!("zstd init: {e}")))?;
    dec.read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("zstd: {e}")))?;
    Ok(())
}

/// In-place variant with a max-uncompressed cap. Pre-reserves
/// `uncompressed_size` bytes (avoiding repeated `Vec` regrowth) and
/// errors if the stream produces more than cap+slack bytes. Use this
/// when reading parquet pages — `PageHeader.uncompressed_page_size`
/// supplies the size.
pub fn decompress_zstd_into_capped(
    compressed: &[u8],
    uncompressed_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.reserve(uncompressed_size);
    let cap = cap_with_slack(uncompressed_size);
    let mut dec = zstd::stream::read::Decoder::new(compressed)
        .map_err(|e| CodecError::Decompress(format!("zstd init: {e}")))?;
    // Take cap+1 so an over-large stream returns an error rather than
    // silently truncating.
    dec.by_ref()
        .take((cap as u64).saturating_add(1))
        .read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("zstd: {e}")))?;
    check_cap(out.len(), cap, "zstd")?;
    Ok(())
}

// ---- GZIP (read) ---------------------------------------------------------

/// GZIP decompression. Parquet bodies are full gzip streams (RFC 1952
/// envelope with header + checksum); `flate2`'s `GzDecoder` reads them.
pub fn decompress_gzip(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_gzip_into(compressed, &mut out)?;
    Ok(out)
}

/// Unbounded. Use `decompress_gzip_into_capped` for parquet pages.
pub fn decompress_gzip_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let mut dec = flate2::read::GzDecoder::new(compressed);
    dec.read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("gzip: {e}")))?;
    Ok(())
}

/// Capped variant; errors if the decompressed stream exceeds cap+slack.
pub fn decompress_gzip_into_capped(
    compressed: &[u8],
    uncompressed_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.reserve(uncompressed_size);
    let cap = cap_with_slack(uncompressed_size);
    let dec = flate2::read::GzDecoder::new(compressed);
    dec.take((cap as u64).saturating_add(1))
        .read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("gzip: {e}")))?;
    check_cap(out.len(), cap, "gzip")?;
    Ok(())
}

// ---- Brotli (read) -------------------------------------------------------

pub fn decompress_brotli(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_brotli_into(compressed, &mut out)?;
    Ok(out)
}

/// Unbounded. Use `decompress_brotli_into_capped` for parquet pages.
pub fn decompress_brotli_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let mut dec = brotli::Decompressor::new(compressed, 4096);
    dec.read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("brotli: {e}")))?;
    Ok(())
}

/// Capped variant; errors if the decompressed stream exceeds cap+slack.
pub fn decompress_brotli_into_capped(
    compressed: &[u8],
    uncompressed_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.reserve(uncompressed_size);
    let cap = cap_with_slack(uncompressed_size);
    let dec = brotli::Decompressor::new(compressed, 4096);
    dec.take((cap as u64).saturating_add(1))
        .read_to_end(out)
        .map_err(|e| CodecError::Decompress(format!("brotli: {e}")))?;
    check_cap(out.len(), cap, "brotli")?;
    Ok(())
}

// ---- LZ4_RAW (read) ------------------------------------------------------

/// LZ4_RAW decompression. Parquet's LZ4_RAW is one or more lz4 blocks
/// concatenated, where every block is preceded by a 4-byte little-endian
/// header giving the *compressed* length of that block. The reader
/// keeps consuming blocks until the input is exhausted. This matches
/// what parquet-rs does for the LZ4_RAW codec.
pub fn decompress_lz4_raw(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_lz4_raw_into(compressed, &mut out)?;
    Ok(out)
}

/// Size-less LZ4_RAW decode. Parquet doesn't actually emit the
/// uncompressed length in the page body — the writer puts it in
/// `PageHeader.uncompressed_page_size`. Prefer
/// `decompress_lz4_raw_into_sized` when the size is known.
///
/// The previous implementation called `lz4_flex::block::decompress`
/// with `compressed.len() * 255` as a worst-case cap, which allocates
/// a ~190 MB scratch buffer per typical parquet page and made LZ4_RAW
/// ~45× slower than Snappy on Q06 SF=10. This variant exists for
/// callers (tests, file-level reads) that genuinely don't have a
/// size; it scans the LZ4 block tags to compute the output length
/// before allocating.
pub fn decompress_lz4_raw_into(compressed: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let uncompressed_size = lz4_raw_uncompressed_size(compressed)?;
    out.resize(uncompressed_size, 0);
    let n = lz4_flex::block::decompress_into(compressed, out)
        .map_err(|e| CodecError::Decompress(format!("lz4_raw: {e}")))?;
    out.truncate(n);
    Ok(())
}

/// Size-known LZ4_RAW decode. The fast path for parquet reads —
/// pre-sizes `out` from the page header and decodes directly into
/// it with no intermediate allocation.
pub fn decompress_lz4_raw_into_sized(
    compressed: &[u8],
    uncompressed_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.resize(uncompressed_size, 0);
    let n = lz4_flex::block::decompress_into(compressed, out)
        .map_err(|e| CodecError::Decompress(format!("lz4_raw: {e}")))?;
    if n != uncompressed_size {
        return Err(CodecError::Decompress(format!(
            "lz4_raw: decoded {n} bytes, page header declared {uncompressed_size}"
        )));
    }
    Ok(())
}

/// Walk an LZ4 block at the protocol level to compute the output size.
/// Used only by the size-less `decompress_lz4_raw_into` path; parquet
/// callers should use `decompress_lz4_raw_into_sized`.
///
/// LZ4 block format (per LZ4 frame format, block payload only):
///   sequence = literal_len_tag, copy_len_tag := token byte
///     literal_len: (token >> 4) plus any 0xFF chain bytes
///     literal bytes follow
///     offset: 2 bytes little-endian (absent on last sequence)
///     match_len: (token & 0x0F) + 4 plus any 0xFF chain bytes
fn lz4_raw_uncompressed_size(buf: &[u8]) -> Result<usize> {
    let mut i = 0;
    let mut out_len: usize = 0;
    let n = buf.len();
    while i < n {
        let token = buf[i] as usize;
        i += 1;
        // Literal length.
        let mut lit_len = token >> 4;
        if lit_len == 15 {
            while i < n {
                let b = buf[i] as usize;
                i += 1;
                lit_len += b;
                if b != 0xFF {
                    break;
                }
            }
        }
        if i + lit_len > n {
            return Err(CodecError::Decompress(
                "lz4_raw: literal runs past end of buffer".into(),
            ));
        }
        i += lit_len;
        out_len += lit_len;
        // Last sequence has no match.
        if i == n {
            break;
        }
        // Skip 2-byte offset.
        if i + 2 > n {
            return Err(CodecError::Decompress(
                "lz4_raw: truncated match offset".into(),
            ));
        }
        i += 2;
        // Match length.
        let mut match_len = (token & 0x0F) + 4;
        if (token & 0x0F) == 15 {
            while i < n {
                let b = buf[i] as usize;
                i += 1;
                match_len += b;
                if b != 0xFF {
                    break;
                }
            }
        }
        out_len += match_len;
    }
    Ok(out_len)
}

// ---- compression (write path) --------------------------------------------

/// Snappy raw-format compression. Inverse of `decompress_snappy`.
/// Parquet uses the framed-less raw variant — no Snappy framing
/// header is added.
pub fn compress_snappy(uncompressed: &[u8]) -> Result<Vec<u8>> {
    let mut enc = snap::raw::Encoder::new();
    enc.compress_vec(uncompressed)
        .map_err(|e| CodecError::Decompress(format!("snappy encode: {e}")))
}

/// ZSTD compression at the default level (matches most parquet writers).
/// One complete frame per call. Inverse of `decompress_zstd`.
pub fn compress_zstd(uncompressed: &[u8]) -> Result<Vec<u8>> {
    compress_zstd_at_level(uncompressed, zstd::DEFAULT_COMPRESSION_LEVEL)
}

/// ZSTD compression at an explicit level. Higher → smaller output,
/// slower encode. Range matches the upstream `zstd` crate (1..=22).
pub fn compress_zstd_at_level(uncompressed: &[u8], level: i32) -> Result<Vec<u8>> {
    zstd::stream::encode_all(uncompressed, level)
        .map_err(|e| CodecError::Decompress(format!("zstd encode: {e}")))
}

/// GZIP compression at flate2's default level (6). Produces a complete
/// gzip stream (with header + CRC) — the inverse of `decompress_gzip`.
pub fn compress_gzip(uncompressed: &[u8]) -> Result<Vec<u8>> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    use std::io::Write as _;
    enc.write_all(uncompressed)
        .map_err(|e| CodecError::Decompress(format!("gzip encode: {e}")))?;
    enc.finish()
        .map_err(|e| CodecError::Decompress(format!("gzip finish: {e}")))
}

/// Brotli compression at quality 6 (a balanced choice between speed and
/// ratio for parquet-shaped payloads). lgwindow = 22 follows the
/// brotli crate default.
pub fn compress_brotli(uncompressed: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(uncompressed);
    let mut enc = brotli::CompressorReader::new(&mut input, 4096, 6, 22);
    enc.read_to_end(&mut out)
        .map_err(|e| CodecError::Decompress(format!("brotli encode: {e}")))?;
    Ok(out)
}

/// LZ4_RAW compression: one lz4 block, no framing, no length prefix.
/// Inverse of `decompress_lz4_raw`. The Parquet writer that consumes
/// this output is responsible for stamping the uncompressed size on
/// the page header so the reader can size its output buffer.
pub fn compress_lz4_raw(uncompressed: &[u8]) -> Result<Vec<u8>> {
    Ok(lz4_flex::block::compress(uncompressed))
}
