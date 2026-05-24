//! [`SourceFingerprint`] computation for a parquet source file.
//!
//! The fingerprint captures four scalars: the byte length of the
//! serialized footer, a CRC32 (IEEE) over those bytes, the file's
//! `num_rows`, and `row_groups.len()`. Together these detect any
//! source-file rewrite that would shift page offsets, change schema,
//! or change row counts — strong enough that a sidecar built against
//! one footer state can be unambiguously rejected if the source has
//! since changed.
//!
//! Not a cryptographic hash; a malicious producer can craft a
//! collision and the only effect is the reader serving stale results
//! to itself. The defense is "don't use sidecars from untrusted
//! sources", same shape as Parquet's own footer.

use ematix_parquet_io::ParquetFile;

use crate::error::{CodecError, Result};
use crate::index::manifest::SourceFingerprint;

/// Compute the fingerprint of `file`'s current footer state. Reading
/// the metadata is required to extract `num_rows` and
/// `num_row_groups`; the footer bytes are already cached on the
/// `ParquetFile`.
pub fn compute_source_fingerprint(file: &ParquetFile) -> Result<SourceFingerprint> {
    let footer = file.footer_bytes();
    let footer_length = u32::try_from(footer.len()).map_err(|_| {
        CodecError::InvalidInput(format!(
            "footer length {} exceeds u32 range (corrupted parquet?)",
            footer.len()
        ))
    })?;
    let footer_crc32 = crc32_ieee(footer);
    let meta = file
        .metadata()
        .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
    let num_rows = meta.num_rows;
    let num_row_groups = u32::try_from(meta.row_groups.len()).map_err(|_| {
        CodecError::InvalidInput(format!(
            "num_row_groups {} exceeds u32 range",
            meta.row_groups.len()
        ))
    })?;
    Ok(SourceFingerprint {
        footer_length,
        footer_crc32,
        num_rows,
        num_row_groups,
    })
}

// ============================================================
// CRC32-IEEE (reflected, polynomial 0xEDB88320)
// ============================================================
//
// Table-driven, byte-at-a-time. The 256-entry table is built at
// compile time via `const fn`, so this adds 1 KiB to .rodata and
// nothing to startup. Matches the CRC32 zlib/gzip use — same
// polynomial as Parquet's `Snappy` framing CRC and what every
// general-purpose `crc32` crate emits.
//
// The footer is at most ~hundreds of KB for the largest files we
// touch; bitwise-per-byte over a table is ~1 GB/s on a modern core,
// which dwarfs the footer-read I/O. No need for a hardware CRC32
// instruction (would require `target_feature` for SSE 4.2 or ARMv8
// CRC32 and runtime detection — not worth the surface for a once-
// per-file call).

const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut c = i;
        let mut j = 0;
        while j < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            j += 1;
        }
        table[i as usize] = c;
        i += 1;
    }
    table
}

/// CRC32-IEEE (also known as CRC-32, the one zlib / gzip / PNG use).
/// Public so the sidecar builder can verify it computes the same
/// number a downstream tool would.
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        let idx = (crc ^ b as u32) & 0xFF;
        crc = (crc >> 8) ^ CRC32_TABLE[idx as usize];
    }
    !crc
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vectors() {
        // Standard test vectors for CRC-32 / ISO HDLC / zlib.
        // Source: every reference implementation in existence
        // (zlib, libpng, RFC 1952). If these change, this file
        // changed; if they don't match what `gzip --no-name`
        // would emit, our impl is wrong.
        assert_eq!(crc32_ieee(b""), 0x0000_0000);
        assert_eq!(crc32_ieee(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32_ieee(b"abc"), 0x3524_41C2);
        assert_eq!(crc32_ieee(b"message digest"), 0x2015_9D7F);
        assert_eq!(crc32_ieee(b"abcdefghijklmnopqrstuvwxyz"), 0x4C27_50BD);
        assert_eq!(
            crc32_ieee(b"123456789"),
            // "check" value from CRC-32/ISO-HDLC
            0xCBF4_3926
        );
    }

    #[test]
    fn crc32_stable_under_repeated_calls() {
        // Belt-and-braces: stateless impl, but make sure it really
        // is stateless (no global cached "last result" trick).
        let data = b"the quick brown fox jumps over the lazy dog";
        let a = crc32_ieee(data);
        let b = crc32_ieee(data);
        let c = crc32_ieee(data);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn crc32_long_buffer_streams_correctly() {
        // Make sure splitting the same buffer over multiple calls
        // would land on a different value — i.e. we're not
        // accidentally re-initialising state mid-buffer. (We have
        // no `update` method; this just confirms the byte-at-a-time
        // loop is doing what it claims.)
        let buf: Vec<u8> = (0u8..255).cycle().take(8192).collect();
        let crc = crc32_ieee(&buf);
        let crc_again = crc32_ieee(&buf);
        assert_eq!(crc, crc_again);
        // And a different buffer must give a different number.
        let mut buf2 = buf.clone();
        buf2[0] ^= 1;
        assert_ne!(crc, crc32_ieee(&buf2));
    }
}
