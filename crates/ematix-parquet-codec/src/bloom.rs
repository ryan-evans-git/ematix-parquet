//! Parquet Split-Block Bloom Filter (SBBF) decoder.
//!
//! Spec: <https://github.com/apache/parquet-format/blob/master/BloomFilter.md>
//!
//! The bitset is a sequence of 32-byte blocks; each block is 8 × 32-bit
//! words. To check membership of a 64-bit hash:
//!   1. Pick a block: `block_idx = ((h_hi32) * num_blocks) >> 32`
//!   2. Compute eight 1-bit masks from the low 32 bits of the hash and
//!      a fixed salt table; each mask targets one word of the block.
//!   3. The value is *possibly* present iff every word AND-mask is
//!      non-zero. Any zero → definitively absent.
//!
//! The hash is XXHash64 with seed 0 of the value's PLAIN-encoded bytes
//! (LE for fixed-width primitives, raw bytes for BYTE_ARRAY without
//! the length prefix). XXHash64 is implemented inline below to keep
//! the dep tree small.

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{
    read_bloom_filter_header, BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHash,
};

use crate::error::{CodecError, Result};

/// Eight magic 32-bit constants used by the SBBF mask computation.
/// These come from the Parquet spec verbatim.
const SALT: [u32; 8] = [
    0x47b6_137b,
    0x44974_d91 & 0xFFFF_FFFF,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Block-bloom-filter view over a borrowed bitset.
///
/// Construction parses the bloom-filter header (which lives just
/// before the bitset bytes) to validate the algorithm/hash/compression
/// match what we support. Today only the spec-defined SplitBlock +
/// XxHash + Uncompressed combination is in the wild; any other
/// combination is rejected explicitly.
pub struct SplitBlockBloomFilter<'a> {
    bitset: &'a [u8],
}

impl<'a> SplitBlockBloomFilter<'a> {
    /// Decode the header at `bytes[0..]`, then attach the bitset
    /// `bytes[header_len..header_len + num_bytes]`.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let mut cur = Cursor::new(bytes);
        let header = read_bloom_filter_header(&mut cur).map_err(format_to_codec)?;
        if !matches!(header.algorithm, BloomFilterAlgorithm::SplitBlock) {
            return Err(CodecError::Unsupported(format!(
                "bloom filter algorithm not supported: {:?}",
                header.algorithm
            )));
        }
        if !matches!(header.hash, BloomFilterHash::XxHash) {
            return Err(CodecError::Unsupported(format!(
                "bloom filter hash not supported: {:?}",
                header.hash
            )));
        }
        if !matches!(header.compression, BloomFilterCompression::Uncompressed) {
            return Err(CodecError::Unsupported(format!(
                "bloom filter compression not supported: {:?}",
                header.compression
            )));
        }
        let num_bytes = header.num_bytes as usize;
        if num_bytes % 32 != 0 {
            return Err(CodecError::InvalidInput(format!(
                "bloom filter num_bytes must be a multiple of 32, got {num_bytes}"
            )));
        }
        let header_len = cur.position();
        let bitset_end = header_len + num_bytes;
        if bytes.len() < bitset_end {
            return Err(CodecError::InvalidInput(format!(
                "bloom filter byte buffer too short: have {}, need {}",
                bytes.len(),
                bitset_end
            )));
        }
        Ok(Self {
            bitset: &bytes[header_len..bitset_end],
        })
    }

    /// Number of 32-byte blocks in the filter.
    pub fn num_blocks(&self) -> usize {
        self.bitset.len() / 32
    }

    /// Membership check for a precomputed XXHash64 of the value
    /// bytes. Returns `false` only when the value is *definitively
    /// absent*; `true` means *possibly present* (the standard bloom
    /// filter semantics).
    pub fn contains_hash(&self, h: u64) -> bool {
        let num_blocks = self.num_blocks() as u64;
        if num_blocks == 0 {
            return false;
        }
        // Block selection by multiply-shift on the upper 32 bits.
        let block_idx = ((h >> 32) * num_blocks) >> 32;
        let block_off = (block_idx as usize) * 32;
        let block = &self.bitset[block_off..block_off + 32];
        let h_lo = h as u32;
        for i in 0..8 {
            let mask_bit = (h_lo.wrapping_mul(SALT[i])) >> 27;
            let mask = 1u32 << mask_bit;
            let word_off = i * 4;
            let word = u32::from_le_bytes([
                block[word_off],
                block[word_off + 1],
                block[word_off + 2],
                block[word_off + 3],
            ]);
            if word & mask == 0 {
                return false;
            }
        }
        true
    }

    /// Convenience: hash an arbitrary byte slice with the parquet
    /// hash function (XXHash64 seed 0) and check membership.
    pub fn contains_bytes(&self, bytes: &[u8]) -> bool {
        self.contains_hash(parquet_xxh64(bytes))
    }
}

fn format_to_codec(e: ematix_parquet_format::error::FormatError) -> CodecError {
    CodecError::InvalidInput(format!("format: {e}"))
}

// ---- XXHash64 ------------------------------------------------------
//
// Parquet uses XXHash64 with seed=0. Inline implementation rather
// than pulling in a crate (small surface, well-specified).

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

/// XXHash64 with seed 0 — the hash function the Parquet bloom-filter
/// spec mandates.
pub fn parquet_xxh64(input: &[u8]) -> u64 {
    parquet_xxh64_seed(input, 0)
}

/// XXHash64 with arbitrary seed. Exposed for completeness; parquet
/// always uses seed=0.
pub fn parquet_xxh64_seed(input: &[u8], seed: u64) -> u64 {
    let len = input.len();
    let mut acc: u64;
    let mut i: usize = 0;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        while i + 32 <= len {
            v1 = round(v1, read_u64_le(&input[i..]));
            v2 = round(v2, read_u64_le(&input[i + 8..]));
            v3 = round(v3, read_u64_le(&input[i + 16..]));
            v4 = round(v4, read_u64_le(&input[i + 24..]));
            i += 32;
        }
        acc = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        acc = merge_round(acc, v1);
        acc = merge_round(acc, v2);
        acc = merge_round(acc, v3);
        acc = merge_round(acc, v4);
    } else {
        acc = seed.wrapping_add(P5);
    }

    acc = acc.wrapping_add(len as u64);

    while i + 8 <= len {
        let k = round(0, read_u64_le(&input[i..]));
        acc ^= k;
        acc = acc.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= len {
        let k = read_u32_le(&input[i..]) as u64;
        acc ^= k.wrapping_mul(P1);
        acc = acc.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    while i < len {
        let k = input[i] as u64;
        acc ^= k.wrapping_mul(P5);
        acc = acc.rotate_left(11).wrapping_mul(P1);
        i += 1;
    }

    // Final avalanche.
    acc ^= acc >> 33;
    acc = acc.wrapping_mul(P2);
    acc ^= acc >> 29;
    acc = acc.wrapping_mul(P3);
    acc ^= acc >> 32;
    acc
}

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    let v = round(0, val);
    (acc ^ v).wrapping_mul(P1).wrapping_add(P4)
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    // XXHash64 reference vectors. The values come from the canonical
    // XXHash64 test suite (xxhsum), seed=0.
    #[test]
    fn xxh64_known_vectors_seed_0() {
        // Empty input.
        assert_eq!(parquet_xxh64(b""), 0xEF46_DB37_51D8_E999);
        // Single byte.
        assert_eq!(parquet_xxh64(b"a"), 0xD24E_C4F1_A98C_6E5B);
        // Short string.
        assert_eq!(parquet_xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
        // Cross the 32-byte boundary.
        let s: &[u8] = b"The quick brown fox jumps over the lazy dog";
        assert_eq!(parquet_xxh64(s), 0x0B24_2D36_1FDA_71BC);
    }

    #[test]
    fn xxh64_input_lengths_avoid_panics() {
        // Walk lengths 0..=80 to exercise the alignment branches
        // (>=32, >=8 tail, >=4 tail, byte tail).
        let data: Vec<u8> = (0u8..80).collect();
        for len in 0..=80 {
            let _ = parquet_xxh64(&data[..len]);
        }
    }
}
