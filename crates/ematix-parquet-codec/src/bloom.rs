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
    0x4497_4d91,
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

// ============================================================
// Split-Block Bloom Filter builder (write side)
// ============================================================
//
// Symmetric to `SplitBlockBloomFilter` above. Insert values via
// `insert_hash(h)` or `insert_bytes(slice)`, then call
// `into_bytes()` to serialise a `BloomFilterHeader` + bitset that
// the decoder above can read back verbatim.
//
// Sizing the filter: `optimal_num_blocks(distinct_count, fpp)`
// returns the smallest power-of-two num_blocks that achieves the
// requested false-positive probability. Spec recommends p ≈ 0.01
// for column-chunk bloom filters.

/// In-memory SBBF builder. The bitset is owned; insertions OR the
/// per-block masks directly into it.
pub struct SplitBlockBloomFilterBuilder {
    bitset: Vec<u8>,
    num_blocks: u32,
}

impl SplitBlockBloomFilterBuilder {
    /// New empty builder with `num_blocks` 32-byte blocks (so the
    /// bitset is `32 * num_blocks` bytes). `num_blocks` should
    /// come from `optimal_num_blocks` unless the caller has its
    /// own sizing rule.
    pub fn new(num_blocks: u32) -> Self {
        let bytes = 32 * num_blocks as usize;
        Self {
            bitset: vec![0u8; bytes],
            num_blocks,
        }
    }

    /// Insert a precomputed XXHash64(value) into the filter. Use
    /// `insert_bytes` if you want this crate to hash for you.
    pub fn insert_hash(&mut self, h: u64) {
        if self.num_blocks == 0 {
            return;
        }
        let block_idx = ((h >> 32) * self.num_blocks as u64) >> 32;
        let block_off = (block_idx as usize) * 32;
        let h_lo = h as u32;
        for i in 0..8 {
            let mask_bit = (h_lo.wrapping_mul(SALT[i])) >> 27;
            let mask = 1u32 << mask_bit;
            let word_off = block_off + i * 4;
            let mut word = u32::from_le_bytes([
                self.bitset[word_off],
                self.bitset[word_off + 1],
                self.bitset[word_off + 2],
                self.bitset[word_off + 3],
            ]);
            word |= mask;
            let b = word.to_le_bytes();
            self.bitset[word_off] = b[0];
            self.bitset[word_off + 1] = b[1];
            self.bitset[word_off + 2] = b[2];
            self.bitset[word_off + 3] = b[3];
        }
    }

    /// Hash `bytes` with `parquet_xxh64` and insert.
    pub fn insert_bytes(&mut self, bytes: &[u8]) {
        self.insert_hash(parquet_xxh64(bytes));
    }

    /// Number of 32-byte blocks in the bitset (returned by-value
    /// for callers that want to verify post-construction).
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Serialise as `BloomFilterHeader` Thrift followed by the
    /// bitset bytes — the on-disk layout the spec calls for and
    /// `SplitBlockBloomFilter::from_bytes` decodes.
    pub fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::new();
        write_bloom_filter_header(&mut out, self.bitset.len() as i32);
        out.extend_from_slice(&self.bitset);
        out
    }
}

/// Recommended `num_blocks` for a target false-positive
/// probability `fpp` with `n` distinct values. Rounds up to the
/// next power of two (the spec doesn't require this, but most
/// readers assume it). Floors at 1 block (32 bytes).
///
/// Derivation: per the SBBF spec, the false-positive rate is
/// ≈ `(1 - exp(-8n / m))^8` where `m = num_blocks * 256` is the
/// bit count. Solving for `m` given a target fpp:
/// `m = -8n / ln(1 - fpp^(1/8))`.
pub fn optimal_num_blocks(n: usize, fpp: f64) -> u32 {
    if n == 0 {
        return 1;
    }
    let fpp = fpp.clamp(1e-12, 0.5);
    let m = -8.0 * n as f64 / (1.0 - fpp.powf(0.125)).ln();
    let bytes = (m / 8.0).ceil() as u64;
    let mut blocks = (bytes.div_ceil(32)).max(1);
    // Round up to next power of two.
    if !blocks.is_power_of_two() {
        blocks = blocks.next_power_of_two();
    }
    blocks.min(u32::MAX as u64) as u32
}

/// Hand-rolled `BloomFilterHeader` Thrift writer. Equivalent to
/// `metadata_writer::write_bloom_filter_header` would be if the
/// format crate exposed one — but for v0.9.1 we keep the
/// serialisation local to the codec so the bloom-builder PR
/// doesn't have to touch the format crate's metadata writer
/// (that's needed only for the full writer-integration step:
/// emitting bloom filters into the parquet file's body and
/// updating `ColumnMetaData.bloom_filter_offset`).
///
/// Compact-protocol layout:
///   field 1 i32 num_bytes
///   field 2 struct BloomFilterAlgorithm { union — SplitBlock = field 1, void }
///   field 3 struct BloomFilterHash      { union — XxHash     = field 1, void }
///   field 4 struct BloomFilterCompression { union — Uncompressed = field 1, void }
///   stop
fn write_bloom_filter_header(out: &mut Vec<u8>, num_bytes: i32) {
    // Field 1: i32 num_bytes — compact i32 is zigzag varint.
    out.push((1u8 << 4) | 5); // field id 1, type i32 (5)
    write_zigzag_i32(out, num_bytes);

    // Field 2: BloomFilterAlgorithm (struct/union)
    out.push((1u8 << 4) | 12); // field id 2 (delta 1 from id 1), type struct (12)
    {
        // Union field 1: SplitBlockAlgorithm — a struct with no fields, encoded as STOP.
        out.push((1u8 << 4) | 12); // field id 1, type struct (12)
        out.push(0); // inner struct STOP
        out.push(0); // union STOP
    }

    // Field 3: BloomFilterHash (struct/union)
    out.push((1u8 << 4) | 12); // field id 3 (delta 1), type struct
    {
        // Union field 1: XxHash — empty struct
        out.push((1u8 << 4) | 12);
        out.push(0);
        out.push(0);
    }

    // Field 4: BloomFilterCompression (struct/union)
    out.push((1u8 << 4) | 12); // field id 4 (delta 1), type struct
    {
        // Union field 1: Uncompressed — empty struct
        out.push((1u8 << 4) | 12);
        out.push(0);
        out.push(0);
    }

    out.push(0); // outer struct STOP
}

fn write_zigzag_i32(out: &mut Vec<u8>, v: i32) {
    let mut z: u32 = ((v << 1) ^ (v >> 31)) as u32;
    loop {
        if (z & !0x7F) == 0 {
            out.push(z as u8);
            return;
        }
        out.push(((z & 0x7F) | 0x80) as u8);
        z >>= 7;
    }
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

    #[test]
    fn builder_round_trips_through_decoder() {
        // Build a filter, insert a known set of values, serialise,
        // parse with the decoder, confirm every inserted value is
        // reported "possibly present" and a handful of non-inserted
        // values are reported "absent" (fpp permitting).
        let inserted: Vec<&[u8]> = vec![
            b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf", b"hotel",
            b"india", b"juliet",
        ];
        let blocks = optimal_num_blocks(inserted.len(), 0.01);
        let mut b = SplitBlockBloomFilterBuilder::new(blocks);
        for v in &inserted {
            b.insert_bytes(v);
        }
        let bytes = b.into_bytes();
        let filter = SplitBlockBloomFilter::from_bytes(&bytes).expect("decode");

        // Every inserted value must be reported present.
        for v in &inserted {
            assert!(
                filter.contains_bytes(v),
                "inserted value {v:?} reported absent (false negative — impossible for a correct SBBF)"
            );
        }

        // Non-inserted values should mostly be reported absent.
        // 100 distinct strings, expect ≤ ~3 false positives at fpp 0.01.
        let mut fp = 0;
        for i in 0..100 {
            let key = format!("absent_{i}");
            if filter.contains_bytes(key.as_bytes()) {
                fp += 1;
            }
        }
        assert!(fp < 10, "too many false positives: {fp} / 100");
    }

    #[test]
    fn optimal_num_blocks_is_power_of_two_and_grows_with_n() {
        let b1 = optimal_num_blocks(100, 0.01);
        let b2 = optimal_num_blocks(10_000, 0.01);
        let b3 = optimal_num_blocks(1_000_000, 0.01);
        assert!(b1.is_power_of_two());
        assert!(b2.is_power_of_two());
        assert!(b3.is_power_of_two());
        assert!(b1 < b2 && b2 < b3);
    }

    #[test]
    fn empty_builder_serialises_and_decodes() {
        // Edge case: a builder with the smallest valid filter
        // (1 block = 32 bytes bitset) must round-trip.
        let b = SplitBlockBloomFilterBuilder::new(1);
        let bytes = b.into_bytes();
        let filter = SplitBlockBloomFilter::from_bytes(&bytes).expect("decode");
        assert_eq!(filter.num_blocks(), 1);
        // No values inserted → every membership check returns false.
        assert!(!filter.contains_bytes(b"anything"));
    }

    #[test]
    fn builder_insert_hash_and_insert_bytes_agree() {
        let mut a = SplitBlockBloomFilterBuilder::new(4);
        let mut b = SplitBlockBloomFilterBuilder::new(4);
        a.insert_hash(parquet_xxh64(b"x"));
        b.insert_bytes(b"x");
        assert_eq!(a.into_bytes(), b.into_bytes());
    }
}
