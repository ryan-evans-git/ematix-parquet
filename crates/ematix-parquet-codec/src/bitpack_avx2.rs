//! AVX2-specialized bit-unpackers for x86_64.
//!
//! Mirror of `bitpack_neon` for the x86 side. Runtime feature
//! detection picks AVX2 vs scalar in `bitpack::unpack_*_into`; this
//! module never assumes the CPU has AVX2 — the dispatcher checks
//! `is_x86_feature_detected!("avx2")` first.
//!
//! ## Coverage status (Π.12)
//!
//! - **bw=16** — byte-aligned, the simplest kernel. ✓ Shipped (Π.12a).
//! - bw=12 / 14 / 15 / 17 / 18 — planned for Π.12b–f, mirror the
//!   NEON shapes via shuffles + variable shifts (`_mm256_sllv_epi32`,
//!   `_mm256_srlv_epi32`, etc.).
//!
//! ## Why bw=16 first
//!
//! The NEON bw=16 path is one load + one widen + one store. AVX2's
//! `_mm256_cvtepu16_epi32` does the widen+extend in one instruction:
//! load 16 bytes (8 u16) into a `__m128i`, widen straight into a
//! 256-bit register of 8 u32, store. No shuffles, no shifts. Same
//! shape as NEON, just using AVX2 ops. Establishes the dispatch
//! pattern + CI x86 runner end-to-end with the lowest risk of a
//! subtle intrinsic bug.

#![cfg(target_arch = "x86_64")]

use crate::error::{CodecError, Result};

/// Bit-unpack `num_values` 16-bit indices from `packed` into `out`.
/// AVX2 path; caller checked `is_x86_feature_detected!("avx2")`
/// before dispatching here.
pub fn unpack_indices_into_avx2_bw16(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values * 2;
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw16: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // bw=16 → exactly 16 bytes per 8-value block. The AVX2 widen
    // instruction _mm256_cvtepu16_epi32 reads a 128-bit register
    // (8 u16) and produces a 256-bit register (8 u32). No overrun
    // risk: total input is exactly `num_values * 2` bytes.
    unsafe {
        unpack_avx2_bw16_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw16(&packed[processed * 2..], remaining, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw16_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        // 1 × 128-bit load: 16 bytes = 8 u16 values.
        let v128: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        // Zero-extend 8 u16 → 8 u32 in a 256-bit register.
        let widened: __m256i = _mm256_cvtepu16_epi32(v128);
        // 32-byte store into the output.
        _mm256_storeu_si256(out_ptr.add(blk * 8) as *mut __m256i, widened);
        src_ptr = src_ptr.add(16);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=16) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw16` but using AVX2's
/// 256-bit widen.
pub fn unpack_lookup_into_avx2_bw16<T: Copy>(
    packed: &[u8],
    num_values: usize,
    dict: &[T],
    out: &mut Vec<T>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    if dict.is_empty() {
        return Err(CodecError::DictIndexOutOfRange {
            index: 0,
            dict_size: 0,
        });
    }
    let required_bytes = num_values * 2;
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw16 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut staging = [0u32; 8];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        // 16-bit indices cap at 65535; dict size beyond that is rare
        // (would need a column with > 64K distinct values). Use the
        // bounds-checked path always — branch is predictable across
        // the whole page.
        unpack_avx2_bw16_into_staging(packed, full_blocks, &mut staging, |idxs| {
            for (lane, &i) in idxs.iter().enumerate() {
                let i_usize = i as usize;
                if i_usize >= dict_size {
                    bad_idx = Some(i);
                    return Err(CodecError::DictIndexOutOfRange {
                        index: i,
                        dict_size,
                    });
                }
                *out_ptr.add(written + lane) = *dict_ptr.add(i_usize);
            }
            written += 8;
            Ok(())
        })?;
        out.set_len(out_start_len + written);
    }
    if let Some(i) = bad_idx {
        return Err(CodecError::DictIndexOutOfRange {
            index: i,
            dict_size,
        });
    }

    let processed = full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw16(&packed[processed * 2..], remaining, &mut tail_idxs);
        for &i in &tail_idxs {
            let i_usize = i as usize;
            if i_usize >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(i_usize);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

/// AVX2 bw=16 staging helper, mirror of
/// `bitpack_neon::unpack_neon_bw16_into_staging`. Calls `sink` once
/// per 8-value block with the unpacked indices in a `[u32; 8]`
/// stack buffer; the sink owns the per-lane gather work.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw16_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v128: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let widened: __m256i = _mm256_cvtepu16_epi32(v128);
        _mm256_storeu_si256(staging_ptr as *mut __m256i, widened);
        sink(staging)?;
        src_ptr = src_ptr.add(16);
    }
    Ok(())
}

/// Streaming bit-buffer scalar fallback for a small remainder.
/// Used for the tail (< 8 values) in AVX2-specialized paths.
fn scalar_bw16(packed: &[u8], n: usize, out: &mut Vec<u32>) {
    // bw=16 is byte-aligned → just read 2 bytes per value.
    for i in 0..n {
        let lo = packed[i * 2] as u32;
        let hi = packed[i * 2 + 1] as u32;
        out.push(lo | (hi << 8));
    }
}

// ============================================================
// bw=14 — direct mirror of NEON
// ============================================================

/// Bit-unpack `num_values` 14-bit indices from `packed` into `out`.
///
/// Per 8-row block: 14 input bytes → 8 u32 outputs.
///
/// Mirror of `bitpack_neon::unpack_indices_into_neon_bw14`. Same
/// shuffle indices and shift constants; only the intrinsics differ:
///   - NEON `vqtbl1q_u8` → SSSE3 `_mm_shuffle_epi8` (in-lane byte
///     table lookup, same semantics).
///   - NEON `vshlq_s32` with negative lane shifts → AVX2
///     `_mm_srlv_epi32` with positive lane shifts.
pub fn unpack_indices_into_avx2_bw14(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 14).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw14: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // The kernel reads 16 bytes per block but only consumes 14.
    // Final iteration could overrun by 2 bytes if packed.len() is
    // exactly 14 * full_blocks. Drop the last block to scalar in
    // that case (same safety dance as NEON bw=14).
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 14 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw14_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 14 / 8..], remaining, 14, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw14_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    // Byte-level shuffle indices — identical to NEON bw=14.
    // Lanes 0-3: u32 windows starting at bytes 0 / 1 / 3 / 5.
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8);
    // Lanes 4-7: u32 windows starting at bytes 7 / 8 / 10 / 12.
    let shuffle_hi: __m128i =
        _mm_setr_epi8(7, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15);
    // Per-lane right shift counts, applied via _mm_srlv_epi32.
    // Magnitudes match NEON's [0, 6, 4, 2] (NEON uses negative
    // shifts on vshlq_s32 to mean right-shift; AVX2 takes
    // positive shift counts directly).
    let shifts: __m128i = _mm_setr_epi32(0, 6, 4, 2);
    let mask: __m128i = _mm_set1_epi32(0x3FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v0, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);

        _mm_storeu_si128(out_ptr.add(blk * 8) as *mut __m128i, lo_masked);
        _mm_storeu_si128(out_ptr.add(blk * 8 + 4) as *mut __m128i, hi_masked);

        src_ptr = src_ptr.add(14);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=14) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw14`.
pub fn unpack_lookup_into_avx2_bw14<T: Copy>(
    packed: &[u8],
    num_values: usize,
    dict: &[T],
    out: &mut Vec<T>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    if dict.is_empty() {
        return Err(CodecError::DictIndexOutOfRange {
            index: 0,
            dict_size: 0,
        });
    }
    let required_bytes = (num_values * 14).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw14 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 14 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut staging = [0u32; 8];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        if dict_size > (1 << 14) - 1 {
            // Bounds-safe fast path: every 14-bit index fits in dict.
            unpack_avx2_bw14_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            // Bounds-checked path; branch is predictable across page.
            unpack_avx2_bw14_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let i_usize = i as usize;
                    if i_usize >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(i_usize);
                }
                written += 8;
                Ok(())
            })?;
        }
        out.set_len(out_start_len + written);
    }
    if let Some(i) = bad_idx {
        return Err(CodecError::DictIndexOutOfRange {
            index: i,
            dict_size,
        });
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 14 / 8..], remaining, 14, &mut tail_idxs);
        for &i in &tail_idxs {
            let i_usize = i as usize;
            if i_usize >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(i_usize);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw14_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8);
    let shuffle_hi: __m128i =
        _mm_setr_epi8(7, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15);
    let shifts: __m128i = _mm_setr_epi32(0, 6, 4, 2);
    let mask: __m128i = _mm_set1_epi32(0x3FFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v0, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);
        _mm_storeu_si128(staging_ptr as *mut __m128i, lo_masked);
        _mm_storeu_si128(staging_ptr.add(4) as *mut __m128i, hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(14);
    }
    Ok(())
}

/// Generic scalar streaming bit-buffer for any bit_width. Used for
/// tail handling in AVX2-specialized paths beyond the byte-aligned
/// bw=16 case.
fn scalar_bw_n(packed: &[u8], n: usize, bit_width: u8, out: &mut Vec<u32>) {
    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bit_width) - 1
    };
    let mut buf: u64 = 0;
    let mut bits: u32 = 0;
    let mut byte_idx = 0usize;
    for _ in 0..n {
        while bits < bit_width as u32 {
            buf |= (packed[byte_idx] as u64) << bits;
            byte_idx += 1;
            bits += 8;
        }
        out.push((buf & mask) as u32);
        buf >>= bit_width;
        bits -= bit_width as u32;
    }
}
