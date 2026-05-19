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

// ============================================================
// Π.12c — bw=15
//
// 8 values × 15 bits = 120 bits = 15 bytes per block.
// Unlike bw=14, the 8 lanes don't all fit within a single 128-bit
// load — lanes 4-7 sit at byte offsets [7..15], so we load a
// second 128-bit window at +7 to feed the high half. Mirror of
// `bitpack_neon::unpack_indices_into_neon_bw15` — same shuffle
// indices, same right-shift magnitudes, same 0x7FFF mask.
// ============================================================

/// AVX2 bit-unpack for bit_width = 15 → u32 indices.
pub fn unpack_indices_into_avx2_bw15(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 15).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw15: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // The kernel does two 16-byte loads per block; the second is at
    // src+7, reaching byte 7+16=23 from the block start. Last block
    // must have at least 24 bytes available from its start.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 15 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw15_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 15 / 8..], remaining, 15, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw15_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    // Lanes 0-3: u32 windows from v0 at byte offsets [0, 1, 3, 5].
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8);
    // Lanes 4-7: u32 windows from v_hi (= v0 shifted by +7 bytes)
    // at byte offsets [0, 2, 4, 6] within v_hi.
    let shuffle_hi: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    // Per-lane right shifts. NEON uses negative-on-vshlq_s32; AVX2
    // takes positive counts directly on _mm_srlv_epi32.
    let shifts_lo: __m128i = _mm_setr_epi32(0, 7, 6, 5);
    let shifts_hi: __m128i = _mm_setr_epi32(4, 3, 2, 1);
    let mask: __m128i = _mm_set1_epi32(0x7FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v_hi: __m128i = _mm_loadu_si128(src_ptr.add(7) as *const __m128i);

        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v_hi, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts_lo);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts_hi);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);

        _mm_storeu_si128(out_ptr.add(blk * 8) as *mut __m128i, lo_masked);
        _mm_storeu_si128(out_ptr.add(blk * 8 + 4) as *mut __m128i, hi_masked);

        src_ptr = src_ptr.add(15);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=15) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw15`.
pub fn unpack_lookup_into_avx2_bw15<T: Copy>(
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
    let required_bytes = (num_values * 15).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw15 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 15 * (full_blocks - 1) + 24 {
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

        if dict_size > (1 << 15) - 1 {
            unpack_avx2_bw15_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw15_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        scalar_bw_n(&packed[processed * 15 / 8..], remaining, 15, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw15_into_staging<F>(
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
    let shuffle_hi: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let shifts_lo: __m128i = _mm_setr_epi32(0, 7, 6, 5);
    let shifts_hi: __m128i = _mm_setr_epi32(4, 3, 2, 1);
    let mask: __m128i = _mm_set1_epi32(0x7FFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v_hi: __m128i = _mm_loadu_si128(src_ptr.add(7) as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v_hi, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts_lo);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts_hi);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);
        _mm_storeu_si128(staging_ptr as *mut __m128i, lo_masked);
        _mm_storeu_si128(staging_ptr.add(4) as *mut __m128i, hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(15);
    }
    Ok(())
}

// ============================================================
// Π.12d — bw=17
//
// 8 values × 17 bits = 136 bits = 17 bytes per block.
// Like bw=15, the 8 lanes don't fit in a single 128-bit load —
// we load two 128-bit windows at +0 and +8 and use a single
// shared shuffle [0,1,2,3, 2,3,4,5, 4,5,6,7, 6,7,8,9] for both
// halves (offset symmetry across bytes 0..16 vs 8..24).
// Mirror of `bitpack_neon::unpack_indices_into_neon_bw17`.
// ============================================================

/// AVX2 bit-unpack for bit_width = 17 → u32 indices.
pub fn unpack_indices_into_avx2_bw17(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 17).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw17: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // Two 16-byte loads per block; the second is at src+8, reaching
    // byte 8+16=24 from the block start. Last block must have at
    // least 24 bytes available.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 17 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw17_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 17 / 8..], remaining, 17, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw17_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    // Same shuffle used for both halves: lane k covers bytes
    // [k*2, k*2+1, k*2+2, k*2+3] within its 128-bit window.
    let shuffle: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let shifts_lo: __m128i = _mm_setr_epi32(0, 1, 2, 3);
    let shifts_hi: __m128i = _mm_setr_epi32(4, 5, 6, 7);
    let mask: __m128i = _mm_set1_epi32(0x1_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v1: __m128i = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);

        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle);
        let hi_b: __m128i = _mm_shuffle_epi8(v1, shuffle);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts_lo);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts_hi);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);

        _mm_storeu_si128(out_ptr.add(blk * 8) as *mut __m128i, lo_masked);
        _mm_storeu_si128(out_ptr.add(blk * 8 + 4) as *mut __m128i, hi_masked);

        src_ptr = src_ptr.add(17);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=17) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw17`.
pub fn unpack_lookup_into_avx2_bw17<T: Copy>(
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
    let required_bytes = (num_values * 17).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw17 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 17 * (full_blocks - 1) + 24 {
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

        if dict_size > (1 << 17) - 1 {
            unpack_avx2_bw17_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw17_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        scalar_bw_n(&packed[processed * 17 / 8..], remaining, 17, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw17_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let shuffle: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let shifts_lo: __m128i = _mm_setr_epi32(0, 1, 2, 3);
    let shifts_hi: __m128i = _mm_setr_epi32(4, 5, 6, 7);
    let mask: __m128i = _mm_set1_epi32(0x1_FFFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v1: __m128i = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle);
        let hi_b: __m128i = _mm_shuffle_epi8(v1, shuffle);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts_lo);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts_hi);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);
        _mm_storeu_si128(staging_ptr as *mut __m128i, lo_masked);
        _mm_storeu_si128(staging_ptr.add(4) as *mut __m128i, hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(17);
    }
    Ok(())
}

// ============================================================
// Π.12e — bw=18
//
// 8 values × 18 bits = 144 bits = 18 bytes per block.
// Two-half layout like bw=15/17: load v0 at +0, v_hi at +7. Two
// different shuffles (lanes 0-3 cover bytes [0,2,4,6]; lanes 4-7
// cover bytes [9,11,13,15] within the original block, which are
// [2,4,6,8] within v_hi). Shifts are symmetric across both halves:
// [0,2,4,6]. Mask = 0x3FFFF.
// Mirror of `bitpack_neon::unpack_indices_into_neon_bw18`.
// ============================================================

/// AVX2 bit-unpack for bit_width = 18 → u32 indices.
pub fn unpack_indices_into_avx2_bw18(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 18).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw18: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // v_hi is loaded at src+7, reaches byte 7+16=23 from block
    // start. Last block must have at least 24 bytes available.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 18 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw18_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 18 / 8..], remaining, 18, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw18_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let shuffle_hi: __m128i = _mm_setr_epi8(2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11);
    let shifts: __m128i = _mm_setr_epi32(0, 2, 4, 6);
    let mask: __m128i = _mm_set1_epi32(0x3_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v_hi: __m128i = _mm_loadu_si128(src_ptr.add(7) as *const __m128i);

        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v_hi, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);

        _mm_storeu_si128(out_ptr.add(blk * 8) as *mut __m128i, lo_masked);
        _mm_storeu_si128(out_ptr.add(blk * 8 + 4) as *mut __m128i, hi_masked);

        src_ptr = src_ptr.add(18);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=18) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw18`.
pub fn unpack_lookup_into_avx2_bw18<T: Copy>(
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
    let required_bytes = (num_values * 18).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw18 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 18 * (full_blocks - 1) + 24 {
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

        if dict_size > (1 << 18) - 1 {
            unpack_avx2_bw18_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw18_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        scalar_bw_n(&packed[processed * 18 / 8..], remaining, 18, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw18_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let shuffle_hi: __m128i = _mm_setr_epi8(2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11);
    let shifts: __m128i = _mm_setr_epi32(0, 2, 4, 6);
    let mask: __m128i = _mm_set1_epi32(0x3_FFFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let v_hi: __m128i = _mm_loadu_si128(src_ptr.add(7) as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v_hi, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);
        _mm_storeu_si128(staging_ptr as *mut __m128i, lo_masked);
        _mm_storeu_si128(staging_ptr.add(4) as *mut __m128i, hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(18);
    }
    Ok(())
}

// ============================================================
// Π.12f — bw=12
//
// 8 values × 12 bits = 96 bits = 12 bytes per block. Unlike
// NEON which uses 16-bit lane shifts (`vshlq_s16`) to keep eight
// lanes in 128 bits, AVX2 has no `_mm_srlv_epi16` until AVX-512
// — so we instead use the same 32-bit-lane shape as bw=14..18:
// two halves of 4 lanes, single shuffle per half, `_mm_srlv_epi32`.
//
// Lane → byte_start / bit_offset map:
//   v[0] → b0  off 0    v[4] → b6  off 0
//   v[1] → b1  off 4    v[5] → b7  off 4
//   v[2] → b3  off 0    v[6] → b9  off 0
//   v[3] → b4  off 4    v[7] → b10 off 4
//
// shuffle_lo = bytes [0,1,3,4]; shuffle_hi = bytes [6,7,9,10].
// shifts (same for both halves) = [0, 4, 0, 4]; mask = 0xFFF.
// Lane 7 reads bytes [10,11,12,13], so each block reads 14 bytes
// from a 12-byte payload — the 16-byte load goes 2 past, with
// the same safety dance as bw=14.
// ============================================================

/// AVX2 bit-unpack for bit_width = 12 → u32 indices.
pub fn unpack_indices_into_avx2_bw12(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 12).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw12: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // Each block reads 16 bytes (one `_mm_loadu_si128`) but only
    // consumes 12. Last block must have at least 16 bytes available.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 12 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw12_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 12 / 8..], remaining, 12, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw12_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 4, 5, 6, 7);
    let shuffle_hi: __m128i = _mm_setr_epi8(6, 7, 8, 9, 7, 8, 9, 10, 9, 10, 11, 12, 10, 11, 12, 13);
    let shifts: __m128i = _mm_setr_epi32(0, 4, 0, 4);
    let mask: __m128i = _mm_set1_epi32(0xFFF);

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

        src_ptr = src_ptr.add(12);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused AVX2 unpack (bw=12) + scalar dict gather. Mirror of
/// `bitpack_neon::unpack_lookup_into_neon_bw12`.
pub fn unpack_lookup_into_avx2_bw12<T: Copy>(
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
    let required_bytes = (num_values * 12).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw12 lookup: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 12 * (full_blocks - 1) + 16 {
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

        if dict_size > (1 << 12) - 1 {
            unpack_avx2_bw12_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw12_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        scalar_bw_n(&packed[processed * 12 / 8..], remaining, 12, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw12_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 4, 5, 6, 7);
    let shuffle_hi: __m128i = _mm_setr_epi8(6, 7, 8, 9, 7, 8, 9, 10, 9, 10, 11, 12, 10, 11, 12, 13);
    let shifts: __m128i = _mm_setr_epi32(0, 4, 0, 4);
    let mask: __m128i = _mm_set1_epi32(0xFFF);

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
        src_ptr = src_ptr.add(12);
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

// ---- bw=8: byte-aligned trivial AVX2 expansion ---------------------
//
// AVX2 mirror of `bitpack_neon::unpack_indices_into_neon_bw8`. One
// 32-value block = 32 source bytes = 4 × 8-u32 SIMD stores via two
// `_mm256_cvtepu8_epi32` widens.

pub fn unpack_indices_into_avx2_bw8(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values;
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw8: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_avx2_bw8_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed..], remaining, 8, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw8_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        // 32 source bytes per block. Each 8-byte chunk widens into
        // a 256-bit register of 8 u32 via _mm256_cvtepu8_epi32 (reads
        // the low 64 bits of a __m128i).
        for chunk in 0..4 {
            let v64: __m128i = _mm_loadl_epi64(src_ptr.add(chunk * 8) as *const __m128i);
            let widened: __m256i = _mm256_cvtepu8_epi32(v64);
            _mm256_storeu_si256(out_ptr.add(blk * 32 + chunk * 8) as *mut __m256i, widened);
        }
        src_ptr = src_ptr.add(32);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=4: nibble-aligned AVX2 expansion ---------------------------
//
// AVX2 mirror of `bitpack_neon::unpack_indices_into_neon_bw4`. 32
// values = 16 source bytes. Two values per byte (lo nibble first per
// parquet LSB-first packing). Strategy: load 16 bytes, mask the low
// nibble in one register and shift-right-4 in another, then interleave
// them per byte position before widening to u32.

pub fn unpack_indices_into_avx2_bw4(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values.div_ceil(2);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw4: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_avx2_bw4_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 2..], remaining, 4, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw4_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);
    let mask_nibble: __m128i = _mm_set1_epi8(0x0F);

    for blk in 0..full_blocks {
        // 16 source bytes → 32 nibbles → 32 u32 values.
        let bytes: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        // Low nibbles = byte & 0x0F.
        let lo_nibbles: __m128i = _mm_and_si128(bytes, mask_nibble);
        // High nibbles = (byte >> 4) & 0x0F. AVX2 has no per-byte shift,
        // so do a 16-bit srli then re-mask.
        let hi_nibbles: __m128i = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask_nibble);
        // Interleave: value[2i] from lo_nibbles[i], value[2i+1] from
        // hi_nibbles[i]. `_mm_unpacklo_epi8 / _mm_unpackhi_epi8`
        // produce exactly that pattern.
        let interleaved_lo: __m128i = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
        let interleaved_hi: __m128i = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles);

        // Each 16-byte interleaved register widens via two
        // _mm256_cvtepu8_epi32 calls (low 64 bits at a time).
        let il_lo: __m256i = _mm256_cvtepu8_epi32(interleaved_lo);
        let il_hi: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_lo, 8));
        let ih_lo: __m256i = _mm256_cvtepu8_epi32(interleaved_hi);
        let ih_hi: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_hi, 8));

        let dst = out_ptr.add(blk * 32);
        _mm256_storeu_si256(dst as *mut __m256i, il_lo);
        _mm256_storeu_si256(dst.add(8) as *mut __m256i, il_hi);
        _mm256_storeu_si256(dst.add(16) as *mut __m256i, ih_lo);
        _mm256_storeu_si256(dst.add(24) as *mut __m256i, ih_hi);
        src_ptr = src_ptr.add(16);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=4 / bw=6 / bw=8 lookup variants (AVX2 mirrors) -------------
//
// Mirror `bitpack_neon::unpack_lookup_into_neon_bw{4,6,8}`. Each
// stages a NEON-equivalent block into a stack buffer and runs the
// scalar dict gather inline. Bounds-safe fast path skips per-element
// checks when dict_size proves every index fits.

/// AVX2 mirror of `bitpack_neon::unpack_lookup_into_neon_bw4`.
pub fn unpack_lookup_into_avx2_bw4<T: Copy>(
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
    let required_bytes = num_values.div_ceil(2);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw4 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 32;
    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut staging = [0u32; 32];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        if dict_size > 15 {
            unpack_avx2_bw4_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw4_into_staging(packed, full_blocks, &mut staging, |idxs| {
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
                written += 32;
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

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed / 2..], remaining, 4, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw4_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();
    let mask_nibble: __m128i = _mm_set1_epi8(0x0F);

    for _ in 0..full_blocks {
        let bytes: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let lo_nibbles: __m128i = _mm_and_si128(bytes, mask_nibble);
        let hi_nibbles: __m128i = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask_nibble);
        let interleaved_lo: __m128i = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
        let interleaved_hi: __m128i = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles);

        let il_lo: __m256i = _mm256_cvtepu8_epi32(interleaved_lo);
        let il_hi: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_lo, 8));
        let ih_lo: __m256i = _mm256_cvtepu8_epi32(interleaved_hi);
        let ih_hi: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_hi, 8));

        _mm256_storeu_si256(staging_ptr as *mut __m256i, il_lo);
        _mm256_storeu_si256(staging_ptr.add(8) as *mut __m256i, il_hi);
        _mm256_storeu_si256(staging_ptr.add(16) as *mut __m256i, ih_lo);
        _mm256_storeu_si256(staging_ptr.add(24) as *mut __m256i, ih_hi);
        sink(staging)?;
        src_ptr = src_ptr.add(16);
    }
    Ok(())
}

/// AVX2 mirror of `bitpack_neon::unpack_lookup_into_neon_bw6`.
pub fn unpack_lookup_into_avx2_bw6<T: Copy>(
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
    let required_bytes = (num_values * 6).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw6 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 10) / 6).min(full_blocks)
    };

    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut staging = [0u32; 8];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        if dict_size > 63 {
            unpack_avx2_bw6_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw6_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        scalar_bw_n(&packed[processed * 6 / 8..], remaining, 6, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw6_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::x86_64::*;
    // Lo lanes (0..3): u32 windows [0..4], [0..4], [1..5], [2..6].
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5);
    // Hi lanes (4..7): u32 windows [3..7], [3..7], [4..8], [5..9].
    let shuffle_hi: __m128i = _mm_setr_epi8(3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8);
    let shifts: __m128i = _mm_setr_epi32(0, 6, 4, 2);
    let mask: __m128i = _mm_set1_epi32(0x3F);

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
        src_ptr = src_ptr.add(6);
    }
    Ok(())
}

/// AVX2 mirror of `bitpack_neon::unpack_lookup_into_neon_bw8`.
pub fn unpack_lookup_into_avx2_bw8<T: Copy>(
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
    let required_bytes = num_values;
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw8 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 32;
    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut staging = [0u32; 32];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        if dict_size > 255 {
            unpack_avx2_bw8_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_avx2_bw8_into_staging(packed, full_blocks, &mut staging, |idxs| {
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
                written += 32;
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

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed..], remaining, 8, &mut tail_idxs);
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
unsafe fn unpack_avx2_bw8_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        // 32 source bytes = two __m128i loads, each widening to a
        // __m256i u32x8 via two _mm256_cvtepu8_epi32 (low 8 / high 8).
        let b0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let b1: __m128i = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let q0: __m256i = _mm256_cvtepu8_epi32(b0);
        let q1: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(b0, 8));
        let q2: __m256i = _mm256_cvtepu8_epi32(b1);
        let q3: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(b1, 8));
        _mm256_storeu_si256(staging_ptr as *mut __m256i, q0);
        _mm256_storeu_si256(staging_ptr.add(8) as *mut __m256i, q1);
        _mm256_storeu_si256(staging_ptr.add(16) as *mut __m256i, q2);
        _mm256_storeu_si256(staging_ptr.add(24) as *mut __m256i, q3);
        sink(staging)?;
        src_ptr = src_ptr.add(32);
    }
    Ok(())
}

// ---- bw=2: 4-streams-per-byte AVX2 expansion -----------------------
//
// AVX2 mirror of `bitpack_neon::unpack_indices_into_neon_bw2`. 32
// values = 8 source bytes. Per byte: 4 values stacked at bit offsets
// 0 / 2 / 4 / 6 (LSB-first). Strategy: load 8 bytes into the low half
// of a __m128i, build 4 streams via shift + mask, then 4-way
// interleave with `_mm_unpacklo_epi8` (byte) + `_mm_unpacklo/hi_epi16`
// (u16-pair) to land 32 output bytes in [s0, s1, s2, s3, ...] order
// before widening to u32x32.

pub fn unpack_indices_into_avx2_bw2(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw2: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_avx2_bw2_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 4..], remaining, 2, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw2_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);
    let mask2: __m128i = _mm_set1_epi8(0x03);

    for blk in 0..full_blocks {
        // 8 source bytes (32 values) loaded into the low 64 bits.
        let src: __m128i = _mm_loadl_epi64(src_ptr as *const __m128i);
        // Four parallel streams. AVX2 has no per-byte shift; do u16
        // shifts then re-mask.
        let s0 = _mm_and_si128(src, mask2);
        let s1 = _mm_and_si128(_mm_srli_epi16(src, 2), mask2);
        let s2 = _mm_and_si128(_mm_srli_epi16(src, 4), mask2);
        let s3 = _mm_and_si128(_mm_srli_epi16(src, 6), mask2);

        // Pair (s0, s1) and (s2, s3) at byte granularity:
        // unpacklo_epi8 zips low-8 bytes of each → 16 bytes.
        let p01: __m128i = _mm_unpacklo_epi8(s0, s1);
        let p23: __m128i = _mm_unpacklo_epi8(s2, s3);

        // 4-way interleave via u16-pair zip: each u16 lane of p01
        // already holds (s0[i] | s1[i]<<8); zipping against p23 yields
        // [s0[i], s1[i], s2[i], s3[i], ...] at byte granularity.
        let block_lo: __m128i = _mm_unpacklo_epi16(p01, p23); // values 0..16
        let block_hi: __m128i = _mm_unpackhi_epi16(p01, p23); // values 16..32

        // Widen each 16-byte block to two u32x8 vectors.
        let q0: __m256i = _mm256_cvtepu8_epi32(block_lo);
        let q1: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(block_lo, 8));
        let q2: __m256i = _mm256_cvtepu8_epi32(block_hi);
        let q3: __m256i = _mm256_cvtepu8_epi32(_mm_srli_si128(block_hi, 8));

        let dst = out_ptr.add(blk * 32);
        _mm256_storeu_si256(dst as *mut __m256i, q0);
        _mm256_storeu_si256(dst.add(8) as *mut __m256i, q1);
        _mm256_storeu_si256(dst.add(16) as *mut __m256i, q2);
        _mm256_storeu_si256(dst.add(24) as *mut __m256i, q3);
        src_ptr = src_ptr.add(8);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=3: AVX2, PSHUFB + variable shift ---------------------------
//
// AVX2 mirror of `bitpack_neon::unpack_indices_into_neon_bw3`. 8
// values = 3 source bytes; values span at most 2 bytes (3 + 7 = 10
// bits). Per-lane (byte, bit_off): (0,0), (0,3), (0,6), (1,1),
// (1,4), (1,7), (2,2), (2,5). Two `_mm_shuffle_epi8` gathers
// build 4-byte windows per lane; `_mm_srlv_epi32` does the per-lane
// right shift; mask to 3 bits. Single 16-byte load covers both
// shuffles (max byte index is 5).

pub fn unpack_indices_into_avx2_bw3(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 3).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw3: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 13) / 3).min(full_blocks)
    };

    unsafe {
        unpack_avx2_bw3_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 3 / 8..], remaining, 3, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw3_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    // Lo lanes (0..3): byte windows [0..4], [0..4], [0..4], [1..5].
    let shuffle_lo: __m128i = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4);
    // Hi lanes (4..7): byte windows [1..5], [1..5], [2..6], [2..6].
    let shuffle_hi: __m128i = _mm_setr_epi8(1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 5);
    let shifts_lo: __m128i = _mm_setr_epi32(0, 3, 6, 1);
    let shifts_hi: __m128i = _mm_setr_epi32(4, 7, 2, 5);
    let mask: __m128i = _mm_set1_epi32(0x07);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0: __m128i = _mm_loadu_si128(src_ptr as *const __m128i);
        let lo_b: __m128i = _mm_shuffle_epi8(v0, shuffle_lo);
        let hi_b: __m128i = _mm_shuffle_epi8(v0, shuffle_hi);
        let lo_shifted: __m128i = _mm_srlv_epi32(lo_b, shifts_lo);
        let hi_shifted: __m128i = _mm_srlv_epi32(hi_b, shifts_hi);
        let lo_masked: __m128i = _mm_and_si128(lo_shifted, mask);
        let hi_masked: __m128i = _mm_and_si128(hi_shifted, mask);

        _mm_storeu_si128(out_ptr.add(blk * 8) as *mut __m128i, lo_masked);
        _mm_storeu_si128(out_ptr.add(blk * 8 + 4) as *mut __m128i, hi_masked);

        src_ptr = src_ptr.add(3);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=1: bit-test AVX2 unpack ------------------------------------
//
// 8 values per byte. 32 values = 4 source bytes. Strategy: broadcast
// the 4 input bytes to 32 lanes, AND with a per-lane bit-mask, and
// compare-not-equal-zero to produce 0/1. The bit-mask layout per byte
// is [1, 2, 4, 8, 16, 32, 64, 128] for value indices [0..8].

pub fn unpack_indices_into_avx2_bw1(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values.div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw1: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_avx2_bw1_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 8..], remaining, 1, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw1_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    // Per-byte bit-mask for the 8 values packed into one source byte.
    // Parquet LSB-first: value[0] is bit 0 = mask 1.
    let bit_masks: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let ones: __m256i = _mm256_set1_epi32(1);

    for blk in 0..full_blocks {
        // Each block = 4 source bytes (32 values). Handle one byte at
        // a time → 8 u32 outputs each.
        for b in 0..4 {
            let byte_val = *src_ptr.add(b) as i32;
            // Broadcast the byte into all 8 lanes, then AND with the
            // per-lane bit-mask and compare to bit_masks (== nonzero).
            let v: __m256i = _mm256_set1_epi32(byte_val);
            let masked: __m256i = _mm256_and_si256(v, bit_masks);
            // masked == bit_masks iff the bit is set. Compare-eq
            // produces -1 / 0; AND with 1 to land 1 / 0.
            let cmp: __m256i = _mm256_cmpeq_epi32(masked, bit_masks);
            let result: __m256i = _mm256_and_si256(cmp, ones);
            _mm256_storeu_si256(out_ptr.add(blk * 32 + b * 8) as *mut __m256i, result);
        }
        src_ptr = src_ptr.add(4);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=20: AVX2, mirrors bw=17/18 shape ---------------------------
//
// 8 values per block = 20 source bytes. Each value lives in 20 bits;
// per-lane start bytes are [0, 2, 5, 7, 10, 12, 15, 17] with bit
// offsets [0, 4, 0, 4, 0, 4, 0, 4]. The block reads 26 bytes total
// (two 16-byte windows with the second offset by 10).

pub fn unpack_indices_into_avx2_bw20(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 20).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw20: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Each block reads up to byte 26; safety check for the final iter.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 20 * (full_blocks - 1) + 26 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw20_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 20 / 8..], remaining, 20, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw20_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    // Per-lane right-shift amounts (bit_offset within source byte).
    let shifts: __m256i = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
    let mask: __m256i = _mm256_set1_epi32(0x0F_FFFF);

    for blk in 0..full_blocks {
        // For each of the 8 lanes, gather 4 source bytes starting at
        // [0, 2, 5, 7, 10, 12, 15, 17] in the block. Build the 8 u32
        // values by gathering via aligned offsets.
        //
        // No AVX2 byte-level gather, so we do this scalar in a small
        // stack buffer and then AVX2-load it for the shift+mask.
        let mut staging = [0u32; 8];
        let offsets = [0usize, 2, 5, 7, 10, 12, 15, 17];
        for (lane, &off) in offsets.iter().enumerate() {
            // Read 4 bytes little-endian into a u32.
            let b0 = *src_ptr.add(off) as u32;
            let b1 = *src_ptr.add(off + 1) as u32;
            let b2 = *src_ptr.add(off + 2) as u32;
            let b3 = *src_ptr.add(off + 3) as u32;
            staging[lane] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        let v: __m256i = _mm256_loadu_si256(staging.as_ptr() as *const __m256i);
        let shifted: __m256i = _mm256_srlv_epi32(v, shifts);
        let masked: __m256i = _mm256_and_si256(shifted, mask);
        _mm256_storeu_si256(out_ptr.add(blk * 8) as *mut __m256i, masked);
        src_ptr = src_ptr.add(20);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=21: AVX2, lane-specific shifts ----------------------------
//
// 8 values per block = 21 source bytes. Per-lane start bytes [0, 2,
// 5, 7, 10, 13, 15, 18] with bit offsets [0, 5, 2, 7, 4, 1, 6, 3] —
// every lane has a different shift.

pub fn unpack_indices_into_avx2_bw21(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 21).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw21: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Reads up to byte 22; safety check for the last block.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 21 * (full_blocks - 1) + 22 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw21_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 21 / 8..], remaining, 21, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw21_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    let shifts: __m256i = _mm256_setr_epi32(0, 5, 2, 7, 4, 1, 6, 3);
    let mask: __m256i = _mm256_set1_epi32(0x1F_FFFF);

    for blk in 0..full_blocks {
        let mut staging = [0u32; 8];
        // Per-lane start bytes for bw=21.
        let offsets = [0usize, 2, 5, 7, 10, 13, 15, 18];
        for (lane, &off) in offsets.iter().enumerate() {
            let b0 = *src_ptr.add(off) as u32;
            let b1 = *src_ptr.add(off + 1) as u32;
            let b2 = *src_ptr.add(off + 2) as u32;
            let b3 = *src_ptr.add(off + 3) as u32;
            staging[lane] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        let v: __m256i = _mm256_loadu_si256(staging.as_ptr() as *const __m256i);
        let shifted: __m256i = _mm256_srlv_epi32(v, shifts);
        let masked: __m256i = _mm256_and_si256(shifted, mask);
        _mm256_storeu_si256(out_ptr.add(blk * 8) as *mut __m256i, masked);
        src_ptr = src_ptr.add(21);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=5: AVX2, lane-specific shifts ------------------------------
//
// 8 values per block = 5 source bytes. Per-lane start bytes
// [0, 0, 1, 1, 2, 3, 3, 4] with bit offsets [0, 5, 2, 7, 4, 1, 6, 3].
// Each lane reads a 4-byte little-endian window (always fits within
// 8 bytes, so we read the block + safety guard at the tail).

pub fn unpack_indices_into_avx2_bw5(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 5).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "avx2 bw5: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Each block reads bytes 0..=7 (lane 7 + 4 bytes = byte 8).
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 5 * (full_blocks - 1) + 8 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_avx2_bw5_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 5 / 8..], remaining, 5, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_bw5_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    let shifts: __m256i = _mm256_setr_epi32(0, 5, 2, 7, 4, 1, 6, 3);
    let mask: __m256i = _mm256_set1_epi32(0x1F);
    let offsets = [0usize, 0, 1, 1, 2, 3, 3, 4];

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let mut staging = [0u32; 8];
        for (lane, &off) in offsets.iter().enumerate() {
            let b0 = *src_ptr.add(off) as u32;
            let b1 = *src_ptr.add(off + 1) as u32;
            let b2 = *src_ptr.add(off + 2) as u32;
            let b3 = *src_ptr.add(off + 3) as u32;
            staging[lane] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        let v: __m256i = _mm256_loadu_si256(staging.as_ptr() as *const __m256i);
        let shifted: __m256i = _mm256_srlv_epi32(v, shifts);
        let masked: __m256i = _mm256_and_si256(shifted, mask);
        _mm256_storeu_si256(out_ptr.add(blk * 8) as *mut __m256i, masked);
        src_ptr = src_ptr.add(5);
    }
    out.set_len(out_start_len + full_blocks * 8);
}
