//! NEON-specialized bit-unpackers for hot widths.
//!
//! Only built on `target_arch = "aarch64"`. NEON is part of the
//! aarch64 baseline (no runtime feature detection needed on Apple
//! Silicon or modern Linux ARM).
//!
//! Currently provides:
//!   - `unpack_indices_into_neon_bw12` — bit_width = 12, the width
//!     used by l_shipdate / l_commitdate / l_receiptdate in TPC-H
//!     lineitem (3 columns × 6M values).
//!
//! Strategy for bw=12:
//! For each block of 8 output values we consume exactly 12 input
//! bytes (8 × 12 bits = 96 bits). Per-lane the source u16 offset
//! and shift are `lane=0..7`, `byte=[0,1,3,4,6,7,9,10]`,
//! `shift=[0,4,0,4,0,4,0,4]`.
//!
//! 1. Load 16 bytes via `vld1q_u8` (we only need the first 13).
//! 2. Shuffle into a u8x16 representing u16x8 little-endian:
//!    [b0,b1, b1,b2, b3,b4, b4,b5, b6,b7, b7,b8, b9,b10, b10,b11].
//! 3. Cast as u16x8, variable right-shift via `vshlq_u16` with
//!    negative signed shifts (NEON's per-lane variable shift).
//! 4. AND with `0x0FFF` to mask off the upper 4 bits.
//! 5. Widen u16x8 → 2 × u32x4 via `vmovl_u16(vget_{low,high}_u16)`.
//! 6. Store both as `vst1q_u32` to the output.
//!
//! ~7 NEON ops produce 8 u32 outputs. Measured on M-series at
//! 0.04 ns/value vs scalar 0.44 ns/value — ~10× speedup.
//!
//! Tail handling: any values not in a multiple of 8 fall through
//! to a small scalar loop with the streaming bit buffer.

#![cfg(target_arch = "aarch64")]

use crate::error::{CodecError, Result};

/// Bit-unpack `num_values` 12-bit indices from `packed` into `out`.
/// Asserts on entry that the caller has reserved enough capacity.
pub fn unpack_indices_into_neon_bw12(
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
            "neon bw12: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // The kernel reads 16 bytes per block but only consumes 12. We
    // need to make sure the final 16-byte load doesn't overrun
    // `packed`: do that by checking we have ≥ 12*(full_blocks-1) + 16
    // bytes available for the final iteration. If the buffer is too
    // small, fall back to scalar for the last block.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 12 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw12_unchecked(packed, safe_full_blocks, out);
    }

    // Any remaining values (scalar block + tail) — total ≤ 15 — use
    // the streaming bit buffer.
    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw12(&packed[processed * 12 / 8..], remaining, out);
    }

    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw12_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // Shuffle table: assembles a u8x16 representing u16x8 (little-endian)
    // sourced from byte offsets [0,1,3,4,6,7,9,10] of the 13-byte
    // window. Lane i reads bytes [src_byte[i], src_byte[i]+1].
    let shuffle: uint8x16_t =
        vld1q_u8([0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr());
    // Per-lane right-shift amounts. NEON's `vshlq_u16` treats negative
    // shifts as right shifts.
    let shifts: int16x8_t = vld1q_s16([0i16, -4, 0, -4, 0, -4, 0, -4].as_ptr());
    let mask: uint16x8_t = vdupq_n_u16(0x0FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let bytes16: uint8x16_t = vld1q_u8(src_ptr);
        let shuffled: uint8x16_t = vqtbl1q_u8(bytes16, shuffle);
        let as_u16: uint16x8_t = vreinterpretq_u16_u8(shuffled);
        // Variable per-lane shift via signed vshl on the unsigned vec.
        let shifted: uint16x8_t =
            vreinterpretq_u16_s16(vshlq_s16(vreinterpretq_s16_u16(as_u16), shifts));
        let masked: uint16x8_t = vandq_u16(shifted, mask);
        let lo: uint32x4_t = vmovl_u16(vget_low_u16(masked));
        let hi: uint32x4_t = vmovl_u16(vget_high_u16(masked));

        vst1q_u32(out_ptr.add(blk * 8), lo);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi);

        src_ptr = src_ptr.add(12);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=12) + scalar dict gather. Mirror of
/// `bitpack::unpack_lookup_into` for the bw=12 specialization.
/// Each block unpacks 8 indices via NEON into a stack buffer, then
/// gathers `dict[idx]` per lane via raw pointer writes — no
/// per-element bounds or capacity checks on the hot path.
pub fn unpack_lookup_into_neon_bw12<T: Copy>(
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
            "neon bw12 lookup: packed has {} bytes, need {}",
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
    // Pre-validate: a single up-front pass through the staging
    // buffer would require allocating one. Instead, we let the
    // hot path write speculatively (raw pointer writes are safe
    // because we reserved capacity), and validate the indices in
    // the same block — bailing out with the post-hoc error if any
    // index is out of range. This keeps the hot path branchless.

    let mut staging = [0u32; 8];
    let mut bad_idx: Option<u32> = None;
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;

        if dict_size > 4095 {
            // Bounds-safe fast path: every 12-bit index fits in dict.
            // No per-element bounds check.
            unpack_neon_bw12_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            // Bounds-checked path. Branch is predictable (taken or
            // not based on dict_size, which is fixed across the page).
            unpack_neon_bw12_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
        // Commit the writes done above.
        out.set_len(out_start_len + written);
    }
    if let Some(i) = bad_idx {
        return Err(CodecError::DictIndexOutOfRange {
            index: i,
            dict_size,
        });
    }

    // Tail: scalar fallback for the < 8 remaining values.
    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw12(&packed[processed * 12 / 8..], remaining, &mut tail_idxs);
        for &i in &tail_idxs {
            let i_usize = i as usize;
            if i_usize >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            // SAFETY: out has reserved num_values capacity, and we've
            // written exactly safe_full_blocks * 8 < num_values so far.
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(i_usize);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

/// Helper: drive the NEON kernel block-by-block, exposing each 8-lane
/// index batch via a callback. Lets the lookup variant interleave the
/// dict gather without re-implementing the unpack inner loop.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw12_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle: uint8x16_t =
        vld1q_u8([0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr());
    let shifts: int16x8_t = vld1q_s16([0i16, -4, 0, -4, 0, -4, 0, -4].as_ptr());
    let mask: uint16x8_t = vdupq_n_u16(0x0FFF);
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let bytes16 = vld1q_u8(src_ptr);
        let shuffled = vqtbl1q_u8(bytes16, shuffle);
        let as_u16 = vreinterpretq_u16_u8(shuffled);
        let shifted = vreinterpretq_u16_s16(vshlq_s16(vreinterpretq_s16_u16(as_u16), shifts));
        let masked = vandq_u16(shifted, mask);
        let lo = vmovl_u16(vget_low_u16(masked));
        let hi = vmovl_u16(vget_high_u16(masked));
        vst1q_u32(staging_ptr, lo);
        vst1q_u32(staging_ptr.add(4), hi);
        sink(staging)?;
        src_ptr = src_ptr.add(12);
    }
    Ok(())
}

/// Predicate-fused decode: unpack bw=12 indices and write the
/// per-row predicate result into a packed bitmap, without ever
/// materializing a Vec<u32> of indices.
///
/// `dict_mask[i]` must be 0 (miss) or 1 (match), and `dict_mask`
/// must be at least 4096 bytes long (zero-padded for any unused
/// dict slots). The caller fills `dict_mask` once per page by
/// applying its predicate to each dict value — typically negligible
/// (~2.5K ops for l_shipdate's ~2525-entry dict).
///
/// Output: `out` receives `num_values.div_ceil(8)` bytes. Bit `i`
/// of byte `k` represents row `8k + i`.
///
/// Per 8 input rows the kernel runs:
///   - 1 NEON load + shuffle + variable shift + mask (≈ 7 ops)
///   - 8 scalar byte loads from dict_mask (L1 cached)
///   - 8 shifted ORs into one u8
///   - 1 byte store
///
/// vs the previous path which writes 8 × i32 (32 bytes) plus a
/// downstream Vec<bool> scan, this writes 1 byte and stops. 32× less
/// output write traffic. Targets the Q14 lever in TPC-H lineitem
/// (l_shipdate predicate).
pub fn decode_predicate_bitmap_neon_bw12(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < 4096 {
        return Err(CodecError::Decompress(format!(
            "neon bw12 fused: dict_mask must be ≥ 4096 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 12).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw12 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    let tail = num_values % 8;

    // Safety boundary on NEON load: kernel reads 16 bytes per block
    // but uses 12. The final iteration may overrun if packed.len() is
    // exactly 12 * full_blocks. Drop the last full block to scalar in
    // that case.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 12 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        fused_predicate_neon_bw12(
            packed,
            safe_full_blocks,
            dict_mask,
            &mut out[out_start..out_start + safe_full_blocks],
        );
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        // Scalar fallback for the trailing blocks + tail.
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw12(&packed[processed * 12 / 8..], remaining, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            // SAFETY: idx is 12-bit ≤ 4095, dict_mask is ≥ 4096.
            let bit = unsafe { *dict_mask.get_unchecked(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    let _ = tail; // consumed via `remaining`
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn fused_predicate_neon_bw12(
    packed: &[u8],
    full_blocks: usize,
    dict_mask: &[u8],
    bitmap_out: &mut [u8],
) {
    use std::arch::aarch64::*;
    debug_assert!(dict_mask.len() >= 4096);
    debug_assert_eq!(bitmap_out.len(), full_blocks);

    let shuffle: uint8x16_t =
        vld1q_u8([0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr());
    let shifts: int16x8_t = vld1q_s16([0i16, -4, 0, -4, 0, -4, 0, -4].as_ptr());
    let idx_mask: uint16x8_t = vdupq_n_u16(0x0FFF);

    let mut src_ptr = packed.as_ptr();
    let mask_ptr = dict_mask.as_ptr();

    for blk in 0..full_blocks {
        let bytes16 = vld1q_u8(src_ptr);
        let shuffled = vqtbl1q_u8(bytes16, shuffle);
        let as_u16 = vreinterpretq_u16_u8(shuffled);
        let shifted = vreinterpretq_u16_s16(vshlq_s16(vreinterpretq_s16_u16(as_u16), shifts));
        let masked = vandq_u16(shifted, idx_mask);

        // 8 lane gathers + bit-pack. LLVM keeps these independent
        // (no false dependency chain) so they pipeline through the
        // load units.
        let i0 = vgetq_lane_u16::<0>(masked) as usize;
        let i1 = vgetq_lane_u16::<1>(masked) as usize;
        let i2 = vgetq_lane_u16::<2>(masked) as usize;
        let i3 = vgetq_lane_u16::<3>(masked) as usize;
        let i4 = vgetq_lane_u16::<4>(masked) as usize;
        let i5 = vgetq_lane_u16::<5>(masked) as usize;
        let i6 = vgetq_lane_u16::<6>(masked) as usize;
        let i7 = vgetq_lane_u16::<7>(masked) as usize;
        // SAFETY: indices ≤ 4095, dict_mask ≥ 4096 (checked by caller).
        let b0 = *mask_ptr.add(i0);
        let b1 = *mask_ptr.add(i1);
        let b2 = *mask_ptr.add(i2);
        let b3 = *mask_ptr.add(i3);
        let b4 = *mask_ptr.add(i4);
        let b5 = *mask_ptr.add(i5);
        let b6 = *mask_ptr.add(i6);
        let b7 = *mask_ptr.add(i7);
        let byte =
            b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7);
        *bitmap_out.get_unchecked_mut(blk) = byte;

        src_ptr = src_ptr.add(12);
    }
}

/// Pack 8 dict-mask lookups into one bitmap byte. Used by the
/// width-N fused predicate kernels below. Each idx must be a valid
/// offset into `mask_ptr` (caller verifies via dict_mask length).
///
/// LLVM keeps the 8 loads independent — they pipeline through the
/// load units. The shifted-OR chain folds in fixed cycles.
#[inline(always)]
unsafe fn pack_predicate_byte(idxs: &[u32; 8], mask_ptr: *const u8) -> u8 {
    let b0 = *mask_ptr.add(idxs[0] as usize);
    let b1 = *mask_ptr.add(idxs[1] as usize);
    let b2 = *mask_ptr.add(idxs[2] as usize);
    let b3 = *mask_ptr.add(idxs[3] as usize);
    let b4 = *mask_ptr.add(idxs[4] as usize);
    let b5 = *mask_ptr.add(idxs[5] as usize);
    let b6 = *mask_ptr.add(idxs[6] as usize);
    let b7 = *mask_ptr.add(idxs[7] as usize);
    b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)
}

/// Predicate-fused decode for bw=14: indices ∈ [0, 16384), `dict_mask`
/// must be ≥ 16384 bytes. Mirror of `decode_predicate_bitmap_neon_bw12`
/// — leverages the existing `unpack_neon_bw14_into_staging` helper.
pub fn decode_predicate_bitmap_neon_bw14(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < (1 << 14) {
        return Err(CodecError::Decompress(format!(
            "neon bw14 fused: dict_mask must be ≥ 16384 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 14).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw14 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 14 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    let mask_ptr = dict_mask.as_ptr();
    let mut staging = [0u32; 8];
    unsafe {
        let bitmap_ptr = out.as_mut_ptr().add(out_start);
        let mut blk_idx = 0usize;
        unpack_neon_bw14_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
            *bitmap_ptr.add(blk_idx) = pack_predicate_byte(idxs, mask_ptr);
            blk_idx += 1;
            Ok(())
        })?;
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 14 / 8..], remaining, 14, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            let bit = unsafe { *mask_ptr.add(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    Ok(())
}

/// Predicate-fused decode for bw=15: indices ∈ [0, 32768), `dict_mask`
/// must be ≥ 32768 bytes.
pub fn decode_predicate_bitmap_neon_bw15(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < (1 << 15) {
        return Err(CodecError::Decompress(format!(
            "neon bw15 fused: dict_mask must be ≥ 32768 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 15).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw15 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 15 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    let mask_ptr = dict_mask.as_ptr();
    let mut staging = [0u32; 8];
    unsafe {
        let bitmap_ptr = out.as_mut_ptr().add(out_start);
        let mut blk_idx = 0usize;
        unpack_neon_bw15_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
            *bitmap_ptr.add(blk_idx) = pack_predicate_byte(idxs, mask_ptr);
            blk_idx += 1;
            Ok(())
        })?;
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 15 / 8..], remaining, 15, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            let bit = unsafe { *mask_ptr.add(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    Ok(())
}

/// Predicate-fused decode for bw=16: indices ∈ [0, 65536), `dict_mask`
/// must be ≥ 65536 bytes. The byte-aligned, simplest kernel.
pub fn decode_predicate_bitmap_neon_bw16(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < (1 << 16) {
        return Err(CodecError::Decompress(format!(
            "neon bw16 fused: dict_mask must be ≥ 65536 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 16).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw16 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    // bw=16 staging helper does one 16B load per block; final iter
    // reads bytes 0..15, so packed.len() ≥ 16*full_blocks suffices.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 16 * full_blocks {
        full_blocks
    } else {
        full_blocks - 1
    };

    let mask_ptr = dict_mask.as_ptr();
    let mut staging = [0u32; 8];
    unsafe {
        let bitmap_ptr = out.as_mut_ptr().add(out_start);
        let mut blk_idx = 0usize;
        unpack_neon_bw16_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
            *bitmap_ptr.add(blk_idx) = pack_predicate_byte(idxs, mask_ptr);
            blk_idx += 1;
            Ok(())
        })?;
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 16 / 8..], remaining, 16, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            let bit = unsafe { *mask_ptr.add(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    Ok(())
}

/// Predicate-fused decode for bw=17: indices ∈ [0, 131072), `dict_mask`
/// must be ≥ 131072 bytes.
pub fn decode_predicate_bitmap_neon_bw17(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < (1 << 17) {
        return Err(CodecError::Decompress(format!(
            "neon bw17 fused: dict_mask must be ≥ 131072 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 17).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw17 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 17 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    let mask_ptr = dict_mask.as_ptr();
    let mut staging = [0u32; 8];
    unsafe {
        let bitmap_ptr = out.as_mut_ptr().add(out_start);
        let mut blk_idx = 0usize;
        unpack_neon_bw17_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
            *bitmap_ptr.add(blk_idx) = pack_predicate_byte(idxs, mask_ptr);
            blk_idx += 1;
            Ok(())
        })?;
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 17 / 8..], remaining, 17, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            let bit = unsafe { *mask_ptr.add(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    Ok(())
}

/// Predicate-fused decode for bw=18: indices ∈ [0, 262144), `dict_mask`
/// must be ≥ 262144 bytes.
pub fn decode_predicate_bitmap_neon_bw18(
    packed: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if dict_mask.len() < (1 << 18) {
        return Err(CodecError::Decompress(format!(
            "neon bw18 fused: dict_mask must be ≥ 262144 entries (got {})",
            dict_mask.len()
        )));
    }
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 18).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw18 fused: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }

    let bitmap_bytes = num_values.div_ceil(8);
    out.reserve(bitmap_bytes);
    let out_start = out.len();
    out.resize(out_start + bitmap_bytes, 0);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 18 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    let mask_ptr = dict_mask.as_ptr();
    let mut staging = [0u32; 8];
    unsafe {
        let bitmap_ptr = out.as_mut_ptr().add(out_start);
        let mut blk_idx = 0usize;
        unpack_neon_bw18_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
            *bitmap_ptr.add(blk_idx) = pack_predicate_byte(idxs, mask_ptr);
            blk_idx += 1;
            Ok(())
        })?;
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 18 / 8..], remaining, 18, &mut idxs);
        for (i, idx) in idxs.into_iter().enumerate() {
            let bit = unsafe { *mask_ptr.add(idx as usize) };
            let row = processed + i;
            out[out_start + row / 8] |= bit << (row % 8);
        }
    }
    Ok(())
}

/// NEON unpacker for bit_width = 17 — the dominant width for
/// l_extendedprice (43%), l_partkey (67%), and l_orderkey (51%)
/// dictionary-encoded data pages at SF=1 TPC-H. Per 8-row block:
/// 17 bytes input, 32 bytes output.
///
/// Strategy:
///   Per-lane the source u32 offset and shift for 8 lanes of 17-bit
///   values are byte_offset=[0,2,4,6,8,10,12,14], shift=[0,1,2,3,4,5,6,7].
///   Each 17-bit value spans up to 3 bytes (17 + 7 bit-shift = 24 bits).
///
///   1. Two u8x16 loads (overlapping by 8): one from base, one from
///      base+8. Together they cover bytes 0..24.
///   2. Group 1 (lanes 0-3, byte_offset 0/2/4/6) shuffled from the
///      first vector via TBL with indices [0,1,2,3, 2,3,4,5, 4,5,6,7,
///      6,7,8,9]. Result is u32x4 of "value bytes starting at 0/2/4/6".
///   3. Group 2 (lanes 4-7, byte_offset 8/10/12/14) shuffled from the
///      second vector with the SAME indices — by construction the
///      pattern aligns.
///   4. Variable per-lane right shift via `vshlq_s32` with negative
///      shifts: group 1 = [0,-1,-2,-3], group 2 = [-4,-5,-6,-7].
///   5. AND with 0x1FFFF to mask off the 15 upper bits of each u32.
///   6. Store 2 × u32x4 = 8 u32 outputs.
///
///   ~10 NEON ops produce 8 u32 outputs. Expected speedup over the
///   scalar bw=17 path (~0.54 ns/val): ~10×, matching bw=12 NEON's
///   memory-bandwidth ceiling.
pub fn unpack_indices_into_neon_bw17(
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
            "neon bw17: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // Each block reads 24 bytes (16 + the second 16-byte load shifted
    // by 8 = byte 23 inclusive) but only consumes 17. The final
    // iteration may overrun if packed.len() < 17*(full_blocks-1) + 24.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 17 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw17_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 17 / 8..], remaining, 17, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw17_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    let shuffle: uint8x16_t = vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -1, -2, -3].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -5, -6, -7].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(8));
        let lo_b = vqtbl1q_u8(v0, shuffle);
        let hi_b = vqtbl1q_u8(v1, shuffle);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);

        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);

        src_ptr = src_ptr.add(17);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=17) + scalar dict gather. Mirror of the
/// bw=12/bw=14 lookup specializations for the bw=17 hot path
/// (l_orderkey 51%, l_partkey 67%, l_extendedprice 43% at SF=1).
pub fn unpack_lookup_into_neon_bw17<T: Copy>(
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
            "neon bw17 lookup: packed has {} bytes, need {}",
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
            unpack_neon_bw17_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw17_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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

/// bw=17 staging helper, mirror of `unpack_neon_bw12_into_staging`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw17_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle: uint8x16_t = vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -1, -2, -3].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -5, -6, -7].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1_FFFF);
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(8));
        let lo_b = vqtbl1q_u8(v0, shuffle);
        let hi_b = vqtbl1q_u8(v1, shuffle);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(17);
    }
    Ok(())
}

/// NEON unpacker for bit_width = 14 — the dominant width for
/// l_suppkey (100% of dict pages, ~6.25M values per TPC-H lineitem
/// SF=1). Per 8-row block: 14 bytes input, 32 bytes output (8 × u32).
///
/// Strategy (mirror of bw=17 but with 4-bit symmetry between halves):
///   Per-lane the source u32 byte_offset and shift for 8 lanes of
///   14-bit values are
///     byte_offset = [0, 1, 3, 5, 7, 8, 10, 12]
///     shift       = [0, 6, 4, 2, 0, 6,  4,  2]
///   Each 14-bit value spans up to 3 bytes (14 + 6 bit-shift = 20 bits),
///   so reading a u32 at the start byte covers it with room to spare.
///
///   1. One u8x16 load: bytes 0..16 of the 14-byte block (we read 16,
///      consume 14, the final iteration must not overrun).
///   2. Two TBL shuffles against the same source vector: low 4 lanes
///      use indices [0,1,2,3, 1,2,3,4, 3,4,5,6, 5,6,7,8] (lanes 0-3),
///      high 4 lanes use [7,8,9,10, 8,9,10,11, 10,11,12,13, 12,13,14,15]
///      (lanes 4-7). Each result is u32x4.
///   3. Variable per-lane right shift via vshlq_s32 with negative
///      shifts. Both halves use [0, -6, -4, -2] — the lane-shift
///      pattern is symmetric every 4 lanes.
///   4. AND with 0x3FFF to mask off the upper 18 bits of each u32.
///   5. Store 2 × u32x4 = 8 u32 outputs.
///
///   ~9 NEON ops produce 8 u32 outputs. Scalar bw=14 measured at
///   0.82 ns/val on M-series; NEON should land near memory bandwidth
///   like bw=12/17 (~0.05 ns/val), an order of magnitude faster.
pub fn unpack_indices_into_neon_bw14(
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
            "neon bw14: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;

    // Each block reads 16 bytes but only consumes 14. The final
    // iteration may overrun if packed.len() < 14*(full_blocks-1) + 16.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 14 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw14_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 14 / 8..], remaining, 14, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw14_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // Lanes 0-3: u32 windows starting at bytes 0/1/3/5.
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr());
    // Lanes 4-7: u32 windows starting at bytes 7/8/10/12.
    let shuffle_hi: uint8x16_t =
        vld1q_u8([7u8, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15].as_ptr());
    // Per-lane right shifts (negative for vshlq_s32). Symmetric
    // between halves: [0, 6, 4, 2] bit shifts inside the u32 window.
    let shifts: int32x4_t = vld1q_s32([0i32, -6, -4, -2].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);

        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);

        src_ptr = src_ptr.add(14);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=14) + scalar dict gather. Mirror of
/// `unpack_lookup_into_neon_bw12` for the bw=14 specialization
/// (l_suppkey is ~6.25M values per TPC-H lineitem SF=1). Uses
/// raw pointer writes — no per-element bounds or capacity checks
/// on the hot path.
pub fn unpack_lookup_into_neon_bw14<T: Copy>(
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
            "neon bw14 lookup: packed has {} bytes, need {}",
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
            unpack_neon_bw14_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            // Bounds-checked path; branch is predictable across page.
            unpack_neon_bw14_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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

    // Tail: scalar fallback for the < 8 remaining values.
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

/// bw=14 staging helper, mirror of `unpack_neon_bw12_into_staging`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw14_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([7u8, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15].as_ptr());
    let shifts: int32x4_t = vld1q_s32([0i32, -6, -4, -2].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3FFF);
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(14);
    }
    Ok(())
}

/// NEON unpacker for bit_width = 16 — byte-aligned, the simplest
/// kernel. 16 bytes input, 8 u16 values, widened to 8 u32 outputs.
/// No shuffle, no shift, no mask — just a load, a cast, and two
/// widening moves. Should be the fastest NEON kernel.
pub fn unpack_indices_into_neon_bw16(
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
            "neon bw16: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;

    // bw=16 is exactly 16 bytes per block; no overrun risk.
    unsafe {
        unpack_neon_bw16_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 2..], remaining, 16, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw16_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v: uint8x16_t = vld1q_u8(src_ptr);
        let as_u16: uint16x8_t = vreinterpretq_u16_u8(v);
        let lo: uint32x4_t = vmovl_u16(vget_low_u16(as_u16));
        let hi: uint32x4_t = vmovl_u16(vget_high_u16(as_u16));
        vst1q_u32(out_ptr.add(blk * 8), lo);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi);
        src_ptr = src_ptr.add(16);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=16) + scalar dict gather. Mirror of the
/// bw=12/14/17 lookup specializations.
pub fn unpack_lookup_into_neon_bw16<T: Copy>(
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
            "neon bw16 lookup: packed has {} bytes, need {}",
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

        if dict_size > (1 << 16) - 1 {
            unpack_neon_bw16_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw16_into_staging(packed, full_blocks, &mut staging, |idxs| {
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

    let processed = full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        let mut tail_idxs: Vec<u32> = Vec::with_capacity(remaining);
        scalar_bw_n(&packed[processed * 2..], remaining, 16, &mut tail_idxs);
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw16_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v = vld1q_u8(src_ptr);
        let as_u16 = vreinterpretq_u16_u8(v);
        let lo = vmovl_u16(vget_low_u16(as_u16));
        let hi = vmovl_u16(vget_high_u16(as_u16));
        vst1q_u32(staging_ptr, lo);
        vst1q_u32(staging_ptr.add(4), hi);
        sink(staging)?;
        src_ptr = src_ptr.add(16);
    }
    Ok(())
}

/// NEON unpacker for bit_width = 15 — covers ~736K values across
/// l_orderkey / l_partkey / l_extendedprice / l_comment dict pages
/// at SF=1.
///
/// Per 8-row block: 15 bytes input, 32 bytes output. Uses a vextq
/// to construct a `bytes[7..23]` view for the high 4 lanes, then
/// shuffles each half into u32x4 windows.
pub fn unpack_indices_into_neon_bw15(
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
            "neon bw15: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;

    // Each block reads 16 + 16 bytes (overlapping), consumes 15. The
    // final iteration reads up to byte src+24, so packed must hold
    // 15*(full_blocks-1) + 24 bytes.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 15 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw15_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 15 / 8..], remaining, 15, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw15_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    // Lanes 0-3: u32 windows from v0 (bytes 0..16) at byte offsets
    // [0, 1, 3, 5]. Same shuffle as bw=14.
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr());
    // Lanes 4-7: u32 windows from v_hi (bytes 7..23) at byte offsets
    // [0, 2, 4, 6] within v_hi (= [7, 9, 11, 13] in the original block).
    let shuffle_hi: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    // Per-lane right shifts (negative for vshlq_s32).
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -7, -6, -5].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -3, -2, -1].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x7FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        // v_hi = bytes 7..23, loaded directly. Reading from src+7
        // means the final block's load reaches byte 7+16=23 past
        // the block start, hence the safe_full_blocks check above.
        let v_hi: uint8x16_t = vld1q_u8(src_ptr.add(7));

        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v_hi, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);
        src_ptr = src_ptr.add(15);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=15) + scalar dict gather.
pub fn unpack_lookup_into_neon_bw15<T: Copy>(
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
            "neon bw15 lookup: packed has {} bytes, need {}",
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
            unpack_neon_bw15_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw15_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw15_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -7, -6, -5].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -3, -2, -1].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x7FFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v_hi: uint8x16_t = vld1q_u8(src_ptr.add(7));
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v_hi, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(15);
    }
    Ok(())
}

/// NEON unpacker for bit_width = 18 — covers ~254K values in TPC-H
/// lineitem dict pages, mostly l_extendedprice tail and l_orderkey
/// overflow rows. Mirror of bw=15 with shifts `[0,-2,-4,-6]` and
/// mask `0x3FFFF`.
pub fn unpack_indices_into_neon_bw18(
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
            "neon bw18: packed has {} bytes, need {}",
            packed.len(),
            required_bytes
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;

    // Each block consumes 18 bytes; the v_hi load reads up to byte
    // src+24. Final iteration needs packed.len() ≥ 18*(full_blocks-1) + 24.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 18 * (full_blocks - 1) + 24 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw18_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 18 / 8..], remaining, 18, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw18_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    // Lanes 0-3: u32 windows from v0 at byte offsets [0, 2, 4, 6].
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    // Lanes 4-7: u32 windows from v_hi (bytes 7..23) at indices
    // [2, 4, 6, 8] (= bytes 9, 11, 13, 15 in the original block).
    let shuffle_hi: uint8x16_t =
        vld1q_u8([2u8, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11].as_ptr());
    // Lane shifts (symmetric every 4 lanes).
    let shifts: int32x4_t = vld1q_s32([0i32, -2, -4, -6].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        // v_hi = bytes 7..23 (loaded directly to avoid the
        // vextq_u8(v0, vld1q_u8(src+8), 7) pitfall — that would
        // give v0[7..16] + v1[0..7] = bytes 7..15 + bytes 8..14,
        // i.e., the second half is *repeated* bytes from v0).
        let v_hi: uint8x16_t = vld1q_u8(src_ptr.add(7));

        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v_hi, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);
        src_ptr = src_ptr.add(18);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

/// Fused NEON unpack (bw=18) + scalar dict gather.
pub fn unpack_lookup_into_neon_bw18<T: Copy>(
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
            "neon bw18 lookup: packed has {} bytes, need {}",
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
            unpack_neon_bw18_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw18_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw18_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([2u8, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11].as_ptr());
    let shifts: int32x4_t = vld1q_s32([0i32, -2, -4, -6].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3FFFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v_hi: uint8x16_t = vld1q_u8(src_ptr.add(7));
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v_hi, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(18);
    }
    Ok(())
}

/// Generic scalar streaming bit-buffer for any bit_width. Used for
/// tail handling in NEON-specialized paths.
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

/// Streaming-bit-buffer scalar fallback for a small remainder. Keeps
/// the public API simple: caller doesn't need to know about block
/// boundaries.
fn scalar_bw12(packed: &[u8], n: usize, out: &mut Vec<u32>) {
    let mut buf: u64 = 0;
    let mut bits: u32 = 0;
    let mut byte_idx = 0usize;
    for _ in 0..n {
        while bits < 12 {
            buf |= (packed[byte_idx] as u64) << bits;
            byte_idx += 1;
            bits += 8;
        }
        out.push((buf & 0xFFF) as u32);
        buf >>= 12;
        bits -= 12;
    }
}

// ---- bw=8: trivial byte-aligned NEON expansion ---------------------
//
// Each value is exactly 1 byte. 32 values = 32 bytes of source. The
// kernel widens each byte to u32 — `vld1q_u8` loads 16 bytes, then
// two `vmovl_u16(vmovl_u8(...))` chains expand to 16 u32s. Two of
// those covers the 32-value block.

/// `unpack_indices_into` for bit_width == 8. One block = 32 values =
/// 32 bytes. Significantly faster than the const-generic scalar path
/// (no inner per-value bit math).
pub fn unpack_indices_into_neon_bw8(
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
            "neon bw8: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_neon_bw8_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed..], remaining, 8, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw8_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        // 32 bytes per block = two 16-byte loads.
        let b0: uint8x16_t = vld1q_u8(src_ptr);
        let b1: uint8x16_t = vld1q_u8(src_ptr.add(16));
        // Widen each 16-byte vector into four u32x4 lanes.
        let lo0_u16 = vmovl_u8(vget_low_u8(b0));
        let hi0_u16 = vmovl_u8(vget_high_u8(b0));
        let lo1_u16 = vmovl_u8(vget_low_u8(b1));
        let hi1_u16 = vmovl_u8(vget_high_u8(b1));
        let q0_lo = vmovl_u16(vget_low_u16(lo0_u16));
        let q0_hi = vmovl_u16(vget_high_u16(lo0_u16));
        let q1_lo = vmovl_u16(vget_low_u16(hi0_u16));
        let q1_hi = vmovl_u16(vget_high_u16(hi0_u16));
        let q2_lo = vmovl_u16(vget_low_u16(lo1_u16));
        let q2_hi = vmovl_u16(vget_high_u16(lo1_u16));
        let q3_lo = vmovl_u16(vget_low_u16(hi1_u16));
        let q3_hi = vmovl_u16(vget_high_u16(hi1_u16));

        let dst = out_ptr.add(blk * 32);
        vst1q_u32(dst, q0_lo);
        vst1q_u32(dst.add(4), q0_hi);
        vst1q_u32(dst.add(8), q1_lo);
        vst1q_u32(dst.add(12), q1_hi);
        vst1q_u32(dst.add(16), q2_lo);
        vst1q_u32(dst.add(20), q2_hi);
        vst1q_u32(dst.add(24), q3_lo);
        vst1q_u32(dst.add(28), q3_hi);
        src_ptr = src_ptr.add(32);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=4: nibble-aligned NEON expansion ---------------------------
//
// 2 values per byte. 32 values = 16 bytes of source. Per byte: low
// nibble is value[i], high nibble is value[i+1].
//
// Strategy: load 16 bytes (= 32 values' worth), broadcast a 0x0F
// mask, extract low nibbles to one u8x16 and high nibbles (shift
// right 4) to another, then widen each to u32 lanes.

/// `unpack_indices_into` for bit_width == 4. One block = 32 values =
/// 16 bytes. Faster than the const-generic scalar path's per-value
/// bit math by ~4× on the common case.
pub fn unpack_indices_into_neon_bw4(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    // bw=4 packs 2 values per byte, but the scalar tail might consume
    // a partial byte, so we ceil here.
    let required_bytes = num_values.div_ceil(2);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw4: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_neon_bw4_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 2..], remaining, 4, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw4_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);
    let mask_nibble: uint8x16_t = vdupq_n_u8(0x0F);

    for blk in 0..full_blocks {
        // 16 source bytes → 32 nibbles → 32 u32 values.
        let bytes: uint8x16_t = vld1q_u8(src_ptr);
        // Low nibbles: byte & 0x0F.
        let lo_nibbles: uint8x16_t = vandq_u8(bytes, mask_nibble);
        // High nibbles: byte >> 4.
        let hi_nibbles: uint8x16_t = vshrq_n_u8(bytes, 4);
        // Interleave: parquet LSB-first packing puts value[2i] in the
        // low nibble of byte i and value[2i+1] in the high nibble.
        // We need output: [lo[0], hi[0], lo[1], hi[1], ...].
        // `vzip1q_u8` + `vzip2q_u8` produce exactly that pattern.
        let interleaved_lo: uint8x16_t = vzip1q_u8(lo_nibbles, hi_nibbles);
        let interleaved_hi: uint8x16_t = vzip2q_u8(lo_nibbles, hi_nibbles);

        // Widen each 16-byte vector into four u32x4 lanes.
        let il_u16 = vmovl_u8(vget_low_u8(interleaved_lo));
        let ih_u16 = vmovl_u8(vget_high_u8(interleaved_lo));
        let jl_u16 = vmovl_u8(vget_low_u8(interleaved_hi));
        let jh_u16 = vmovl_u8(vget_high_u8(interleaved_hi));
        let q0_lo = vmovl_u16(vget_low_u16(il_u16));
        let q0_hi = vmovl_u16(vget_high_u16(il_u16));
        let q1_lo = vmovl_u16(vget_low_u16(ih_u16));
        let q1_hi = vmovl_u16(vget_high_u16(ih_u16));
        let q2_lo = vmovl_u16(vget_low_u16(jl_u16));
        let q2_hi = vmovl_u16(vget_high_u16(jl_u16));
        let q3_lo = vmovl_u16(vget_low_u16(jh_u16));
        let q3_hi = vmovl_u16(vget_high_u16(jh_u16));

        let dst = out_ptr.add(blk * 32);
        vst1q_u32(dst, q0_lo);
        vst1q_u32(dst.add(4), q0_hi);
        vst1q_u32(dst.add(8), q1_lo);
        vst1q_u32(dst.add(12), q1_hi);
        vst1q_u32(dst.add(16), q2_lo);
        vst1q_u32(dst.add(20), q2_hi);
        vst1q_u32(dst.add(24), q3_lo);
        vst1q_u32(dst.add(28), q3_hi);
        src_ptr = src_ptr.add(16);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=4 / bw=6 / bw=8 lookup variants ----------------------------
//
// Fused NEON unpack + scalar dict gather, mirroring the
// `unpack_lookup_into_neon_bw{12,14}` shape: NEON fills a stack
// `staging` per block, then a sink closure gathers from `dict`. The
// fast path skips per-element bounds checks when the dict size proves
// every index fits; the slow path bounds-checks each lane.
//
// Targets:
//   bw=4 — small numeric dicts (l_quantity has ~50 distinct values
//          but TPC-H sometimes encodes a 4-bit subset; broader use
//          for boolean-product columns).
//   bw=6 — l_quantity (50 distinct values fits in 6 bits).
//   bw=8 — l_linenumber (7 distinct values, but 8-bit packing is
//          chosen when the writer rounds up; also l_returnflag /
//          l_linestatus byte-keys).

/// `unpack_lookup_into` for bit_width == 4. One NEON block = 32 values
/// = 16 source bytes.
pub fn unpack_lookup_into_neon_bw4<T: Copy>(
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
            "neon bw4 lookup: packed has {} bytes, need {required_bytes}",
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
            // Bounds-safe fast path: every 4-bit index fits in dict.
            unpack_neon_bw4_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_neon_bw4_into_staging(packed, full_blocks, &mut staging, |idxs| {
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw4_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();
    let mask_nibble: uint8x16_t = vdupq_n_u8(0x0F);

    for _ in 0..full_blocks {
        let bytes: uint8x16_t = vld1q_u8(src_ptr);
        let lo_nibbles: uint8x16_t = vandq_u8(bytes, mask_nibble);
        let hi_nibbles: uint8x16_t = vshrq_n_u8(bytes, 4);
        let interleaved_lo: uint8x16_t = vzip1q_u8(lo_nibbles, hi_nibbles);
        let interleaved_hi: uint8x16_t = vzip2q_u8(lo_nibbles, hi_nibbles);
        let il_u16 = vmovl_u8(vget_low_u8(interleaved_lo));
        let ih_u16 = vmovl_u8(vget_high_u8(interleaved_lo));
        let jl_u16 = vmovl_u8(vget_low_u8(interleaved_hi));
        let jh_u16 = vmovl_u8(vget_high_u8(interleaved_hi));
        vst1q_u32(staging_ptr, vmovl_u16(vget_low_u16(il_u16)));
        vst1q_u32(staging_ptr.add(4), vmovl_u16(vget_high_u16(il_u16)));
        vst1q_u32(staging_ptr.add(8), vmovl_u16(vget_low_u16(ih_u16)));
        vst1q_u32(staging_ptr.add(12), vmovl_u16(vget_high_u16(ih_u16)));
        vst1q_u32(staging_ptr.add(16), vmovl_u16(vget_low_u16(jl_u16)));
        vst1q_u32(staging_ptr.add(20), vmovl_u16(vget_high_u16(jl_u16)));
        vst1q_u32(staging_ptr.add(24), vmovl_u16(vget_low_u16(jh_u16)));
        vst1q_u32(staging_ptr.add(28), vmovl_u16(vget_high_u16(jh_u16)));
        sink(staging)?;
        src_ptr = src_ptr.add(16);
    }
    Ok(())
}

/// `unpack_lookup_into` for bit_width == 6. NEON block = 8 values =
/// 6 source bytes. Per-lane (byte, bit_off): (0,0), (0,6), (1,4),
/// (2,2), (3,0), (3,6), (4,4), (5,2). Each value spans ≤ 2 bytes;
/// load 4 source bytes per lane via PSHUFB-style TBL, then per-lane
/// right-shift via `vshlq_s32` with negative shifts, mask to 6 bits.
pub fn unpack_lookup_into_neon_bw6<T: Copy>(
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
            "neon bw6 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Kernel reads 16 bytes from packed[6*blk..]; the last safe block
    // satisfies 6*(safe-1) + 16 ≤ len, so safe ≤ (len - 10) / 6.
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
            // Bounds-safe fast path: every 6-bit index fits in dict.
            unpack_neon_bw6_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw6_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw6_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    // Lo lanes (0..3): u32 windows [0..4], [0..4], [1..5], [2..6].
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5].as_ptr());
    // Hi lanes (4..7): u32 windows [3..7], [3..7], [4..8], [5..9].
    let shuffle_hi: uint8x16_t =
        vld1q_u8([3u8, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8].as_ptr());
    // Per-lane right shifts (both halves share the same pattern).
    let shifts: int32x4_t = vld1q_s32([0i32, -6, -4, -2].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3F);
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(6);
    }
    Ok(())
}

/// `unpack_lookup_into` for bit_width == 8. Byte-aligned; one NEON
/// block = 32 values = 32 source bytes. No shifts or shuffles needed:
/// the source bytes ARE the indices, just widened to u32.
pub fn unpack_lookup_into_neon_bw8<T: Copy>(
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
            "neon bw8 lookup: packed has {} bytes, need {required_bytes}",
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
            unpack_neon_bw8_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_neon_bw8_into_staging(packed, full_blocks, &mut staging, |idxs| {
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
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw8_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let b0: uint8x16_t = vld1q_u8(src_ptr);
        let b1: uint8x16_t = vld1q_u8(src_ptr.add(16));
        let lo0_u16 = vmovl_u8(vget_low_u8(b0));
        let hi0_u16 = vmovl_u8(vget_high_u8(b0));
        let lo1_u16 = vmovl_u8(vget_low_u8(b1));
        let hi1_u16 = vmovl_u8(vget_high_u8(b1));
        vst1q_u32(staging_ptr, vmovl_u16(vget_low_u16(lo0_u16)));
        vst1q_u32(staging_ptr.add(4), vmovl_u16(vget_high_u16(lo0_u16)));
        vst1q_u32(staging_ptr.add(8), vmovl_u16(vget_low_u16(hi0_u16)));
        vst1q_u32(staging_ptr.add(12), vmovl_u16(vget_high_u16(hi0_u16)));
        vst1q_u32(staging_ptr.add(16), vmovl_u16(vget_low_u16(lo1_u16)));
        vst1q_u32(staging_ptr.add(20), vmovl_u16(vget_high_u16(lo1_u16)));
        vst1q_u32(staging_ptr.add(24), vmovl_u16(vget_low_u16(hi1_u16)));
        vst1q_u32(staging_ptr.add(28), vmovl_u16(vget_high_u16(hi1_u16)));
        sink(staging)?;
        src_ptr = src_ptr.add(32);
    }
    Ok(())
}

// ---- bw=2: 4-streams-per-byte NEON expansion -----------------------
//
// 4 values per byte. 32 values = 8 source bytes. Per byte:
//   value[0] = byte & 0b11
//   value[1] = (byte >> 2) & 0b11
//   value[2] = (byte >> 4) & 0b11
//   value[3] = (byte >> 6) & 0b11
//
// Strategy: load 8 source bytes into a uint8x8_t. Build 4 streams
// (s0..s3) by shifting + masking. Interleave 4-way using paired
// vzip1/vzip2 at u8 then u16 granularity so the output byte stream
// is [s0[0], s1[0], s2[0], s3[0], s0[1], s1[1], ...]. Then widen
// the 32-byte block to eight u32x4 lanes — same tail as bw=4 / bw=8.
//
// Targets l_shipinstruct (~4-element dict) and similar tiny dicts.

pub fn unpack_indices_into_neon_bw2(
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
            "neon bw2: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_neon_bw2_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 4..], remaining, 2, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw2_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);
    let mask2: uint8x8_t = vdup_n_u8(0x03);

    for blk in 0..full_blocks {
        // 8 source bytes = 32 values.
        let src: uint8x8_t = vld1_u8(src_ptr);
        // Four parallel streams, one per bit-pair within the byte.
        let s0 = vand_u8(src, mask2);
        let s1 = vand_u8(vshr_n_u8(src, 2), mask2);
        let s2 = vand_u8(vshr_n_u8(src, 4), mask2);
        // Top 2 bits — no mask needed since shift produces zero high bits.
        let s3 = vshr_n_u8(src, 6);

        // Pair (s0, s1) and (s2, s3): vzip1/vzip2 on uint8x8_t zip
        // 4 lanes from each input. Result: [s0[i], s1[i], s0[i+1], ...].
        let p01_a: uint8x8_t = vzip1_u8(s0, s1);
        let p01_b: uint8x8_t = vzip2_u8(s0, s1);
        let p23_a: uint8x8_t = vzip1_u8(s2, s3);
        let p23_b: uint8x8_t = vzip2_u8(s2, s3);

        // 4-way interleave via u16 zip: each lane of p01 holds an
        // (s0, s1) byte-pair → 16-bit element. Zipping at u16
        // granularity against p23 yields the final 4-way pattern
        // [s0, s1, s2, s3, s0, s1, s2, s3, ...].
        let p01_a_u16 = vreinterpret_u16_u8(p01_a);
        let p23_a_u16 = vreinterpret_u16_u8(p23_a);
        let p01_b_u16 = vreinterpret_u16_u8(p01_b);
        let p23_b_u16 = vreinterpret_u16_u8(p23_b);

        let q0_u16 = vzip1_u16(p01_a_u16, p23_a_u16);
        let q1_u16 = vzip2_u16(p01_a_u16, p23_a_u16);
        let q2_u16 = vzip1_u16(p01_b_u16, p23_b_u16);
        let q3_u16 = vzip2_u16(p01_b_u16, p23_b_u16);

        // Two 16-byte blocks covering values 0..16 and 16..32.
        let block_lo: uint8x16_t =
            vcombine_u8(vreinterpret_u8_u16(q0_u16), vreinterpret_u8_u16(q1_u16));
        let block_hi: uint8x16_t =
            vcombine_u8(vreinterpret_u8_u16(q2_u16), vreinterpret_u8_u16(q3_u16));

        // Widen u8 → u32, identical shape to bw=4 / bw=8.
        let lo0_u16 = vmovl_u8(vget_low_u8(block_lo));
        let hi0_u16 = vmovl_u8(vget_high_u8(block_lo));
        let lo1_u16 = vmovl_u8(vget_low_u8(block_hi));
        let hi1_u16 = vmovl_u8(vget_high_u8(block_hi));
        let q0_lo = vmovl_u16(vget_low_u16(lo0_u16));
        let q0_hi = vmovl_u16(vget_high_u16(lo0_u16));
        let q1_lo = vmovl_u16(vget_low_u16(hi0_u16));
        let q1_hi = vmovl_u16(vget_high_u16(hi0_u16));
        let q2_lo = vmovl_u16(vget_low_u16(lo1_u16));
        let q2_hi = vmovl_u16(vget_high_u16(lo1_u16));
        let q3_lo = vmovl_u16(vget_low_u16(hi1_u16));
        let q3_hi = vmovl_u16(vget_high_u16(hi1_u16));

        let dst = out_ptr.add(blk * 32);
        vst1q_u32(dst, q0_lo);
        vst1q_u32(dst.add(4), q0_hi);
        vst1q_u32(dst.add(8), q1_lo);
        vst1q_u32(dst.add(12), q1_hi);
        vst1q_u32(dst.add(16), q2_lo);
        vst1q_u32(dst.add(20), q2_hi);
        vst1q_u32(dst.add(24), q3_lo);
        vst1q_u32(dst.add(28), q3_hi);

        src_ptr = src_ptr.add(8);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=3: NEON, per-lane TBL + variable shift ---------------------
//
// 8 values per block = 3 source bytes. Per-lane (byte, bit_off):
//   (0,0), (0,3), (0,6), (1,1), (1,4), (1,7), (2,2), (2,5).
// Each value spans at most 2 source bytes (3 + 7 = 10 < 16), so we
// load 4 source bytes per lane as a u32, shift right by the lane's
// bit_off, and mask to 3 bits. Two shuffles (`shuffle_lo` /
// `shuffle_hi`) gather the right 4-byte windows for lanes 0..3 and
// 4..7. The 16-byte register load covers up to byte 5 — same single-
// load shape as the bw=5 kernel.
//
// Targets l_shipmode-style dicts with 5-8 distinct values.

pub fn unpack_indices_into_neon_bw3(
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
            "neon bw3: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Each kernel iteration loads 16 bytes from `packed[3*blk..]`.
    // We need `3*blk + 16 <= packed.len()` for every block we keep,
    // i.e. the last safe block satisfies `3*(safe-1) + 16 <= len`,
    // so `safe <= (len - 13) / 3`.
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 13) / 3).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw3_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 3 / 8..], remaining, 3, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw3_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // Lo lanes (0..3): values at (0,0), (0,3), (0,6), (1,1).
    // Load byte windows [0..4], [0..4], [0..4], [1..5] respectively.
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4].as_ptr());
    // Hi lanes (4..7): values at (1,4), (1,7), (2,2), (2,5).
    // Load byte windows [1..5], [1..5], [2..6], [2..6].
    let shuffle_hi: uint8x16_t =
        vld1q_u8([1u8, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 5].as_ptr());
    // Per-lane right shifts (encoded as negative for vshlq_s32).
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -3, -6, -1].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -7, -2, -5].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x07);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        // One 16-byte load covers both shuffles (max byte index is 5).
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);

        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);

        src_ptr = src_ptr.add(3);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=1: bit-test NEON unpack ------------------------------------
//
// 8 values per byte. 32 values = 4 source bytes. Strategy: broadcast
// each source byte into 8 lanes, AND with a per-lane bit-mask, and
// produce 0 / 1 outputs. The bit-mask layout per byte is
// [1, 2, 4, 8, 16, 32, 64, 128] for value indices [0..8].

pub fn unpack_indices_into_neon_bw1(
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
            "neon bw1: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 32;

    unsafe {
        unpack_neon_bw1_unchecked(packed, full_blocks, out);
    }

    let processed = full_blocks * 32;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed / 8..], remaining, 1, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw1_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    // Per-byte bit-mask vector (parquet LSB-first: value[0] = bit 0).
    let bit_masks_lo: uint32x4_t = vld1q_u32([1u32, 2, 4, 8].as_ptr());
    let bit_masks_hi: uint32x4_t = vld1q_u32([16u32, 32, 64, 128].as_ptr());

    for blk in 0..full_blocks {
        // 4 source bytes per block = 32 values.
        for b in 0..4 {
            let byte_val = *src_ptr.add(b) as u32;
            let v: uint32x4_t = vdupq_n_u32(byte_val);
            // AND with the lo half mask, compare against the mask: each
            // lane gets all-1s if its bit is set, 0 otherwise. AND with
            // 1 to land 0/1.
            let lo_anded = vandq_u32(v, bit_masks_lo);
            let lo_cmp = vceqq_u32(lo_anded, bit_masks_lo);
            let lo_out = vandq_u32(lo_cmp, vdupq_n_u32(1));
            let hi_anded = vandq_u32(v, bit_masks_hi);
            let hi_cmp = vceqq_u32(hi_anded, bit_masks_hi);
            let hi_out = vandq_u32(hi_cmp, vdupq_n_u32(1));
            vst1q_u32(out_ptr.add(blk * 32 + b * 8), lo_out);
            vst1q_u32(out_ptr.add(blk * 32 + b * 8 + 4), hi_out);
        }
        src_ptr = src_ptr.add(4);
    }
    out.set_len(out_start_len + full_blocks * 32);
}

// ---- bw=20: NEON, mirrors bw=17/18 shape ---------------------------
//
// 8 values per block = 20 source bytes. Per-lane start bytes
// [0, 2, 5, 7, 10, 12, 15, 17] with bit offsets [0, 4, 0, 4, 0, 4,
// 0, 4]. Each lane reads a 4-byte little-endian u32.

pub fn unpack_indices_into_neon_bw20(
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
            "neon bw20: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Each block reads up to byte 26; safety guard for the final iter.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 20 * (full_blocks - 1) + 26 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw20_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 20 / 8..], remaining, 20, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw20_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // bw=20 per-lane start bytes: [0, 2, 5, 7] lo, [10, 12, 15, 17] hi.
    // Lo lanes read from v0 = packed[0..16].
    // Hi lanes read from v1 = packed[10..26], so within v1 the per-lane
    // start bytes are [0, 2, 5, 7] (= [10-10, 12-10, 15-10, 17-10]).
    let shuffle: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8, 7, 8, 9, 10].as_ptr());
    // Variable per-lane right-shift (LSB-first packing).
    let shifts: int32x4_t = vld1q_s32([0i32, -4, 0, -4].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x0F_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(10));
        let lo_b = vqtbl1q_u8(v0, shuffle);
        let hi_b = vqtbl1q_u8(v1, shuffle);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);

        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);

        src_ptr = src_ptr.add(20);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=21: NEON, lane-specific shifts ----------------------------
//
// 8 values per block = 21 source bytes. Per-lane start bytes
// [0, 2, 5, 7, 10, 13, 15, 18] with bit offsets [0, 5, 2, 7, 4, 1, 6, 3].
// Lo and hi halves use different shuffle tables because the spacings
// differ between halves.

pub fn unpack_indices_into_neon_bw21(
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
            "neon bw21: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Reads up to byte 22 per block; safety guard.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 21 * (full_blocks - 1) + 22 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw21_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 21 / 8..], remaining, 21, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw21_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // Lo lanes (from v0 = packed[0..16]): start bytes [0, 2, 5, 7].
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8, 7, 8, 9, 10].as_ptr());
    // Hi lanes (from v1 = packed[10..26]): start bytes within v1 are
    // [13-10, 15-10, 18-10] = [3, 5, 8] for lanes 5, 6, 7; lane 4 is
    // [10-10] = 0.
    let shuffle_hi: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 3, 4, 5, 6, 5, 6, 7, 8, 8, 9, 10, 11].as_ptr());
    // Per-lane right shifts: lanes 0..3 = [0, -5, -2, -7];
    // lanes 4..7 = [-4, -1, -6, -3].
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -5, -2, -7].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -1, -6, -3].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1F_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(10));
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v1, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);

        vst1q_u32(out_ptr.add(blk * 8), lo_masked);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi_masked);

        src_ptr = src_ptr.add(21);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=5: NEON, lane-specific shifts per-byte ---------------------
//
// 8 values per block = 5 source bytes. Per-lane start (byte, bit_off):
// (0,0), (0,5), (1,2), (1,7), (2,4), (3,1), (3,6), (4,3). Each value
// spans at most 2 source bytes (5 + 7 = 12 < 16), so we extract one
// u16 per lane, variable-shift right, mask to 5 bits, widen to u32.

pub fn unpack_indices_into_neon_bw5(
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
            "neon bw5: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    // Each block reads bytes 0..=5 inclusive (lane 7 needs byte 4 + 5
    // for its 2-byte u16 window). The 16-byte vector load reads through
    // byte 15 of the current block's offset; safety check covers it.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 5 * (full_blocks - 1) + 16 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw5_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 5 / 8..], remaining, 5, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw5_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;

    // Per-lane u16 windows (16 bytes = 8 u16 little-endian):
    // lane 0 = bytes [0, 1], lane 1 = [0, 1], lane 2 = [1, 2],
    // lane 3 = [1, 2], lane 4 = [2, 3], lane 5 = [3, 4],
    // lane 6 = [3, 4], lane 7 = [4, 5].
    let shuffle: uint8x16_t = vld1q_u8([0u8, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5].as_ptr());
    // Per-lane right-shift in bits (LSB-first packing).
    let shifts: int16x8_t = vld1q_s16([0i16, -5, -2, -7, -4, -1, -6, -3].as_ptr());
    let mask: uint16x8_t = vdupq_n_u16(0x001F);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v: uint8x16_t = vld1q_u8(src_ptr);
        let shuffled: uint8x16_t = vqtbl1q_u8(v, shuffle);
        let as_u16: uint16x8_t = vreinterpretq_u16_u8(shuffled);
        let shifted: uint16x8_t =
            vreinterpretq_u16_s16(vshlq_s16(vreinterpretq_s16_u16(as_u16), shifts));
        let masked: uint16x8_t = vandq_u16(shifted, mask);
        let lo: uint32x4_t = vmovl_u16(vget_low_u16(masked));
        let hi: uint32x4_t = vmovl_u16(vget_high_u16(masked));

        vst1q_u32(out_ptr.add(blk * 8), lo);
        vst1q_u32(out_ptr.add(blk * 8 + 4), hi);

        src_ptr = src_ptr.add(5);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ============================================================
// Fused unpack + dict-gather kernels for the widths that already
// have raw-indices SIMD coverage but were going through the
// const-generic scalar lookup path.
//
// Each staging helper mirrors the raw-indices unpacker exactly but
// writes into a stack `[u32; N]` and invokes a per-block `sink`
// callback, so the outer wrapper can run either the bounds-safe
// fast-path gather (skipped per-element bounds check when
// `dict_size` proves every unpacked index fits) or the
// bounds-checked path.
// ============================================================

// ---- bw=1: lookup --------------------------------------------------

pub fn unpack_lookup_into_neon_bw1<T: Copy>(
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
    let required_bytes = num_values.div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw1 lookup: packed has {} bytes, need {required_bytes}",
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
        if dict_size > 1 {
            unpack_neon_bw1_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_neon_bw1_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed / 8..], remaining, 1, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw1_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    let bit_masks_lo: uint32x4_t = vld1q_u32([1u32, 2, 4, 8].as_ptr());
    let bit_masks_hi: uint32x4_t = vld1q_u32([16u32, 32, 64, 128].as_ptr());
    let one: uint32x4_t = vdupq_n_u32(1);

    for _ in 0..full_blocks {
        for b in 0..4 {
            let byte_val = *src_ptr.add(b) as u32;
            let v: uint32x4_t = vdupq_n_u32(byte_val);
            let lo_anded = vandq_u32(v, bit_masks_lo);
            let lo_cmp = vceqq_u32(lo_anded, bit_masks_lo);
            let lo_out = vandq_u32(lo_cmp, one);
            let hi_anded = vandq_u32(v, bit_masks_hi);
            let hi_cmp = vceqq_u32(hi_anded, bit_masks_hi);
            let hi_out = vandq_u32(hi_cmp, one);
            vst1q_u32(staging_ptr.add(b * 8), lo_out);
            vst1q_u32(staging_ptr.add(b * 8 + 4), hi_out);
        }
        sink(staging)?;
        src_ptr = src_ptr.add(4);
    }
    Ok(())
}

// ---- bw=2: lookup --------------------------------------------------

pub fn unpack_lookup_into_neon_bw2<T: Copy>(
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
    let required_bytes = num_values.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw2 lookup: packed has {} bytes, need {required_bytes}",
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
        if dict_size > 3 {
            unpack_neon_bw2_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 32;
                Ok(())
            })?;
        } else {
            unpack_neon_bw2_into_staging(packed, full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed / 4..], remaining, 2, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw2_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 32],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 32]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();
    let mask2: uint8x8_t = vdup_n_u8(0x03);

    for _ in 0..full_blocks {
        let src: uint8x8_t = vld1_u8(src_ptr);
        let s0 = vand_u8(src, mask2);
        let s1 = vand_u8(vshr_n_u8(src, 2), mask2);
        let s2 = vand_u8(vshr_n_u8(src, 4), mask2);
        let s3 = vshr_n_u8(src, 6);
        let p01_a: uint8x8_t = vzip1_u8(s0, s1);
        let p01_b: uint8x8_t = vzip2_u8(s0, s1);
        let p23_a: uint8x8_t = vzip1_u8(s2, s3);
        let p23_b: uint8x8_t = vzip2_u8(s2, s3);
        let p01_a_u16 = vreinterpret_u16_u8(p01_a);
        let p23_a_u16 = vreinterpret_u16_u8(p23_a);
        let p01_b_u16 = vreinterpret_u16_u8(p01_b);
        let p23_b_u16 = vreinterpret_u16_u8(p23_b);
        let q0_u16 = vzip1_u16(p01_a_u16, p23_a_u16);
        let q1_u16 = vzip2_u16(p01_a_u16, p23_a_u16);
        let q2_u16 = vzip1_u16(p01_b_u16, p23_b_u16);
        let q3_u16 = vzip2_u16(p01_b_u16, p23_b_u16);
        let block_lo: uint8x16_t =
            vcombine_u8(vreinterpret_u8_u16(q0_u16), vreinterpret_u8_u16(q1_u16));
        let block_hi: uint8x16_t =
            vcombine_u8(vreinterpret_u8_u16(q2_u16), vreinterpret_u8_u16(q3_u16));
        let lo0_u16 = vmovl_u8(vget_low_u8(block_lo));
        let hi0_u16 = vmovl_u8(vget_high_u8(block_lo));
        let lo1_u16 = vmovl_u8(vget_low_u8(block_hi));
        let hi1_u16 = vmovl_u8(vget_high_u8(block_hi));
        vst1q_u32(staging_ptr, vmovl_u16(vget_low_u16(lo0_u16)));
        vst1q_u32(staging_ptr.add(4), vmovl_u16(vget_high_u16(lo0_u16)));
        vst1q_u32(staging_ptr.add(8), vmovl_u16(vget_low_u16(hi0_u16)));
        vst1q_u32(staging_ptr.add(12), vmovl_u16(vget_high_u16(hi0_u16)));
        vst1q_u32(staging_ptr.add(16), vmovl_u16(vget_low_u16(lo1_u16)));
        vst1q_u32(staging_ptr.add(20), vmovl_u16(vget_high_u16(lo1_u16)));
        vst1q_u32(staging_ptr.add(24), vmovl_u16(vget_low_u16(hi1_u16)));
        vst1q_u32(staging_ptr.add(28), vmovl_u16(vget_high_u16(hi1_u16)));
        sink(staging)?;
        src_ptr = src_ptr.add(8);
    }
    Ok(())
}

// ---- bw=3: lookup --------------------------------------------------

pub fn unpack_lookup_into_neon_bw3<T: Copy>(
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
    let required_bytes = (num_values * 3).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw3 lookup: packed has {} bytes, need {required_bytes}",
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

    let dict_size = dict.len();
    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();
    let mut staging = [0u32; 8];
    let mut bad_idx: Option<u32> = None;

    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        let mut written = 0usize;
        if dict_size > 7 {
            unpack_neon_bw3_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw3_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed * 3 / 8..], remaining, 3, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw3_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([1u8, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 5].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -3, -6, -1].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -7, -2, -5].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x07);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(staging_ptr, vandq_u32(lo_shifted, mask));
        vst1q_u32(staging_ptr.add(4), vandq_u32(hi_shifted, mask));
        sink(staging)?;
        src_ptr = src_ptr.add(3);
    }
    Ok(())
}

// ---- bw=5: lookup --------------------------------------------------

pub fn unpack_lookup_into_neon_bw5<T: Copy>(
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
    let required_bytes = (num_values * 5).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw5 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 5 * (full_blocks - 1) + 16 {
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
        if dict_size > 31 {
            unpack_neon_bw5_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw5_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed * 5 / 8..], remaining, 5, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw5_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle: uint8x16_t = vld1q_u8([0u8, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5].as_ptr());
    let shifts: int16x8_t = vld1q_s16([0i16, -5, -2, -7, -4, -1, -6, -3].as_ptr());
    let mask: uint16x8_t = vdupq_n_u16(0x001F);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v: uint8x16_t = vld1q_u8(src_ptr);
        let shuffled: uint8x16_t = vqtbl1q_u8(v, shuffle);
        let as_u16: uint16x8_t = vreinterpretq_u16_u8(shuffled);
        let shifted: uint16x8_t =
            vreinterpretq_u16_s16(vshlq_s16(vreinterpretq_s16_u16(as_u16), shifts));
        let masked: uint16x8_t = vandq_u16(shifted, mask);
        let lo: uint32x4_t = vmovl_u16(vget_low_u16(masked));
        let hi: uint32x4_t = vmovl_u16(vget_high_u16(masked));
        vst1q_u32(staging_ptr, lo);
        vst1q_u32(staging_ptr.add(4), hi);
        sink(staging)?;
        src_ptr = src_ptr.add(5);
    }
    Ok(())
}

// ---- bw=20: lookup -------------------------------------------------

pub fn unpack_lookup_into_neon_bw20<T: Copy>(
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
    let required_bytes = (num_values * 20).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw20 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 20 * (full_blocks - 1) + 26 {
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
        if dict_size > (1usize << 20) - 1 {
            unpack_neon_bw20_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw20_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed * 20 / 8..], remaining, 20, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw20_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8, 7, 8, 9, 10].as_ptr());
    let shifts: int32x4_t = vld1q_s32([0i32, -4, 0, -4].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x0F_FFFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(10));
        let lo_b = vqtbl1q_u8(v0, shuffle);
        let hi_b = vqtbl1q_u8(v1, shuffle);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        vst1q_u32(staging_ptr, vandq_u32(lo_shifted, mask));
        vst1q_u32(staging_ptr.add(4), vandq_u32(hi_shifted, mask));
        sink(staging)?;
        src_ptr = src_ptr.add(20);
    }
    Ok(())
}

// ---- bw=21: lookup -------------------------------------------------

pub fn unpack_lookup_into_neon_bw21<T: Copy>(
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
    let required_bytes = (num_values * 21).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw21 lookup: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);

    let full_blocks = num_values / 8;
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 21 * (full_blocks - 1) + 22 {
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
        if dict_size > (1usize << 21) - 1 {
            unpack_neon_bw21_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    *out_ptr.add(written + lane) = *dict_ptr.add(i as usize);
                }
                written += 8;
                Ok(())
            })?;
        } else {
            unpack_neon_bw21_into_staging(packed, safe_full_blocks, &mut staging, |idxs| {
                for (lane, &i) in idxs.iter().enumerate() {
                    let iu = i as usize;
                    if iu >= dict_size {
                        bad_idx = Some(i);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: i,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + lane) = *dict_ptr.add(iu);
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
        scalar_bw_n(&packed[processed * 21 / 8..], remaining, 21, &mut tail_idxs);
        for &i in &tail_idxs {
            let iu = i as usize;
            if iu >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: i,
                    dict_size,
                });
            }
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(iu);
                out.set_len(out.len() + 1);
            }
        }
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw21_into_staging<F>(
    packed: &[u8],
    full_blocks: usize,
    staging: &mut [u32; 8],
    mut sink: F,
) -> Result<()>
where
    F: FnMut(&[u32; 8]) -> Result<()>,
{
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8, 7, 8, 9, 10].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 3, 4, 5, 6, 5, 6, 7, 8, 8, 9, 10, 11].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -5, -2, -7].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -1, -6, -3].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1F_FFFF);

    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(10));
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v1, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(staging_ptr, vandq_u32(lo_shifted, mask));
        vst1q_u32(staging_ptr.add(4), vandq_u32(hi_shifted, mask));
        sink(staging)?;
        src_ptr = src_ptr.add(21);
    }
    Ok(())
}

// ---- bw=6: raw-indices NEON ----------------------------------------
//
// 8 values per block = 6 source bytes. Per-lane (byte, bit_off):
// (0,0), (0,6), (1,4), (2,2), (3,0), (3,6), (4,4), (5,2). Each value
// spans ≤ 2 bytes; 4-byte u32 windows + variable right-shift, mask
// to 6 bits.
//
// Targets dicts of 33–64 distinct values (status enums, country
// codes, etc.).

pub fn unpack_indices_into_neon_bw6(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 6).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw6: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Kernel reads 16 bytes from packed[6*blk..]; last safe block at
    // 6*(safe-1) + 16 ≤ len → safe ≤ (len - 10) / 6.
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 10) / 6).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw6_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 6 / 8..], remaining, 6, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw6_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([3u8, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8].as_ptr());
    let shifts: int32x4_t = vld1q_s32([0i32, -6, -4, -2].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3F);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(6);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=7: raw-indices NEON ----------------------------------------
//
// 8 values per block = 7 source bytes. Per-lane bit_off [0, 7, 6, 5,
// 4, 3, 2, 1] (every lane different). Per-lane start byte [0, 0, 1,
// 2, 3, 4, 5, 6]. Each value spans ≤ 2 bytes.

pub fn unpack_indices_into_neon_bw7(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 7).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw7: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Kernel reads 16 bytes from packed[7*blk..]; last safe block at
    // 7*(safe-1) + 16 ≤ len → safe ≤ (len - 9) / 7.
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 9) / 7).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw7_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 7 / 8..], remaining, 7, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw7_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    // Lo lanes (0..3): u32 windows [0..4], [0..4], [1..5], [2..6].
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5].as_ptr());
    // Hi lanes (4..7): u32 windows [3..7], [4..8], [5..9], [6..10].
    let shuffle_hi: uint8x16_t =
        vld1q_u8([3u8, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -7, -6, -5].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -3, -2, -1].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x7F);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(7);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=9: raw-indices NEON ----------------------------------------
//
// 8 values per block = 9 source bytes. Per-lane start bytes
// [0, 1, 2, 3, 4, 5, 6, 7] with bit offsets [0, 1, 2, 3, 4, 5, 6, 7].

pub fn unpack_indices_into_neon_bw9(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 9).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw9: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Reads 16 bytes from packed[9*blk..]; last safe block at
    // 9*(safe-1) + 16 ≤ len → safe ≤ (len - 7) / 9.
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 7) / 9).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw9_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 9 / 8..], remaining, 9, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw9_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([4u8, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -1, -2, -3].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -5, -6, -7].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1FF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(9);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=10: raw-indices NEON ---------------------------------------
//
// 8 values per block = 10 source bytes. Per-lane start bytes
// [0, 1, 2, 3, 5, 6, 7, 8] with bit offsets [0, 2, 4, 6, 0, 2, 4, 6].

pub fn unpack_indices_into_neon_bw10(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 10).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw10: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Reads 16 bytes; last safe block at 10*(safe-1) + 16 ≤ len.
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 6) / 10).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw10_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 10 / 8..], remaining, 10, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw10_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([5u8, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 8, 9, 10, 11].as_ptr());
    let shifts: int32x4_t = vld1q_s32([0i32, -2, -4, -6].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x3FF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(10);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=11: raw-indices NEON ---------------------------------------
//
// 8 values per block = 11 source bytes. Per-lane start bytes
// [0, 1, 2, 4, 5, 6, 8, 9] with bit offsets [0, 3, 6, 1, 4, 7, 2, 5].

pub fn unpack_indices_into_neon_bw11(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 11).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw11: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 5) / 11).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw11_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 11 / 8..], remaining, 11, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw11_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 4, 5, 6, 7].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([5u8, 6, 7, 8, 6, 7, 8, 9, 8, 9, 10, 11, 9, 10, 11, 12].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -3, -6, -1].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -7, -2, -5].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x7FF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(11);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=13: raw-indices NEON ---------------------------------------
//
// 8 values per block = 13 source bytes. Per-lane start bytes
// [0, 1, 3, 4, 6, 8, 9, 11] with bit offsets [0, 5, 2, 7, 4, 1, 6, 3].
// Lane 7 reads bytes [11..15], so max byte index is 14 → fits in one
// 16-byte load.

pub fn unpack_indices_into_neon_bw13(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 13).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw13: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    let safe_full_blocks = if packed.len() < 16 {
        0
    } else {
        ((packed.len() - 3) / 13).min(full_blocks)
    };

    unsafe {
        unpack_neon_bw13_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 13 / 8..], remaining, 13, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw13_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 4, 5, 6, 7].as_ptr());
    let shuffle_hi: uint8x16_t =
        vld1q_u8([6u8, 7, 8, 9, 8, 9, 10, 11, 9, 10, 11, 12, 11, 12, 13, 14].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -5, -2, -7].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -1, -6, -3].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x1FFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v0, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(13);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ---- bw=19: raw-indices NEON ---------------------------------------
//
// 8 values per block = 19 source bytes. Per-lane start bytes
// [0, 2, 4, 7, 9, 11, 14, 16] with bit offsets [0, 3, 6, 1, 4, 7, 2, 5].
// Hi lanes start at byte 9 and beyond; we use a second 16-byte load
// at offset 9 so the per-lane windows for lanes 4..7 reduce to
// [0, 2, 5, 7] within v1.

pub fn unpack_indices_into_neon_bw19(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = (num_values * 19).div_ceil(8);
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw19: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Two loads per block: v0 = packed[19*blk..+16], v1 =
    // packed[19*blk+9..+16]. Last byte read at 19*blk + 9 + 10 = 19*blk
    // + 19 = 19*(blk+1). Safety: 19*safe ≤ len + (16-final non-read).
    // Conservatively require 19*(safe-1) + 25 ≤ len for the final iter.
    let safe_full_blocks = if full_blocks == 0 {
        0
    } else if packed.len() >= 19 * (full_blocks - 1) + 25 {
        full_blocks
    } else {
        full_blocks - 1
    };

    unsafe {
        unpack_neon_bw19_unchecked(packed, safe_full_blocks, out);
    }

    let processed = safe_full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 19 / 8..], remaining, 19, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw19_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    // Lo lanes (0..3) from v0: start bytes [0, 2, 4, 7].
    let shuffle_lo: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 7, 8, 9, 10].as_ptr());
    // Hi lanes (4..7) from v1 = packed[9..25]: start bytes [0, 2, 5, 7]
    // (= [9-9, 11-9, 14-9, 16-9]).
    let shuffle_hi: uint8x16_t =
        vld1q_u8([0u8, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8, 7, 8, 9, 10].as_ptr());
    let shifts_lo: int32x4_t = vld1q_s32([0i32, -3, -6, -1].as_ptr());
    let shifts_hi: int32x4_t = vld1q_s32([-4i32, -7, -2, -5].as_ptr());
    let mask: uint32x4_t = vdupq_n_u32(0x7_FFFF);

    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);

    for blk in 0..full_blocks {
        let v0 = vld1q_u8(src_ptr);
        let v1 = vld1q_u8(src_ptr.add(9));
        let lo_b = vqtbl1q_u8(v0, shuffle_lo);
        let hi_b = vqtbl1q_u8(v1, shuffle_hi);
        let lo = vreinterpretq_u32_u8(lo_b);
        let hi = vreinterpretq_u32_u8(hi_b);
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
        vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask));
        vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask));
        src_ptr = src_ptr.add(19);
    }
    out.set_len(out_start_len + full_blocks * 8);
}

// ============================================================
// bw=22..32: wide widths. Per-block geometry pre-tabulated by
// `start_bit = lane * bw`, `start_byte = start_bit / 8`,
// `bit_off = start_bit % 8`.
//
// For bw=22..25 each value fits in a 4-byte little-endian u32
// window (max 25 + 7 = 32 bits). For bw=26..31 the value can span
// 5 bytes, so we gather 8 bytes per lane into a u64 staging buffer.
// bw=32 is byte-aligned (no shift / no mask).
//
// Dict sizes that need these widths exceed 2M distinct values;
// rare in practice but the kernels close the SIMD specialisation
// table so the const-generic scalar path is never the only option.
// ============================================================

#[inline]
fn read_u32_le_at(packed: &[u8], off: usize) -> u32 {
    let b0 = packed[off] as u32;
    let b1 = packed[off + 1] as u32;
    let b2 = packed[off + 2] as u32;
    let b3 = packed[off + 3] as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

#[inline]
fn read_u64_le_at(packed: &[u8], off: usize) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..8 {
        acc |= (packed[off + i] as u64) << (i * 8);
    }
    acc
}

// ---- bw=22..25: u32 staging ----------------------------------------

macro_rules! neon_wide_u32_kernel {
    ($pubfn:ident, $unchecked:ident, $bw:literal, $offsets:expr, $shifts:expr, $mask:literal, $block_bytes:literal) => {
        pub fn $pubfn(packed: &[u8], num_values: usize, out: &mut Vec<u32>) -> Result<()> {
            if num_values == 0 {
                return Ok(());
            }
            let required_bytes = (num_values * $bw).div_ceil(8);
            if packed.len() < required_bytes {
                return Err(CodecError::Decompress(format!(
                    concat!("neon bw", stringify!($bw), ": packed has {} bytes, need {}"),
                    packed.len(),
                    required_bytes
                )));
            }
            out.reserve(num_values);
            let full_blocks = num_values / 8;
            // Last lane reads [start_byte + 4], so the final block needs
            // start_byte_of_last_lane + 4 bytes available. Compute:
            // max_byte_per_block = last_lane_start + 4.
            let offsets: [usize; 8] = $offsets;
            let max_read_per_block = offsets[7] + 4;
            let safe_full_blocks = if full_blocks == 0 {
                0
            } else if packed.len() >= $block_bytes * (full_blocks - 1) + max_read_per_block {
                full_blocks
            } else {
                full_blocks - 1
            };
            unsafe {
                $unchecked(packed, safe_full_blocks, out);
            }
            let processed = safe_full_blocks * 8;
            let remaining = num_values - processed;
            if remaining > 0 {
                scalar_bw_n(&packed[processed * $bw / 8..], remaining, $bw, out);
            }
            Ok(())
        }

        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn $unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
            use std::arch::aarch64::*;
            let offsets: [usize; 8] = $offsets;
            let shifts: [i32; 8] = $shifts;
            let mask_v: uint32x4_t = vdupq_n_u32($mask as u32);
            let shifts_lo: int32x4_t = vld1q_s32(shifts[0..4].as_ptr());
            let shifts_hi: int32x4_t = vld1q_s32(shifts[4..8].as_ptr());

            let mut src_ptr = packed.as_ptr();
            let out_start_len = out.len();
            let out_ptr = out.as_mut_ptr().add(out_start_len);

            for blk in 0..full_blocks {
                let lane0 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[0],
                );
                let lane1 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[1],
                );
                let lane2 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[2],
                );
                let lane3 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[3],
                );
                let lane4 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[4],
                );
                let lane5 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[5],
                );
                let lane6 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[6],
                );
                let lane7 = read_u32_le_at(
                    std::slice::from_raw_parts(src_ptr, offsets[7] + 4),
                    offsets[7],
                );
                let lo_arr: [u32; 4] = [lane0, lane1, lane2, lane3];
                let hi_arr: [u32; 4] = [lane4, lane5, lane6, lane7];
                let lo: uint32x4_t = vld1q_u32(lo_arr.as_ptr());
                let hi: uint32x4_t = vld1q_u32(hi_arr.as_ptr());
                let lo_shifted =
                    vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(lo), shifts_lo));
                let hi_shifted =
                    vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(hi), shifts_hi));
                vst1q_u32(out_ptr.add(blk * 8), vandq_u32(lo_shifted, mask_v));
                vst1q_u32(out_ptr.add(blk * 8 + 4), vandq_u32(hi_shifted, mask_v));
                src_ptr = src_ptr.add($block_bytes);
            }
            out.set_len(out_start_len + full_blocks * 8);
        }
    };
}

// Per-width tables: offsets are start_byte_per_lane; shifts are
// -bit_off_per_lane (negative for vshlq_s32 right-shift).
neon_wide_u32_kernel!(
    unpack_indices_into_neon_bw22,
    unpack_neon_bw22_unchecked,
    22,
    [0, 2, 5, 8, 11, 13, 16, 19],
    [0, -6, -4, -2, 0, -6, -4, -2],
    0x3F_FFFF,
    22
);
neon_wide_u32_kernel!(
    unpack_indices_into_neon_bw23,
    unpack_neon_bw23_unchecked,
    23,
    [0, 2, 5, 8, 11, 14, 17, 20],
    [0, -7, -6, -5, -4, -3, -2, -1],
    0x7F_FFFF,
    23
);
neon_wide_u32_kernel!(
    unpack_indices_into_neon_bw24,
    unpack_neon_bw24_unchecked,
    24,
    [0, 3, 6, 9, 12, 15, 18, 21],
    [0, 0, 0, 0, 0, 0, 0, 0],
    0xFF_FFFF,
    24
);
neon_wide_u32_kernel!(
    unpack_indices_into_neon_bw25,
    unpack_neon_bw25_unchecked,
    25,
    [0, 3, 6, 9, 12, 15, 18, 21],
    [0, -1, -2, -3, -4, -5, -6, -7],
    0x1FF_FFFF,
    25
);

// ---- bw=26..31: u64 staging ----------------------------------------
//
// A 26-bit value at bit_off=7 spans 5 source bytes (33 bits). u32 can't
// hold it; we gather 8 bytes per lane into a u64 staging buffer, do
// the shift + mask in 64-bit lanes, then narrow back to u32 on store.

macro_rules! neon_wide_u64_kernel {
    ($pubfn:ident, $unchecked:ident, $bw:literal, $offsets:expr, $shifts:expr, $mask:literal, $block_bytes:literal) => {
        pub fn $pubfn(packed: &[u8], num_values: usize, out: &mut Vec<u32>) -> Result<()> {
            if num_values == 0 {
                return Ok(());
            }
            let required_bytes = (num_values * $bw).div_ceil(8);
            if packed.len() < required_bytes {
                return Err(CodecError::Decompress(format!(
                    concat!("neon bw", stringify!($bw), ": packed has {} bytes, need {}"),
                    packed.len(),
                    required_bytes
                )));
            }
            out.reserve(num_values);
            let full_blocks = num_values / 8;
            let offsets: [usize; 8] = $offsets;
            let max_read_per_block = offsets[7] + 8;
            let safe_full_blocks = if full_blocks == 0 {
                0
            } else if packed.len() >= $block_bytes * (full_blocks - 1) + max_read_per_block {
                full_blocks
            } else {
                full_blocks - 1
            };
            unsafe {
                $unchecked(packed, safe_full_blocks, out);
            }
            let processed = safe_full_blocks * 8;
            let remaining = num_values - processed;
            if remaining > 0 {
                scalar_bw_n(&packed[processed * $bw / 8..], remaining, $bw, out);
            }
            Ok(())
        }

        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn $unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
            use std::arch::aarch64::*;
            let offsets: [usize; 8] = $offsets;
            let shifts: [i64; 8] = $shifts;
            let mask_v: uint64x2_t = vdupq_n_u64($mask as u64);
            let shifts_0: int64x2_t = vld1q_s64(shifts[0..2].as_ptr());
            let shifts_1: int64x2_t = vld1q_s64(shifts[2..4].as_ptr());
            let shifts_2: int64x2_t = vld1q_s64(shifts[4..6].as_ptr());
            let shifts_3: int64x2_t = vld1q_s64(shifts[6..8].as_ptr());

            let mut src_ptr = packed.as_ptr();
            let out_start_len = out.len();
            let out_ptr = out.as_mut_ptr().add(out_start_len);

            for blk in 0..full_blocks {
                let region = std::slice::from_raw_parts(src_ptr, offsets[7] + 8);
                let mut staging = [0u64; 8];
                for lane in 0..8 {
                    staging[lane] = read_u64_le_at(region, offsets[lane]);
                }
                let v0: uint64x2_t = vld1q_u64(staging[0..2].as_ptr());
                let v1: uint64x2_t = vld1q_u64(staging[2..4].as_ptr());
                let v2: uint64x2_t = vld1q_u64(staging[4..6].as_ptr());
                let v3: uint64x2_t = vld1q_u64(staging[6..8].as_ptr());
                let s0 = vreinterpretq_u64_s64(vshlq_s64(vreinterpretq_s64_u64(v0), shifts_0));
                let s1 = vreinterpretq_u64_s64(vshlq_s64(vreinterpretq_s64_u64(v1), shifts_1));
                let s2 = vreinterpretq_u64_s64(vshlq_s64(vreinterpretq_s64_u64(v2), shifts_2));
                let s3 = vreinterpretq_u64_s64(vshlq_s64(vreinterpretq_s64_u64(v3), shifts_3));
                let m0 = vandq_u64(s0, mask_v);
                let m1 = vandq_u64(s1, mask_v);
                let m2 = vandq_u64(s2, mask_v);
                let m3 = vandq_u64(s3, mask_v);
                // Narrow each pair of u64 lanes to u32 and write 8 outputs.
                vst1q_u64(staging[0..2].as_mut_ptr(), m0);
                vst1q_u64(staging[2..4].as_mut_ptr(), m1);
                vst1q_u64(staging[4..6].as_mut_ptr(), m2);
                vst1q_u64(staging[6..8].as_mut_ptr(), m3);
                for lane in 0..8 {
                    *out_ptr.add(blk * 8 + lane) = staging[lane] as u32;
                }
                src_ptr = src_ptr.add($block_bytes);
            }
            out.set_len(out_start_len + full_blocks * 8);
        }
    };
}

neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw26,
    unpack_neon_bw26_unchecked,
    26,
    [0, 3, 6, 9, 13, 16, 19, 22],
    [0, -2, -4, -6, 0, -2, -4, -6],
    0x3FF_FFFF,
    26
);
neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw27,
    unpack_neon_bw27_unchecked,
    27,
    [0, 3, 6, 10, 13, 16, 20, 23],
    [0, -3, -6, -1, -4, -7, -2, -5],
    0x7FF_FFFF,
    27
);
neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw28,
    unpack_neon_bw28_unchecked,
    28,
    [0, 3, 7, 10, 14, 17, 21, 24],
    [0, -4, 0, -4, 0, -4, 0, -4],
    0xFFF_FFFF,
    28
);
neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw29,
    unpack_neon_bw29_unchecked,
    29,
    [0, 3, 7, 10, 14, 18, 21, 25],
    [0, -5, -2, -7, -4, -1, -6, -3],
    0x1FFF_FFFF,
    29
);
neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw30,
    unpack_neon_bw30_unchecked,
    30,
    [0, 3, 7, 11, 15, 18, 22, 26],
    [0, -6, -4, -2, 0, -6, -4, -2],
    0x3FFF_FFFF,
    30
);
neon_wide_u64_kernel!(
    unpack_indices_into_neon_bw31,
    unpack_neon_bw31_unchecked,
    31,
    [0, 3, 7, 11, 15, 19, 23, 27],
    [0, -7, -6, -5, -4, -3, -2, -1],
    0x7FFF_FFFF,
    31
);

// ---- bw=32: byte-aligned trivial copy ------------------------------

pub fn unpack_indices_into_neon_bw32(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    if num_values == 0 {
        return Ok(());
    }
    let required_bytes = num_values * 4;
    if packed.len() < required_bytes {
        return Err(CodecError::Decompress(format!(
            "neon bw32: packed has {} bytes, need {required_bytes}",
            packed.len()
        )));
    }
    out.reserve(num_values);
    let full_blocks = num_values / 8;
    // Each block reads exactly 32 bytes, no over-read.
    unsafe {
        unpack_neon_bw32_unchecked(packed, full_blocks, out);
    }
    let processed = full_blocks * 8;
    let remaining = num_values - processed;
    if remaining > 0 {
        scalar_bw_n(&packed[processed * 4..], remaining, 32, out);
    }
    Ok(())
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_neon_bw32_unchecked(packed: &[u8], full_blocks: usize, out: &mut Vec<u32>) {
    use std::arch::aarch64::*;
    let mut src_ptr = packed.as_ptr();
    let out_start_len = out.len();
    let out_ptr = out.as_mut_ptr().add(out_start_len);
    for blk in 0..full_blocks {
        // Two 16-byte loads = 8 u32 values, byte-aligned.
        let v0 = vld1q_u32(src_ptr as *const u32);
        let v1 = vld1q_u32(src_ptr.add(16) as *const u32);
        vst1q_u32(out_ptr.add(blk * 8), v0);
        vst1q_u32(out_ptr.add(blk * 8 + 4), v1);
        src_ptr = src_ptr.add(32);
    }
    out.set_len(out_start_len + full_blocks * 8);
}
