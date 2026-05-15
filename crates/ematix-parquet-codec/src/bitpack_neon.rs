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
    let shuffle: uint8x16_t = vld1q_u8(
        [0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr(),
    );
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
        let shifted: uint16x8_t = vreinterpretq_u16_s16(vshlq_s16(
            vreinterpretq_s16_u16(as_u16),
            shifts,
        ));
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
    let shuffle: uint8x16_t = vld1q_u8(
        [0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr(),
    );
    let shifts: int16x8_t = vld1q_s16([0i16, -4, 0, -4, 0, -4, 0, -4].as_ptr());
    let mask: uint16x8_t = vdupq_n_u16(0x0FFF);
    let mut src_ptr = packed.as_ptr();
    let staging_ptr = staging.as_mut_ptr();

    for _ in 0..full_blocks {
        let bytes16 = vld1q_u8(src_ptr);
        let shuffled = vqtbl1q_u8(bytes16, shuffle);
        let as_u16 = vreinterpretq_u16_u8(shuffled);
        let shifted = vreinterpretq_u16_s16(vshlq_s16(
            vreinterpretq_s16_u16(as_u16),
            shifts,
        ));
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

    let shuffle: uint8x16_t = vld1q_u8(
        [0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11].as_ptr(),
    );
    let shifts: int16x8_t = vld1q_s16([0i16, -4, 0, -4, 0, -4, 0, -4].as_ptr());
    let idx_mask: uint16x8_t = vdupq_n_u16(0x0FFF);

    let mut src_ptr = packed.as_ptr();
    let mask_ptr = dict_mask.as_ptr();

    for blk in 0..full_blocks {
        let bytes16 = vld1q_u8(src_ptr);
        let shuffled = vqtbl1q_u8(bytes16, shuffle);
        let as_u16 = vreinterpretq_u16_u8(shuffled);
        let shifted = vreinterpretq_u16_s16(vshlq_s16(
            vreinterpretq_s16_u16(as_u16),
            shifts,
        ));
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
        let byte = b0
            | (b1 << 1)
            | (b2 << 2)
            | (b3 << 3)
            | (b4 << 4)
            | (b5 << 5)
            | (b6 << 6)
            | (b7 << 7);
        *bitmap_out.get_unchecked_mut(blk) = byte;

        src_ptr = src_ptr.add(12);
    }
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

    let shuffle: uint8x16_t = vld1q_u8(
        [0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr(),
    );
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
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(lo),
            shifts_lo,
        ));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(hi),
            shifts_hi,
        ));
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
    let shuffle: uint8x16_t = vld1q_u8(
        [0u8, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9].as_ptr(),
    );
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
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(lo),
            shifts_lo,
        ));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(hi),
            shifts_hi,
        ));
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
    let shuffle_lo: uint8x16_t = vld1q_u8(
        [0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr(),
    );
    // Lanes 4-7: u32 windows starting at bytes 7/8/10/12.
    let shuffle_hi: uint8x16_t = vld1q_u8(
        [7u8, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15].as_ptr(),
    );
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
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(lo),
            shifts,
        ));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(hi),
            shifts,
        ));
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
    let shuffle_lo: uint8x16_t = vld1q_u8(
        [0u8, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8].as_ptr(),
    );
    let shuffle_hi: uint8x16_t = vld1q_u8(
        [7u8, 8, 9, 10, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15].as_ptr(),
    );
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
        let lo_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(lo),
            shifts,
        ));
        let hi_shifted = vreinterpretq_u32_s32(vshlq_s32(
            vreinterpretq_s32_u32(hi),
            shifts,
        ));
        let lo_masked = vandq_u32(lo_shifted, mask);
        let hi_masked = vandq_u32(hi_shifted, mask);
        vst1q_u32(staging_ptr, lo_masked);
        vst1q_u32(staging_ptr.add(4), hi_masked);
        sink(staging)?;
        src_ptr = src_ptr.add(14);
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
