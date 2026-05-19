//! Const-generic per-bit-width bit unpacker.
//!
//! Inspired by polars-parquet's `unpack32<const NUM_BITS>` pattern.
//! For each runtime `bit_width`, a dispatch picks one of 33
//! monomorphized variants (`NUM_BITS` ∈ [0, 32]). Inside each variant
//! the offsets are compile-time constants, so the inner loop is a
//! straight-line sequence of loads/shifts/masks that LLVM can fully
//! unroll and auto-vectorize.
//!
//! Polars's own numbers for this pattern over a runtime-bit-width
//! scalar loop: ~4.5× from unrolling + ~2× from the jumptable
//! dispatch. We expect similar wins; the streaming bit-buffer in
//! `dict.rs` is the comparison baseline.
//!
//! Public surface:
//!   - `unpack_lookup_into<T>(packed, num_values, bit_width, dict, out)`
//!     fused bit-unpack + dict lookup, writes `dict[idx]` per value
//!     into `out`. The Q14 path uses this with `dict: &[bool]` to
//!     produce a row mask directly.
//!
//! Tail handling: bit-packed pages emit values in multiples of 8;
//! we process 32 at a time and fall back to a per-value path for the
//! trailing 0..31 values.

use crate::error::{CodecError, Result};

/// Decode `num_values` indices of `bit_width` bits each into looked-up
/// dict values, appending to `out`.
///
/// Generic over the dict's value type `T`. The hot loop is unrolled
/// per-bit-width via `dispatch_unpack!`; the dict lookup happens
/// after each 32-value batch.
pub fn unpack_lookup_into<T: Copy>(
    packed: &[u8],
    num_values: usize,
    bit_width: u8,
    dict: &[T],
    out: &mut Vec<T>,
) -> Result<()> {
    if bit_width > 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(());
    }
    if bit_width == 0 {
        if dict.is_empty() {
            return Err(CodecError::DictIndexOutOfRange {
                index: 0,
                dict_size: 0,
            });
        }
        let v = dict[0];
        out.reserve(num_values);
        for _ in 0..num_values {
            out.push(v);
        }
        return Ok(());
    }

    out.reserve(num_values);

    // NEON specializations for hot widths in TPC-H lineitem.
    // bw=12: shipdate / commitdate / receiptdate.
    // bw=14: l_suppkey 100%.
    // bw=15/16/18: tail of l_orderkey / l_partkey / l_extendedprice
    //              / l_comment dict pages.
    // bw=17: l_orderkey 51% / l_partkey 67% / l_extendedprice 43%.
    //
    // NEON unpack into a stack staging buffer, then scalar dict
    // gather per lane (raw pointer writes, no per-element bounds
    // or capacity checks).
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    {
        match bit_width {
            12 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw12(
                    packed, num_values, dict, out,
                );
            }
            14 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw14(
                    packed, num_values, dict, out,
                );
            }
            15 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw15(
                    packed, num_values, dict, out,
                );
            }
            16 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw16(
                    packed, num_values, dict, out,
                );
            }
            17 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw17(
                    packed, num_values, dict, out,
                );
            }
            18 => {
                return crate::bitpack_neon::unpack_lookup_into_neon_bw18(
                    packed, num_values, dict, out,
                );
            }
            _ => {}
        }
    }

    // AVX2 specializations on x86_64. Runtime feature detection
    // because pre-Haswell Intel + pre-Excavator AMD don't have AVX2;
    // the dispatcher silently falls through to scalar on those.
    // Π.12 ships bw=16 first (byte-aligned, simplest); future
    // Π.12b–f add 12 / 14 / 15 / 17 / 18.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            match bit_width {
                12 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw12(
                        packed, num_values, dict, out,
                    );
                }
                14 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw14(
                        packed, num_values, dict, out,
                    );
                }
                15 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw15(
                        packed, num_values, dict, out,
                    );
                }
                16 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw16(
                        packed, num_values, dict, out,
                    );
                }
                17 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw17(
                        packed, num_values, dict, out,
                    );
                }
                18 => {
                    return crate::bitpack_avx2::unpack_lookup_into_avx2_bw18(
                        packed, num_values, dict, out,
                    );
                }
                _ => {}
            }
        }
    }

    // Dispatch once on bit_width; the chosen monomorphization drives
    // the whole page.
    dispatch_unpack!(bit_width, packed, num_values, dict, out)
}

/// Same shape as `unpack_lookup_into` but emits raw u32 indices
/// instead of looked-up values. Useful for the `DictColumnChunk`
/// construction path where we want to keep the indices in segment
/// form.
pub fn unpack_indices_into(
    packed: &[u8],
    num_values: usize,
    bit_width: u8,
    out: &mut Vec<u32>,
) -> Result<()> {
    if bit_width > 32 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(());
    }
    if bit_width == 0 {
        out.resize(out.len() + num_values, 0);
        return Ok(());
    }
    out.reserve(num_values);

    // NEON specializations on M-series: every kernel hits the
    // ~78 GB/s output ceiling. Widths covered: bw=12 (date columns),
    // bw=14 (l_suppkey), bw=15 / bw=16 / bw=18 (l_orderkey /
    // l_partkey / l_extendedprice / l_comment tail), bw=17
    // (l_extendedprice 43% / l_partkey 67% / l_orderkey 51%).
    // Other widths fall through to scalar.
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    {
        match bit_width {
            1 => return crate::bitpack_neon::unpack_indices_into_neon_bw1(packed, num_values, out),
            4 => return crate::bitpack_neon::unpack_indices_into_neon_bw4(packed, num_values, out),
            5 => return crate::bitpack_neon::unpack_indices_into_neon_bw5(packed, num_values, out),
            8 => return crate::bitpack_neon::unpack_indices_into_neon_bw8(packed, num_values, out),
            12 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw12(packed, num_values, out)
            }
            14 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw14(packed, num_values, out)
            }
            15 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw15(packed, num_values, out)
            }
            16 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw16(packed, num_values, out)
            }
            17 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw17(packed, num_values, out)
            }
            18 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw18(packed, num_values, out)
            }
            20 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw20(packed, num_values, out)
            }
            21 => {
                return crate::bitpack_neon::unpack_indices_into_neon_bw21(packed, num_values, out)
            }
            _ => {}
        }
    }

    // AVX2 specializations on x86_64. Mirror of the lookup path
    // above; bw=1, 4, 5, 8, 12, 14, 15, 16, 17, 18, 20, 21 routed
    // to AVX2.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            match bit_width {
                1 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw1(
                        packed, num_values, out,
                    )
                }
                4 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw4(
                        packed, num_values, out,
                    )
                }
                5 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw5(
                        packed, num_values, out,
                    )
                }
                8 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw8(
                        packed, num_values, out,
                    )
                }
                12 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw12(
                        packed, num_values, out,
                    )
                }
                14 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw14(
                        packed, num_values, out,
                    )
                }
                15 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw15(
                        packed, num_values, out,
                    )
                }
                16 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw16(
                        packed, num_values, out,
                    )
                }
                17 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw17(
                        packed, num_values, out,
                    )
                }
                18 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw18(
                        packed, num_values, out,
                    )
                }
                20 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw20(
                        packed, num_values, out,
                    )
                }
                21 => {
                    return crate::bitpack_avx2::unpack_indices_into_avx2_bw21(
                        packed, num_values, out,
                    )
                }
                _ => {}
            }
        }
    }

    dispatch_unpack_indices!(bit_width, packed, num_values, out)
}

/// Monomorphized worker: unpacks `num_values` of `NUM_BITS` bits each,
/// looks each one up in `dict`, appends results to `out`.
///
/// Processes 32 values at a time. For the trailing <32 values uses
/// the same scalar logic but with NUM_BITS still const-known.
///
/// Hot-path uses raw pointer writes (capacity reserved by the
/// caller; per-value bounds-check on the dict is the only branch).
#[inline(always)]
fn unpack_chunks<const NUM_BITS: usize, T: Copy>(
    packed: &[u8],
    num_values: usize,
    dict: &[T],
    out: &mut Vec<T>,
) -> Result<()> {
    // Bit_width 0 is handled at the dispatch site; NUM_BITS in
    // [1, 32] here.
    debug_assert!(NUM_BITS >= 1 && NUM_BITS <= 32);
    let mask: u64 = if NUM_BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << NUM_BITS) - 1
    };
    let dict_size = dict.len();
    let bounds_safe = (mask as usize) < dict_size;

    let chunk_bytes = (NUM_BITS * 32) / 8;
    let full_chunks = num_values / 32;
    let tail = num_values % 32;

    let dict_ptr = dict.as_ptr();
    let out_start_len = out.len();

    let mut packed_idx = 0usize;
    let mut written = 0usize;
    // SAFETY: caller reserved `num_values` capacity ahead of this
    // function; out_ptr.add(written + i) is in-bounds for all writes
    // because total writes never exceed num_values.
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);

        if bounds_safe {
            for _ in 0..full_chunks {
                let chunk = &packed[packed_idx..packed_idx + chunk_bytes];
                for i in 0..32usize {
                    let start_bit = i * NUM_BITS;
                    let start_byte = start_bit / 8;
                    let bit_in_byte = (start_bit % 8) as u32;
                    let bytes_needed = (NUM_BITS + bit_in_byte as usize).div_ceil(8);
                    let mut acc: u64 = 0;
                    for j in 0..bytes_needed {
                        acc |= (chunk[start_byte + j] as u64) << (j * 8);
                    }
                    let idx = ((acc >> bit_in_byte) & mask) as usize;
                    *out_ptr.add(written + i) = *dict_ptr.add(idx);
                }
                written += 32;
                packed_idx += chunk_bytes;
            }
        } else {
            for _ in 0..full_chunks {
                let chunk = &packed[packed_idx..packed_idx + chunk_bytes];
                for i in 0..32usize {
                    let start_bit = i * NUM_BITS;
                    let start_byte = start_bit / 8;
                    let bit_in_byte = (start_bit % 8) as u32;
                    let bytes_needed = (NUM_BITS + bit_in_byte as usize).div_ceil(8);
                    let mut acc: u64 = 0;
                    for j in 0..bytes_needed {
                        acc |= (chunk[start_byte + j] as u64) << (j * 8);
                    }
                    let idx = ((acc >> bit_in_byte) & mask) as usize;
                    if idx >= dict_size {
                        // Commit what we've written so far so that
                        // any partial state doesn't leak — though
                        // in practice the caller drops `out` on err.
                        out.set_len(out_start_len + written);
                        return Err(CodecError::DictIndexOutOfRange {
                            index: idx as u32,
                            dict_size,
                        });
                    }
                    *out_ptr.add(written + i) = *dict_ptr.add(idx);
                }
                written += 32;
                packed_idx += chunk_bytes;
            }
        }
        out.set_len(out_start_len + written);
    }

    // Tail: 0..32 values, scalar-streaming bit buffer.
    if tail > 0 {
        let tail_bytes = (tail * NUM_BITS).div_ceil(8);
        let chunk = &packed[packed_idx..packed_idx + tail_bytes];
        let mut buf: u64 = 0;
        let mut bits: u32 = 0;
        let mut byte_idx = 0usize;
        for _ in 0..tail {
            while bits < NUM_BITS as u32 {
                buf |= (chunk[byte_idx] as u64) << bits;
                byte_idx += 1;
                bits += 8;
            }
            let idx = (buf & mask) as usize;
            buf >>= NUM_BITS;
            bits -= NUM_BITS as u32;
            if !bounds_safe && idx >= dict_size {
                return Err(CodecError::DictIndexOutOfRange {
                    index: idx as u32,
                    dict_size,
                });
            }
            // SAFETY: tail is bounded by 31 and capacity was reserved.
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = *dict_ptr.add(idx);
                out.set_len(out.len() + 1);
            }
        }
    }

    Ok(())
}

/// Same as `unpack_chunks` but emits raw u32 indices.
#[inline(always)]
fn unpack_chunks_indices<const NUM_BITS: usize>(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    debug_assert!(NUM_BITS >= 1 && NUM_BITS <= 32);
    let mask: u64 = if NUM_BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << NUM_BITS) - 1
    };

    let chunk_bytes = (NUM_BITS * 32) / 8;
    let full_chunks = num_values / 32;
    let tail = num_values % 32;

    let out_start_len = out.len();
    let mut packed_idx = 0usize;
    let mut written = 0usize;
    // SAFETY: caller reserved `num_values` capacity; raw pointer
    // writes never exceed that bound because total writes equal
    // num_values exactly.
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        for _ in 0..full_chunks {
            let chunk = &packed[packed_idx..packed_idx + chunk_bytes];
            for i in 0..32usize {
                let start_bit = i * NUM_BITS;
                let start_byte = start_bit / 8;
                let bit_in_byte = (start_bit % 8) as u32;
                let bytes_needed = (NUM_BITS + bit_in_byte as usize).div_ceil(8);
                let mut acc: u64 = 0;
                for j in 0..bytes_needed {
                    acc |= (chunk[start_byte + j] as u64) << (j * 8);
                }
                *out_ptr.add(written + i) = ((acc >> bit_in_byte) & mask) as u32;
            }
            written += 32;
            packed_idx += chunk_bytes;
        }
        out.set_len(out_start_len + written);
    }

    if tail > 0 {
        let tail_bytes = (tail * NUM_BITS).div_ceil(8);
        let chunk = &packed[packed_idx..packed_idx + tail_bytes];
        let mut buf: u64 = 0;
        let mut bits: u32 = 0;
        let mut byte_idx = 0usize;
        for _ in 0..tail {
            while bits < NUM_BITS as u32 {
                buf |= (chunk[byte_idx] as u64) << bits;
                byte_idx += 1;
                bits += 8;
            }
            // SAFETY: capacity reserved; tail < 32.
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = (buf & mask) as u32;
                out.set_len(out.len() + 1);
            }
            buf >>= NUM_BITS;
            bits -= NUM_BITS as u32;
        }
    }
    Ok(())
}

// 33-arm dispatch: maps runtime bit_width to a const-generic
// monomorphization. The match is per-page (cheap), the call inside
// is a fully-specialized function that handles all 1M values.
macro_rules! dispatch_unpack {
    ($bw:expr, $packed:expr, $n:expr, $dict:expr, $out:expr) => {
        match $bw {
            1 => unpack_chunks::<1, _>($packed, $n, $dict, $out),
            2 => unpack_chunks::<2, _>($packed, $n, $dict, $out),
            3 => unpack_chunks::<3, _>($packed, $n, $dict, $out),
            4 => unpack_chunks::<4, _>($packed, $n, $dict, $out),
            5 => unpack_chunks::<5, _>($packed, $n, $dict, $out),
            6 => unpack_chunks::<6, _>($packed, $n, $dict, $out),
            7 => unpack_chunks::<7, _>($packed, $n, $dict, $out),
            8 => unpack_chunks::<8, _>($packed, $n, $dict, $out),
            9 => unpack_chunks::<9, _>($packed, $n, $dict, $out),
            10 => unpack_chunks::<10, _>($packed, $n, $dict, $out),
            11 => unpack_chunks::<11, _>($packed, $n, $dict, $out),
            12 => unpack_chunks::<12, _>($packed, $n, $dict, $out),
            13 => unpack_chunks::<13, _>($packed, $n, $dict, $out),
            14 => unpack_chunks::<14, _>($packed, $n, $dict, $out),
            15 => unpack_chunks::<15, _>($packed, $n, $dict, $out),
            16 => unpack_chunks::<16, _>($packed, $n, $dict, $out),
            17 => unpack_chunks::<17, _>($packed, $n, $dict, $out),
            18 => unpack_chunks::<18, _>($packed, $n, $dict, $out),
            19 => unpack_chunks::<19, _>($packed, $n, $dict, $out),
            20 => unpack_chunks::<20, _>($packed, $n, $dict, $out),
            21 => unpack_chunks::<21, _>($packed, $n, $dict, $out),
            22 => unpack_chunks::<22, _>($packed, $n, $dict, $out),
            23 => unpack_chunks::<23, _>($packed, $n, $dict, $out),
            24 => unpack_chunks::<24, _>($packed, $n, $dict, $out),
            25 => unpack_chunks::<25, _>($packed, $n, $dict, $out),
            26 => unpack_chunks::<26, _>($packed, $n, $dict, $out),
            27 => unpack_chunks::<27, _>($packed, $n, $dict, $out),
            28 => unpack_chunks::<28, _>($packed, $n, $dict, $out),
            29 => unpack_chunks::<29, _>($packed, $n, $dict, $out),
            30 => unpack_chunks::<30, _>($packed, $n, $dict, $out),
            31 => unpack_chunks::<31, _>($packed, $n, $dict, $out),
            32 => unpack_chunks::<32, _>($packed, $n, $dict, $out),
            other => Err($crate::error::CodecError::BitWidthOutOfRange(other)),
        }
    };
}
use dispatch_unpack;

macro_rules! dispatch_unpack_indices {
    ($bw:expr, $packed:expr, $n:expr, $out:expr) => {
        match $bw {
            1 => unpack_chunks_indices::<1>($packed, $n, $out),
            2 => unpack_chunks_indices::<2>($packed, $n, $out),
            3 => unpack_chunks_indices::<3>($packed, $n, $out),
            4 => unpack_chunks_indices::<4>($packed, $n, $out),
            5 => unpack_chunks_indices::<5>($packed, $n, $out),
            6 => unpack_chunks_indices::<6>($packed, $n, $out),
            7 => unpack_chunks_indices::<7>($packed, $n, $out),
            8 => unpack_chunks_indices::<8>($packed, $n, $out),
            9 => unpack_chunks_indices::<9>($packed, $n, $out),
            10 => unpack_chunks_indices::<10>($packed, $n, $out),
            11 => unpack_chunks_indices::<11>($packed, $n, $out),
            12 => unpack_chunks_indices::<12>($packed, $n, $out),
            13 => unpack_chunks_indices::<13>($packed, $n, $out),
            14 => unpack_chunks_indices::<14>($packed, $n, $out),
            15 => unpack_chunks_indices::<15>($packed, $n, $out),
            16 => unpack_chunks_indices::<16>($packed, $n, $out),
            17 => unpack_chunks_indices::<17>($packed, $n, $out),
            18 => unpack_chunks_indices::<18>($packed, $n, $out),
            19 => unpack_chunks_indices::<19>($packed, $n, $out),
            20 => unpack_chunks_indices::<20>($packed, $n, $out),
            21 => unpack_chunks_indices::<21>($packed, $n, $out),
            22 => unpack_chunks_indices::<22>($packed, $n, $out),
            23 => unpack_chunks_indices::<23>($packed, $n, $out),
            24 => unpack_chunks_indices::<24>($packed, $n, $out),
            25 => unpack_chunks_indices::<25>($packed, $n, $out),
            26 => unpack_chunks_indices::<26>($packed, $n, $out),
            27 => unpack_chunks_indices::<27>($packed, $n, $out),
            28 => unpack_chunks_indices::<28>($packed, $n, $out),
            29 => unpack_chunks_indices::<29>($packed, $n, $out),
            30 => unpack_chunks_indices::<30>($packed, $n, $out),
            31 => unpack_chunks_indices::<31>($packed, $n, $out),
            32 => unpack_chunks_indices::<32>($packed, $n, $out),
            other => Err($crate::error::CodecError::BitWidthOutOfRange(other)),
        }
    };
}
use dispatch_unpack_indices;

// ---- u64-output unpacker (DELTA-i64 high-bit-width path) -----------
//
// Mirrors `unpack_indices_into` / `unpack_chunks_indices` but emits
// u64 values and supports bit_widths in 1..=64. The wide path uses a
// u128 accumulator because at NUM_BITS > 57 a single value can span
// 9 source bytes (start_bit % 8 of up to 7, plus NUM_BITS bits),
// which doesn't fit in u64. No SIMD specializations — DELTA-i64 with
// bit_width > 32 is rare in practice (writer normally falls back to
// PLAIN at that point), so scalar correctness wins over peak speed.

/// Public u64-output sibling of [`unpack_indices_into`].
///
/// Supports `bit_width` in 0..=64. Writes `num_values` u64 values
/// into `out`. Caller is responsible for reserving capacity; the
/// function does its own `reserve`.
pub fn unpack_indices64_into(
    packed: &[u8],
    num_values: usize,
    bit_width: u8,
    out: &mut Vec<u64>,
) -> Result<()> {
    if bit_width > 64 {
        return Err(CodecError::BitWidthOutOfRange(bit_width));
    }
    if num_values == 0 {
        return Ok(());
    }
    if bit_width == 0 {
        out.resize(out.len() + num_values, 0);
        return Ok(());
    }
    out.reserve(num_values);
    dispatch_unpack_indices64!(bit_width, packed, num_values, out)
}

/// Monomorphized worker for the u64-output path. Const-generic on
/// the bit width so each width compiles to its own straight-line
/// scalar kernel. Uses a u128 accumulator to safely handle the
/// >57-bit case where a single value can span 9 bytes.
#[inline(always)]
fn unpack_chunks_indices64<const NUM_BITS: usize>(
    packed: &[u8],
    num_values: usize,
    out: &mut Vec<u64>,
) -> Result<()> {
    debug_assert!(NUM_BITS >= 1 && NUM_BITS <= 64);
    let mask: u128 = if NUM_BITS == 64 {
        u64::MAX as u128
    } else {
        (1u128 << NUM_BITS) - 1
    };

    let chunk_bytes = (NUM_BITS * 32) / 8;
    let full_chunks = num_values / 32;
    let tail = num_values % 32;

    let out_start_len = out.len();
    let mut packed_idx = 0usize;
    let mut written = 0usize;
    // SAFETY: caller reserved `num_values` capacity; writes never
    // exceed it because the total equals num_values exactly.
    unsafe {
        let out_ptr = out.as_mut_ptr().add(out_start_len);
        for _ in 0..full_chunks {
            let chunk = &packed[packed_idx..packed_idx + chunk_bytes];
            for i in 0..32usize {
                let start_bit = i * NUM_BITS;
                let start_byte = start_bit / 8;
                let bit_in_byte = (start_bit % 8) as u32;
                let bytes_needed = (NUM_BITS + bit_in_byte as usize).div_ceil(8);
                let mut acc: u128 = 0;
                for j in 0..bytes_needed {
                    acc |= (chunk[start_byte + j] as u128) << (j * 8);
                }
                *out_ptr.add(written + i) = ((acc >> bit_in_byte) & mask) as u64;
            }
            written += 32;
            packed_idx += chunk_bytes;
        }
        out.set_len(out_start_len + written);
    }

    if tail > 0 {
        let tail_bytes = (tail * NUM_BITS).div_ceil(8);
        let chunk = &packed[packed_idx..packed_idx + tail_bytes];
        let mut buf: u128 = 0;
        let mut bits: u32 = 0;
        let mut byte_idx = 0usize;
        for _ in 0..tail {
            while bits < NUM_BITS as u32 {
                buf |= (chunk[byte_idx] as u128) << bits;
                byte_idx += 1;
                bits += 8;
            }
            // SAFETY: capacity reserved; tail < 32.
            unsafe {
                let out_ptr = out.as_mut_ptr().add(out.len());
                *out_ptr = (buf & mask) as u64;
                out.set_len(out.len() + 1);
            }
            buf >>= NUM_BITS;
            bits -= NUM_BITS as u32;
        }
    }
    Ok(())
}

macro_rules! dispatch_unpack_indices64 {
    ($bw:expr, $packed:expr, $n:expr, $out:expr) => {
        match $bw {
            1 => unpack_chunks_indices64::<1>($packed, $n, $out),
            2 => unpack_chunks_indices64::<2>($packed, $n, $out),
            3 => unpack_chunks_indices64::<3>($packed, $n, $out),
            4 => unpack_chunks_indices64::<4>($packed, $n, $out),
            5 => unpack_chunks_indices64::<5>($packed, $n, $out),
            6 => unpack_chunks_indices64::<6>($packed, $n, $out),
            7 => unpack_chunks_indices64::<7>($packed, $n, $out),
            8 => unpack_chunks_indices64::<8>($packed, $n, $out),
            9 => unpack_chunks_indices64::<9>($packed, $n, $out),
            10 => unpack_chunks_indices64::<10>($packed, $n, $out),
            11 => unpack_chunks_indices64::<11>($packed, $n, $out),
            12 => unpack_chunks_indices64::<12>($packed, $n, $out),
            13 => unpack_chunks_indices64::<13>($packed, $n, $out),
            14 => unpack_chunks_indices64::<14>($packed, $n, $out),
            15 => unpack_chunks_indices64::<15>($packed, $n, $out),
            16 => unpack_chunks_indices64::<16>($packed, $n, $out),
            17 => unpack_chunks_indices64::<17>($packed, $n, $out),
            18 => unpack_chunks_indices64::<18>($packed, $n, $out),
            19 => unpack_chunks_indices64::<19>($packed, $n, $out),
            20 => unpack_chunks_indices64::<20>($packed, $n, $out),
            21 => unpack_chunks_indices64::<21>($packed, $n, $out),
            22 => unpack_chunks_indices64::<22>($packed, $n, $out),
            23 => unpack_chunks_indices64::<23>($packed, $n, $out),
            24 => unpack_chunks_indices64::<24>($packed, $n, $out),
            25 => unpack_chunks_indices64::<25>($packed, $n, $out),
            26 => unpack_chunks_indices64::<26>($packed, $n, $out),
            27 => unpack_chunks_indices64::<27>($packed, $n, $out),
            28 => unpack_chunks_indices64::<28>($packed, $n, $out),
            29 => unpack_chunks_indices64::<29>($packed, $n, $out),
            30 => unpack_chunks_indices64::<30>($packed, $n, $out),
            31 => unpack_chunks_indices64::<31>($packed, $n, $out),
            32 => unpack_chunks_indices64::<32>($packed, $n, $out),
            33 => unpack_chunks_indices64::<33>($packed, $n, $out),
            34 => unpack_chunks_indices64::<34>($packed, $n, $out),
            35 => unpack_chunks_indices64::<35>($packed, $n, $out),
            36 => unpack_chunks_indices64::<36>($packed, $n, $out),
            37 => unpack_chunks_indices64::<37>($packed, $n, $out),
            38 => unpack_chunks_indices64::<38>($packed, $n, $out),
            39 => unpack_chunks_indices64::<39>($packed, $n, $out),
            40 => unpack_chunks_indices64::<40>($packed, $n, $out),
            41 => unpack_chunks_indices64::<41>($packed, $n, $out),
            42 => unpack_chunks_indices64::<42>($packed, $n, $out),
            43 => unpack_chunks_indices64::<43>($packed, $n, $out),
            44 => unpack_chunks_indices64::<44>($packed, $n, $out),
            45 => unpack_chunks_indices64::<45>($packed, $n, $out),
            46 => unpack_chunks_indices64::<46>($packed, $n, $out),
            47 => unpack_chunks_indices64::<47>($packed, $n, $out),
            48 => unpack_chunks_indices64::<48>($packed, $n, $out),
            49 => unpack_chunks_indices64::<49>($packed, $n, $out),
            50 => unpack_chunks_indices64::<50>($packed, $n, $out),
            51 => unpack_chunks_indices64::<51>($packed, $n, $out),
            52 => unpack_chunks_indices64::<52>($packed, $n, $out),
            53 => unpack_chunks_indices64::<53>($packed, $n, $out),
            54 => unpack_chunks_indices64::<54>($packed, $n, $out),
            55 => unpack_chunks_indices64::<55>($packed, $n, $out),
            56 => unpack_chunks_indices64::<56>($packed, $n, $out),
            57 => unpack_chunks_indices64::<57>($packed, $n, $out),
            58 => unpack_chunks_indices64::<58>($packed, $n, $out),
            59 => unpack_chunks_indices64::<59>($packed, $n, $out),
            60 => unpack_chunks_indices64::<60>($packed, $n, $out),
            61 => unpack_chunks_indices64::<61>($packed, $n, $out),
            62 => unpack_chunks_indices64::<62>($packed, $n, $out),
            63 => unpack_chunks_indices64::<63>($packed, $n, $out),
            64 => unpack_chunks_indices64::<64>($packed, $n, $out),
            other => Err($crate::error::CodecError::BitWidthOutOfRange(other)),
        }
    };
}
use dispatch_unpack_indices64;
