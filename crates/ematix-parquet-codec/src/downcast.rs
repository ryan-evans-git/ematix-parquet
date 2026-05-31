//! REV.12 — integer downcast-on-read (foundation).
//!
//! When a physically-INT64 column's value range (from row-group / page
//! statistics) fits in a narrower integer width, decode it directly
//! into that narrower width. A narrower in-memory representation means
//! less decode bandwidth and a smaller cache footprint in the hash
//! tables / sort buffers the column feeds downstream — the lever DuckDB
//! gets from `__internal_compress_integral_uinteger`.
//!
//! This module is the FOUNDATION: the width DECISION plus the narrowing
//! PLAIN decoders. It is opt-in and additive — the existing
//! [`crate::plain::decode_plain_i64`] `-> Vec<i64>` path is untouched.
//! Wiring real row-group statistics into the decode orchestrator and a
//! public column-level entry point is a follow-on slice; frame-of-
//! reference (offset-from-min) narrowing for clustered-but-large ranges
//! is a further extension on top of [`narrowest_int_target`].
//!
//! ## Why keys are the prime target
//!
//! For group-by / join KEYS only equality (and, for sorts, ordering)
//! matters — never the arithmetic value. A width narrowing preserves
//! both, so the whole hash/probe can run on the narrow type and only the
//! small final output is re-widened. Value columns that get summed would
//! pay a re-widen per row; keys don't.
//!
//! ## Safety
//!
//! `f64 -> f32` is deliberately NOT offered here: it loses precision and
//! would break exact value-validation on aggregate sums. Downcast in this
//! module is integer-only and lossless — guarded by [`narrowest_int_target`],
//! which never returns a target that cannot hold every value in the range.

use crate::error::{CodecError, Result};
use crate::plain::decode_plain_i64;

/// The narrowest integer target that losslessly holds a value range.
///
/// Ordered conceptually narrowest-first by byte width (1 → 2 → 4 → 8).
/// Within a width, the signed variant is preferred; the unsigned variant
/// is chosen only when a non-negative minimum lets it cover a `max` the
/// signed variant cannot (e.g. `[0, 200]` → `U8`, since `200 > i8::MAX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntTarget {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    /// No narrowing — the range needs the full 64 bits.
    I64,
}

impl IntTarget {
    /// Bytes per value in the narrowed in-memory representation.
    pub fn width_bytes(self) -> usize {
        match self {
            IntTarget::I8 | IntTarget::U8 => 1,
            IntTarget::I16 | IntTarget::U16 => 2,
            IntTarget::I32 | IntTarget::U32 => 4,
            IntTarget::I64 => 8,
        }
    }

    /// True if this target narrows below the source's 8 bytes.
    pub fn is_narrowing(self) -> bool {
        self.width_bytes() < 8
    }
}

/// Pick the narrowest [`IntTarget`] that losslessly holds every value in
/// `[min, max]`. Caller supplies the column's true min/max (e.g. decoded
/// from row-group statistics). Decides by byte width first (1 → 2 → 4 →
/// 8); within a width prefers signed, falling to unsigned only when a
/// non-negative `min` lets it reach a `max` the signed variant can't.
///
/// The returned target is guaranteed to hold the whole range, so a
/// subsequent narrowing decode is lossless.
pub fn narrowest_int_target(min: i64, max: i64) -> IntTarget {
    debug_assert!(min <= max, "narrowest_int_target: min {min} > max {max}");
    // 1 byte
    if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
        return IntTarget::I8;
    }
    if min >= 0 && max <= u8::MAX as i64 {
        return IntTarget::U8;
    }
    // 2 bytes
    if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
        return IntTarget::I16;
    }
    if min >= 0 && max <= u16::MAX as i64 {
        return IntTarget::U16;
    }
    // 4 bytes
    if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
        return IntTarget::I32;
    }
    if min >= 0 && max <= u32::MAX as i64 {
        return IntTarget::U32;
    }
    // 8 bytes — no narrowing possible.
    IntTarget::I64
}

/// Result of a narrowing INT64 PLAIN decode. The variant matches the
/// [`IntTarget`] used; [`NarrowedI64::I64`] means no narrowing was applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NarrowedI64 {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
}

impl NarrowedI64 {
    pub fn len(&self) -> usize {
        match self {
            NarrowedI64::I8(v) => v.len(),
            NarrowedI64::U8(v) => v.len(),
            NarrowedI64::I16(v) => v.len(),
            NarrowedI64::U16(v) => v.len(),
            NarrowedI64::I32(v) => v.len(),
            NarrowedI64::U32(v) => v.len(),
            NarrowedI64::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The [`IntTarget`] this result was decoded to.
    pub fn target(&self) -> IntTarget {
        match self {
            NarrowedI64::I8(_) => IntTarget::I8,
            NarrowedI64::U8(_) => IntTarget::U8,
            NarrowedI64::I16(_) => IntTarget::I16,
            NarrowedI64::U16(_) => IntTarget::U16,
            NarrowedI64::I32(_) => IntTarget::I32,
            NarrowedI64::U32(_) => IntTarget::U32,
            NarrowedI64::I64(_) => IntTarget::I64,
        }
    }

    /// Heap bytes held by the value buffer. Lets callers confirm the
    /// footprint saving vs the i64 path (`len * 8`).
    pub fn byte_size(&self) -> usize {
        self.len() * self.target().width_bytes()
    }

    /// Re-widen every value back to i64. Used for verification and by
    /// consumers that still want i64 (the narrowing was lossless, so
    /// this round-trips the original values exactly).
    pub fn to_i64(&self) -> Vec<i64> {
        match self {
            NarrowedI64::I8(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::U8(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::I16(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::U16(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::I32(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::U32(v) => v.iter().map(|&x| x as i64).collect(),
            NarrowedI64::I64(v) => v.clone(),
        }
    }
}

/// Decode a PLAIN-encoded INT64 buffer, narrowing each value to `target`.
///
/// The buffer length must be an exact multiple of 8 (same wire contract
/// as [`decode_plain_i64`]). For [`IntTarget::I64`] this is exactly
/// `decode_plain_i64` (no narrowing). The caller is responsible for
/// having chosen `target` via [`narrowest_int_target`] from the column's
/// true range; narrowing to a target that does not hold every value would
/// truncate (debug-asserted per value, release-mode wraps via `as`).
pub fn decode_plain_i64_narrowed(bytes: &[u8], target: IntTarget) -> Result<NarrowedI64> {
    Ok(match target {
        IntTarget::I8 => NarrowedI64::I8(decode_plain_i64_as_i8(bytes)?),
        IntTarget::U8 => NarrowedI64::U8(decode_plain_i64_as_u8(bytes)?),
        IntTarget::I16 => NarrowedI64::I16(decode_plain_i64_as_i16(bytes)?),
        IntTarget::U16 => NarrowedI64::U16(decode_plain_i64_as_u16(bytes)?),
        IntTarget::I32 => NarrowedI64::I32(decode_plain_i64_as_i32(bytes)?),
        IntTarget::U32 => NarrowedI64::U32(decode_plain_i64_as_u32(bytes)?),
        IntTarget::I64 => NarrowedI64::I64(decode_plain_i64(bytes)?),
    })
}

// ---------------------------------------------------------------------
// Per-target narrowing PLAIN decoders. The caller must have proven the
// range fits (via `narrowest_int_target`) — narrowing is lossless and
// debug-asserted per value. These are the homogeneous `Fn(&[u8]) ->
// Result<Vec<T>>` decoders that `read::read_column_i64_downcast` hands to
// the generic chunk orchestrator (`decode_chunk_into`), so an INT64
// column narrows DURING decode (PLAIN + dictionary pages alike) with no
// transient `Vec<i64>` — no 2× memory peak on a 600M-row SF=100 column.
// ---------------------------------------------------------------------

pub fn decode_plain_i64_as_i8(bytes: &[u8]) -> Result<Vec<i8>> {
    narrow_decode(bytes, |v| {
        debug_assert!(
            v >= i8::MIN as i64 && v <= i8::MAX as i64,
            "i8 downcast lost {v}"
        );
        v as i8
    })
}

pub fn decode_plain_i64_as_u8(bytes: &[u8]) -> Result<Vec<u8>> {
    narrow_decode(bytes, |v| {
        debug_assert!(v >= 0 && v <= u8::MAX as i64, "u8 downcast lost {v}");
        v as u8
    })
}

pub fn decode_plain_i64_as_i16(bytes: &[u8]) -> Result<Vec<i16>> {
    narrow_decode(bytes, |v| {
        debug_assert!(
            v >= i16::MIN as i64 && v <= i16::MAX as i64,
            "i16 downcast lost {v}"
        );
        v as i16
    })
}

pub fn decode_plain_i64_as_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    narrow_decode(bytes, |v| {
        debug_assert!(v >= 0 && v <= u16::MAX as i64, "u16 downcast lost {v}");
        v as u16
    })
}

pub fn decode_plain_i64_as_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    narrow_decode(bytes, |v| {
        debug_assert!(
            v >= i32::MIN as i64 && v <= i32::MAX as i64,
            "i32 downcast lost {v}"
        );
        v as i32
    })
}

pub fn decode_plain_i64_as_u32(bytes: &[u8]) -> Result<Vec<u32>> {
    narrow_decode(bytes, |v| {
        debug_assert!(v >= 0 && v <= u32::MAX as i64, "u32 downcast lost {v}");
        v as u32
    })
}

/// Convenience: pick the narrowest target from `[min, max]` and decode in
/// one call. The common entry point once stats are in hand.
pub fn decode_plain_i64_auto(bytes: &[u8], min: i64, max: i64) -> Result<NarrowedI64> {
    decode_plain_i64_narrowed(bytes, narrowest_int_target(min, max))
}

// ---------------------------------------------------------------------
// Frame-of-reference (offset-from-min) narrowing.
//
// Absolute narrowing keys off `[min, max]`; a clustered-but-large column
// (timestamps, high-SF row-group-local keys) has a large absolute value
// yet a small SPAN. Frame-of-reference decodes `value - min` into the
// SPAN's width and carries `min`. Equality and ordering are preserved
// (the offset is a per-column constant), so a group/join/sort runs
// entirely on the narrow delta and only the small final output re-widens.
// This generalises and subsumes absolute narrowing (it equals it when
// `min == 0`), and reaches widths absolute narrowing cannot when the
// values are offset far from zero.
// ---------------------------------------------------------------------

/// A narrowing INT64 decode with a carried `offset` (= the column min).
/// `data` holds `value - offset` in the span's narrowest width;
/// [`FrameNarrowed::to_i64`] adds the offset back losslessly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameNarrowed {
    /// The constant subtracted from every value before narrowing (the
    /// column min). `0` when no frame narrowing was applied.
    pub offset: i64,
    /// The narrowed `value - offset` deltas.
    pub data: NarrowedI64,
}

impl FrameNarrowed {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The [`IntTarget`] the deltas were narrowed to.
    pub fn target(&self) -> IntTarget {
        self.data.target()
    }

    /// Heap bytes held by the delta buffer (`len * target width`).
    pub fn byte_size(&self) -> usize {
        self.data.byte_size()
    }

    /// Re-widen every value back to i64 (`delta + offset`). Lossless.
    pub fn to_i64(&self) -> Vec<i64> {
        let off = self.offset;
        self.data.to_i64().into_iter().map(|d| d + off).collect()
    }
}

/// Pick the frame-of-reference `(offset, target)` for value range
/// `[min, max]`: `offset = min` and `target` is the narrowest width that
/// holds the span `max - min` (always non-negative). If the span exceeds
/// `u32::MAX` (cannot narrow below 8 bytes), returns `(0, I64)` so a
/// subsequent decode is exactly the plain path.
pub fn frame_of_reference_target(min: i64, max: i64) -> (i64, IntTarget) {
    debug_assert!(
        min <= max,
        "frame_of_reference_target: min {min} > max {max}"
    );
    let span = (max as i128) - (min as i128); // >= 0
    if span <= u32::MAX as i128 {
        // span fits in [0, u32::MAX] -> narrowest_int_target picks U8..U32
        // (or I8/I16/I32 when the signed variant covers it). Always < 8B.
        (min, narrowest_int_target(0, span as i64))
    } else {
        (0, IntTarget::I64)
    }
}

/// Decode a PLAIN INT64 buffer as frame-of-reference: narrow each
/// `value - offset` to `target`. With `(0, I64)` this is exactly the
/// plain decode. Caller picks `(offset, target)` via
/// [`frame_of_reference_target`]; narrowing is lossless (debug-asserted
/// per value on the delta).
pub fn decode_plain_i64_frame(
    bytes: &[u8],
    offset: i64,
    target: IntTarget,
) -> Result<FrameNarrowed> {
    let data = match target {
        IntTarget::I8 => NarrowedI64::I8(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(
                d >= i8::MIN as i64 && d <= i8::MAX as i64,
                "i8 frame lost {v}"
            );
            d as i8
        })?),
        IntTarget::U8 => NarrowedI64::U8(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(d >= 0 && d <= u8::MAX as i64, "u8 frame lost {v}");
            d as u8
        })?),
        IntTarget::I16 => NarrowedI64::I16(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(
                d >= i16::MIN as i64 && d <= i16::MAX as i64,
                "i16 frame lost {v}"
            );
            d as i16
        })?),
        IntTarget::U16 => NarrowedI64::U16(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(d >= 0 && d <= u16::MAX as i64, "u16 frame lost {v}");
            d as u16
        })?),
        IntTarget::I32 => NarrowedI64::I32(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(
                d >= i32::MIN as i64 && d <= i32::MAX as i64,
                "i32 frame lost {v}"
            );
            d as i32
        })?),
        IntTarget::U32 => NarrowedI64::U32(narrow_decode(bytes, |v| {
            let d = v - offset;
            debug_assert!(d >= 0 && d <= u32::MAX as i64, "u32 frame lost {v}");
            d as u32
        })?),
        // No narrowing: offset is 0 here, so this equals decode_plain_i64.
        IntTarget::I64 => NarrowedI64::I64(decode_plain_i64(bytes)?),
    };
    Ok(FrameNarrowed { offset, data })
}

/// Convenience: pick the frame `(offset, target)` from `[min, max]` and
/// decode in one call.
pub fn decode_plain_i64_frame_auto(bytes: &[u8], min: i64, max: i64) -> Result<FrameNarrowed> {
    let (offset, target) = frame_of_reference_target(min, max);
    decode_plain_i64_frame(bytes, offset, target)
}

/// Shared narrowing loop: read each 8-byte LE i64 and map it through `f`.
/// Distinct from [`crate::plain`]'s `plain_memcpy` fast path because the
/// destination width differs from the source — a per-value cast is
/// required (no raw memcpy possible across widths).
#[inline]
fn narrow_decode<T, F>(bytes: &[u8], f: F) -> Result<Vec<T>>
where
    F: Fn(i64) -> T,
{
    if bytes.len() % 8 != 0 {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: 8,
            buffer_len: bytes.len(),
        });
    }
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        out.push(f(i64::from_le_bytes(chunk.try_into().unwrap())));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plain::decode_plain_i64;

    fn plain_bytes(vals: &[i64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    #[test]
    fn narrowest_target_picks_smallest_fitting_width() {
        // 1 byte
        assert_eq!(narrowest_int_target(1, 7), IntTarget::I8); // l_linenumber
        assert_eq!(narrowest_int_target(1, 50), IntTarget::I8); // l_quantity
        assert_eq!(narrowest_int_target(0, 24), IntTarget::I8); // nationkey
        assert_eq!(narrowest_int_target(-128, 127), IntTarget::I8);
        // exceeds i8 but non-negative & fits u8
        assert_eq!(narrowest_int_target(0, 200), IntTarget::U8);
        assert_eq!(narrowest_int_target(0, 255), IntTarget::U8);
        // 2 bytes
        assert_eq!(narrowest_int_target(-129, 127), IntTarget::I16);
        assert_eq!(narrowest_int_target(0, 32_767), IntTarget::I16);
        assert_eq!(narrowest_int_target(0, 40_000), IntTarget::U16);
        assert_eq!(narrowest_int_target(0, 65_535), IntTarget::U16);
        // 4 bytes — TPC-H SF=100 l_orderkey range (~600M fits i32)
        assert_eq!(narrowest_int_target(1, 600_000_000), IntTarget::I32);
        assert_eq!(narrowest_int_target(-1, i32::MAX as i64), IntTarget::I32);
        // exceeds i32 but non-negative & fits u32 (SF~300-700 headroom)
        assert_eq!(narrowest_int_target(0, i32::MAX as i64 + 1), IntTarget::U32);
        assert_eq!(narrowest_int_target(0, u32::MAX as i64), IntTarget::U32);
        // 8 bytes — no narrowing
        assert_eq!(narrowest_int_target(0, u32::MAX as i64 + 1), IntTarget::I64);
        assert_eq!(narrowest_int_target(i64::MIN, i64::MAX), IntTarget::I64);
        assert_eq!(narrowest_int_target(-1, i64::MAX), IntTarget::I64);
    }

    #[test]
    fn width_bytes_are_correct() {
        assert_eq!(IntTarget::I8.width_bytes(), 1);
        assert_eq!(IntTarget::U8.width_bytes(), 1);
        assert_eq!(IntTarget::I16.width_bytes(), 2);
        assert_eq!(IntTarget::U16.width_bytes(), 2);
        assert_eq!(IntTarget::I32.width_bytes(), 4);
        assert_eq!(IntTarget::U32.width_bytes(), 4);
        assert_eq!(IntTarget::I64.width_bytes(), 8);
        assert!(IntTarget::I32.is_narrowing());
        assert!(!IntTarget::I64.is_narrowing());
    }

    #[test]
    fn auto_decode_roundtrips_against_independent_i64_path() {
        // Cross-check: the narrowed decode, re-widened, must equal the
        // independent decode_plain_i64 path. A symmetric bug in both can't
        // pass because they share no code (one casts per value, the other
        // memcpys 8-byte words).
        let cases: Vec<Vec<i64>> = vec![
            vec![1, 2, 3, 4, 5, 6, 7],         // -> I8
            vec![0, 24, 13, 7, 1],             // -> I8
            vec![0, 200, 100, 255],            // -> U8
            vec![-100, 30_000, 0, -1],         // -> I16
            vec![1, 600_000_000, 250_000_000], // -> I32 (SF=100 orderkey-ish)
            vec![0, 3_000_000_000, 42],        // -> U32 (exceeds i32)
            vec![0, 10_000_000_000, 1],        // -> I64 (no narrowing)
        ];
        let expected_targets = [
            IntTarget::I8,
            IntTarget::I8,
            IntTarget::U8,
            IntTarget::I16,
            IntTarget::I32,
            IntTarget::U32,
            IntTarget::I64,
        ];
        for (vals, want_target) in cases.iter().zip(expected_targets) {
            let bytes = plain_bytes(vals);
            let min = *vals.iter().min().unwrap();
            let max = *vals.iter().max().unwrap();
            let narrowed = decode_plain_i64_auto(&bytes, min, max).unwrap();
            assert_eq!(narrowed.target(), want_target, "wrong target for {vals:?}");
            // Re-widened narrowed values == independent i64 decode.
            let ground_truth = decode_plain_i64(&bytes).unwrap();
            assert_eq!(
                narrowed.to_i64(),
                ground_truth,
                "value mismatch for {vals:?}"
            );
            assert_eq!(narrowed.len(), vals.len());
        }
    }

    #[test]
    fn narrowing_actually_shrinks_footprint() {
        // SF=100 orderkey-shaped: i32 target halves bytes vs i64.
        let vals: Vec<i64> = (1..=1000).map(|i| i * 600_000).collect(); // max 600M
        let bytes = plain_bytes(&vals);
        let narrowed = decode_plain_i64_auto(&bytes, 600_000, 600_000_000).unwrap();
        assert_eq!(narrowed.target(), IntTarget::I32);
        assert_eq!(narrowed.byte_size(), vals.len() * 4);
        // vs the i64 path footprint
        let i64_footprint = decode_plain_i64(&bytes).unwrap().len() * 8;
        assert_eq!(narrowed.byte_size() * 2, i64_footprint);
    }

    #[test]
    fn explicit_target_decode_matches_cast() {
        let vals: Vec<i64> = vec![-5, -1, 0, 1, 100, 127];
        let bytes = plain_bytes(&vals);
        let narrowed = decode_plain_i64_narrowed(&bytes, IntTarget::I8).unwrap();
        match narrowed {
            NarrowedI64::I8(v) => {
                let expect: Vec<i8> = vals.iter().map(|&x| x as i8).collect();
                assert_eq!(v, expect);
            }
            other => panic!("expected I8, got {other:?}"),
        }
    }

    #[test]
    fn i64_target_equals_plain_decode() {
        let vals: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX, 10_000_000_000];
        let bytes = plain_bytes(&vals);
        let narrowed = decode_plain_i64_narrowed(&bytes, IntTarget::I64).unwrap();
        assert_eq!(
            narrowed,
            NarrowedI64::I64(decode_plain_i64(&bytes).unwrap())
        );
    }

    #[test]
    fn empty_buffer_is_empty_not_error() {
        let narrowed = decode_plain_i64_auto(&[], 0, 0).unwrap();
        assert!(narrowed.is_empty());
        // min==max==0 -> narrowest is I8
        assert_eq!(narrowed.target(), IntTarget::I8);
    }

    #[test]
    fn unaligned_buffer_errors() {
        let bytes = vec![0u8; 12]; // not a multiple of 8
        let err = decode_plain_i64_narrowed(&bytes, IntTarget::I32);
        assert!(matches!(
            err,
            Err(CodecError::UnalignedPlainBuffer {
                value_width: 8,
                buffer_len: 12
            })
        ));
    }

    // -----------------------------------------------------------------
    // Frame-of-reference (offset-from-min) narrowing — REV.12 follow-on.
    // For clustered-but-large ranges the ABSOLUTE value needs a wide
    // target but the SPAN (max-min) fits a narrower one; decode
    // `value - min` into the span's width and carry `min`. Order +
    // equality are preserved (offset is a constant) so keys/dates run
    // narrow and only the final output re-widens.
    // -----------------------------------------------------------------

    #[test]
    fn frame_target_narrows_clustered_large_range() {
        // Absolute range needs I32 (max ~1e9); the span (2000) fits I16.
        let (offset, target) = frame_of_reference_target(1_000_000_000, 1_000_002_000);
        assert_eq!(offset, 1_000_000_000);
        assert_eq!(target, IntTarget::I16);
        // ...whereas absolute narrowing only reaches I32.
        assert_eq!(
            narrowest_int_target(1_000_000_000, 1_000_002_000),
            IntTarget::I32
        );
    }

    #[test]
    fn frame_roundtrips_to_i64() {
        let vals: Vec<i64> = vec![1_000_000_000, 1_000_000_500, 1_000_002_000, 1_000_000_001];
        let bytes = plain_bytes(&vals);
        let fr = decode_plain_i64_frame_auto(&bytes, 1_000_000_000, 1_000_002_000).unwrap();
        assert_eq!(fr.target(), IntTarget::I16);
        assert_eq!(fr.offset, 1_000_000_000);
        assert_eq!(fr.byte_size(), vals.len() * 2); // narrow delta width
        assert_eq!(fr.len(), vals.len());
        assert_eq!(fr.to_i64(), vals); // add-back is lossless
    }

    #[test]
    fn frame_handles_negative_min() {
        let vals: Vec<i64> = vec![-1000, 0, 1000, -1, 999];
        let bytes = plain_bytes(&vals);
        let fr = decode_plain_i64_frame_auto(&bytes, -1000, 1000).unwrap();
        assert_eq!(fr.offset, -1000);
        assert_eq!(fr.target(), IntTarget::I16); // span 2000 -> I16
        assert_eq!(fr.to_i64(), vals);
    }

    #[test]
    fn frame_beats_absolute_when_clustered() {
        // SF~1000 orderkey row-group shape: base > i32::MAX (absolute U32,
        // 4 bytes) but a single row-group spans <65536 -> U16 (2 bytes).
        let base = 2_400_000_000i64; // > i32::MAX
        let vals: Vec<i64> = (0..1000).map(|i| base + i * 60).collect(); // span 59_940
        let bytes = plain_bytes(&vals);
        let abs = decode_plain_i64_auto(&bytes, base, base + 59_940).unwrap();
        let fr = decode_plain_i64_frame_auto(&bytes, base, base + 59_940).unwrap();
        assert_eq!(abs.target(), IntTarget::U32); // absolute: 4 bytes
        assert_eq!(fr.target(), IntTarget::U16); // frame-of-ref: 2 bytes
        assert!(fr.byte_size() < abs.byte_size()); // FOR narrows further
        assert_eq!(fr.to_i64(), abs.to_i64()); // identical values
    }

    #[test]
    fn frame_span_overflow_falls_back_to_plain() {
        // Full-i64 span can't narrow -> offset 0, I64, identity (== plain).
        let vals: Vec<i64> = vec![i64::MIN, 0, i64::MAX];
        let bytes = plain_bytes(&vals);
        let fr = decode_plain_i64_frame_auto(&bytes, i64::MIN, i64::MAX).unwrap();
        assert_eq!(fr.offset, 0);
        assert_eq!(fr.target(), IntTarget::I64);
        assert_eq!(fr.to_i64(), vals);
    }

    #[test]
    fn frame_empty_buffer_is_empty() {
        let fr = decode_plain_i64_frame_auto(&[], 100, 100).unwrap();
        assert!(fr.is_empty());
    }
}
