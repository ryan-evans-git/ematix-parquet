//! Page-skip via column index.
//!
//! When a Parquet writer emits a `ColumnIndex` for a column chunk, it
//! records per-page `[min, max]` bounds (plus a null-page bitmap).
//! Selective queries can use that to decide which data pages to even
//! decompress + decode — a 10× win when the predicate filters out most
//! pages (e.g. range queries on sorted-within-rowgroup timestamps).
//!
//! This module is pure logic on top of `format::ColumnIndex`. The
//! caller is responsible for:
//!   - Fetching the index bytes and decoding them into a `ColumnIndex`
//!     (see `read_column_index` in the format crate)
//!   - Honouring the returned `Vec<bool>` when walking pages — e.g.
//!     skipping `next_page()` results whose index is `false`. The
//!     ordering between `ColumnIndex.min_values` and the actual
//!     pages in `PageWalker` is the same: data-page i in the chunk
//!     corresponds to `ColumnIndex` entry i (dictionary pages are
//!     not represented in `ColumnIndex`).

use ematix_parquet_format::metadata::ColumnIndex;

use crate::error::{CodecError, Result};

/// Page-overlap selector for INT32 columns. Returns a `Vec<bool>`
/// the same length as the index's pages, with `true` for pages whose
/// `[min, max]` intersects `[lo, hi]` (inclusive on both ends).
///
/// Null pages always select `false` — a null page contributes no
/// rows to a non-null predicate.
pub fn select_pages_overlapping_i32(idx: &ColumnIndex, lo: i32, hi: i32) -> Result<Vec<bool>> {
    if lo > hi {
        return Err(CodecError::Decompress(
            "select_pages_overlapping: lo > hi".into(),
        ));
    }
    let n = idx.null_pages.len();
    if idx.min_values.len() != n || idx.max_values.len() != n {
        return Err(CodecError::Decompress(
            "ColumnIndex: null_pages/min/max length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if idx.null_pages[i] {
            out.push(false);
            continue;
        }
        let pmin = decode_i32(idx.min_values[i])?;
        let pmax = decode_i32(idx.max_values[i])?;
        out.push(pmax >= lo && pmin <= hi);
    }
    Ok(out)
}

/// INT64 variant of [`select_pages_overlapping_i32`].
pub fn select_pages_overlapping_i64(idx: &ColumnIndex, lo: i64, hi: i64) -> Result<Vec<bool>> {
    if lo > hi {
        return Err(CodecError::Decompress(
            "select_pages_overlapping: lo > hi".into(),
        ));
    }
    let n = idx.null_pages.len();
    if idx.min_values.len() != n || idx.max_values.len() != n {
        return Err(CodecError::Decompress(
            "ColumnIndex: null_pages/min/max length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if idx.null_pages[i] {
            out.push(false);
            continue;
        }
        let pmin = decode_i64(idx.min_values[i])?;
        let pmax = decode_i64(idx.max_values[i])?;
        out.push(pmax >= lo && pmin <= hi);
    }
    Ok(out)
}

fn decode_i32(bytes: &[u8]) -> Result<i32> {
    if bytes.len() != 4 {
        return Err(CodecError::Decompress(format!(
            "ColumnIndex i32 stat: expected 4 bytes, got {}",
            bytes.len()
        )));
    }
    Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
}

fn decode_i64(bytes: &[u8]) -> Result<i64> {
    if bytes.len() != 8 {
        return Err(CodecError::Decompress(format!(
            "ColumnIndex i64 stat: expected 8 bytes, got {}",
            bytes.len()
        )));
    }
    Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
}
