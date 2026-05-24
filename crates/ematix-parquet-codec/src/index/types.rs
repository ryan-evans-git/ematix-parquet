//! Public types for sidecar-index lookups.
//!
//! `Key` is the typed query value passed to [`crate::index::ParquetIndex::lookup_eq`]
//! and friends. It's an enum (rather than a generic) so the public
//! API doesn't need monomorphization per call-site and so the
//! manifest can carry the source column's physical type with the
//! Key variant matched at runtime.
//!
//! `IndexHit` is the unit a lookup returns — points at a specific
//! page within a specific row group of the source file, plus a
//! packed row-bitmap saying which rows within that page match. Feeds
//! directly into `read_column_*_masked_into` after assembling the
//! per-page rowsets into a chunk-wide bitmap.

use crate::index::PhysicalType;

/// Typed query value for an index lookup. Mirrors the subset of
/// physical types the sorted-index builder supports.
///
/// MVP (Π.17b) wires `I64` only. `I32` and `Bytes` variants are
/// reserved so the public API doesn't change when Π.18 adds them.
/// Calling `lookup_eq` with a key whose variant doesn't match the
/// index's `physical_type` returns
/// [`crate::CodecError::InvalidInput`].
#[derive(Debug, Clone)]
pub enum Key<'a> {
    I64(i64),
    I32(i32),
    Bytes(&'a [u8]),
}

impl<'a> Key<'a> {
    /// Best-effort match of this key's variant against a manifest
    /// entry's declared `physical_type`. Used at the API boundary to
    /// fail fast on type mismatch (e.g. passing `Key::I32` to an
    /// `INT64` index).
    pub(crate) fn matches_physical_type(&self, ty: PhysicalType) -> bool {
        matches!(
            (self, ty),
            (Key::I64(_), PhysicalType::Int64)
                | (Key::I32(_), PhysicalType::Int32)
                | (Key::Bytes(_), PhysicalType::ByteArray)
        )
    }

    /// Human-readable variant name. Used in error messages.
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            Key::I64(_) => "I64",
            Key::I32(_) => "I32",
            Key::Bytes(_) => "Bytes",
        }
    }
}

/// One hit returned by a sidecar lookup: a specific page of a
/// specific row group in the source file, plus a packed bitmap
/// naming which rows within that page match.
///
/// `rowset` is `num_values(source_page).div_ceil(8)` bytes long;
/// bit `i` of byte `k` represents row `8k + i` *within the page*
/// (NOT within the row group — the consumer offsets by the page's
/// first-row index when assembling a chunk-wide bitmap for
/// `read_column_*_masked_into`).
#[derive(Debug, Clone)]
pub struct IndexHit {
    /// Source-file row group ordinal (matches
    /// `FileMetaData.row_groups[row_group]`).
    pub row_group: u32,
    /// Zero-based ordinal of the matching data page within the
    /// source row group's column chunk. Dictionary pages are not
    /// counted.
    pub page: u32,
    /// Packed row bitmap, length = `num_values(page).div_ceil(8)`.
    /// Bit `i` of byte `k` set ⇔ row `8k + i` (page-relative)
    /// matches the lookup key.
    pub rowset: Vec<u8>,
}
