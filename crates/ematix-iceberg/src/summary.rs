//! Per-index summary attached to one Iceberg `data_file` entry.
//!
//! A summary is a tiny piece of metadata — `(min_key, max_key)` and
//! an optional dataset-level Bloom blob — that lets a query planner
//! eliminate a file from consideration **without opening the per-file
//! sidecar**. Manifests are already in memory after Iceberg loads
//! them; pruning here is `O(1)` per file vs. `O(open + lookup)` for
//! the sidecar route.
//!
//! ## Pruning semantics
//!
//! The pruning predicate is **conservative**: it returns `false` only
//! when the key is *provably* out of the file's range. Any uncertain
//! case (missing min/max, mixed types not yet handled, ambiguous
//! Bloom result) returns `true` so the caller falls through to the
//! sidecar — never a false negative, possibly a wasted sidecar
//! open. The Bloom dimension is **not yet wired in Π.21a** (the field
//! is decoded and round-tripped but consulted in a follow-up phase);
//! it always behaves as "could contain" for now.
//!
//! ## Physical types
//!
//! [`SummaryKey`] mirrors the subset that [`ematix_parquet_codec::index::Key`]
//! supports: [`I64`](SummaryKey::I64), [`I32`](SummaryKey::I32), and
//! [`Bytes`](SummaryKey::Bytes). Bytes comparison is **lexicographic
//! on the raw bytes** — the same ordering Parquet uses for sorted
//! `BYTE_ARRAY` indexes.

use ematix_parquet_codec::index::Key;

use crate::error::{IcebergIndexError, Result};

/// One value of a summary's `min_key` / `max_key`. Owned (vs.
/// borrowed [`Key<'a>`]) because summaries live inside [`IndexSummary`]
/// structs that the JSON decode produces.
///
/// Only the three physical types the sidecar-index layer indexes are
/// represented; extending to FLOAT / DOUBLE / FIXED_LEN_BYTE_ARRAY
/// follows the codec's expansion, not the iceberg layer's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryKey {
    I64(i64),
    I32(i32),
    /// Lexicographic comparison applies (`Vec<u8>`'s `Ord` impl).
    Bytes(Vec<u8>),
}

impl SummaryKey {
    /// Human-readable variant name. Used in error messages when the
    /// query key's physical type doesn't match the summary's.
    fn variant_name(&self) -> &'static str {
        match self {
            Self::I64(_) => "I64",
            Self::I32(_) => "I32",
            Self::Bytes(_) => "Bytes",
        }
    }

    /// Borrow as a [`Key`] for comparison against a query key. Cheap —
    /// no allocation; `Key::Bytes` borrows the slice.
    fn as_key(&self) -> Key<'_> {
        match self {
            Self::I64(v) => Key::I64(*v),
            Self::I32(v) => Key::I32(*v),
            Self::Bytes(b) => Key::Bytes(b),
        }
    }
}

/// Per-index file-level summary. One per index per `data_file`
/// manifest entry. Sized so the entire summary list fits in the
/// manifest entry's existing extensibility channels (Iceberg's
/// `key_metadata` blob or a property string).
///
/// Field meanings:
///
/// - `name` — matches the sidecar-side `IndexEntry.name` (from
///   [`ematix_parquet_codec::index::IndexManifest`]). The planner
///   uses this to pair a logical index name with both layers.
/// - `min_key` / `max_key` — bounding box of the indexed column's
///   values in this file. Either may be `None` (no information).
/// - `dataset_bloom` — optional file-level SBBF blob (bytes encoded
///   the same way as [`ematix_parquet_codec::bloom::SplitBlockBloomFilter`]).
///   Not consulted yet by Π.21a's pruning predicate; reserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSummary {
    pub name: String,
    pub min_key: Option<SummaryKey>,
    pub max_key: Option<SummaryKey>,
    pub dataset_bloom: Option<Vec<u8>>,
}

impl IndexSummary {
    /// Construct an empty summary (no min/max, no bloom) for an
    /// index that has no per-file statistics yet. The pruning
    /// predicates will return `true` for any query — the planner
    /// falls through to the sidecar.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            min_key: None,
            max_key: None,
            dataset_bloom: None,
        }
    }

    /// Set both bounds at once. Convenience for builders.
    pub fn with_range(mut self, min: SummaryKey, max: SummaryKey) -> Self {
        self.min_key = Some(min);
        self.max_key = Some(max);
        self
    }

    /// Attach a dataset-level Bloom blob. Bytes are whatever
    /// [`ematix_parquet_codec::bloom::SplitBlockBloomFilter`] produced
    /// when serialized — the iceberg layer is opaque to the encoding.
    pub fn with_bloom(mut self, blob: Vec<u8>) -> Self {
        self.dataset_bloom = Some(blob);
        self
    }

    /// **File-level equality pruning.** Returns `false` if `key` is
    /// provably outside the summary's bounds; returns `true` in every
    /// uncertain case (missing bounds, Bloom said maybe, …). Never a
    /// false negative — a `false` result is always safe to skip the
    /// file on.
    ///
    /// Errors on physical-type mismatch (e.g. passing `Key::I32`
    /// against an I64 summary). Mismatch is a programming error
    /// rather than a data-driven false: the planner picked the wrong
    /// index for the predicate.
    pub fn could_contain_eq(&self, key: &Key<'_>) -> Result<bool> {
        // Pull the summary's bounds; if neither is present, no info,
        // be conservative.
        if let Some(min) = &self.min_key {
            self.check_type(min, key)?;
            if key_lt(key, &min.as_key()) {
                return Ok(false);
            }
        }
        if let Some(max) = &self.max_key {
            self.check_type(max, key)?;
            if key_gt(key, &max.as_key()) {
                return Ok(false);
            }
        }
        // Bloom not consulted yet (Π.21c will). Conservative true.
        Ok(true)
    }

    /// **File-level range pruning.** Returns `false` iff `[low, high]`
    /// is provably disjoint from `[min_key, max_key]`. Either bound
    /// of the query may be `None` (open-ended on that side); either
    /// bound of the summary may be `None` (unknown).
    ///
    /// Inclusive on both ends. Errors on physical-type mismatch.
    pub fn could_contain_range(
        &self,
        low: Option<&Key<'_>>,
        high: Option<&Key<'_>>,
    ) -> Result<bool> {
        // If query's `high < summary.min`, no overlap.
        if let (Some(high), Some(min)) = (high, &self.min_key) {
            self.check_type(min, high)?;
            if key_lt(high, &min.as_key()) {
                return Ok(false);
            }
        }
        // If query's `low > summary.max`, no overlap.
        if let (Some(low), Some(max)) = (low, &self.max_key) {
            self.check_type(max, low)?;
            if key_gt(low, &max.as_key()) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Verify the query key matches the summary's physical type.
    /// Returns [`IcebergIndexError::Malformed`] on mismatch — same
    /// "this is a planner bug" signal as the codec's analogous check.
    fn check_type(&self, bound: &SummaryKey, query: &Key<'_>) -> Result<()> {
        let ok = matches!(
            (bound, query),
            (SummaryKey::I64(_), Key::I64(_))
                | (SummaryKey::I32(_), Key::I32(_))
                | (SummaryKey::Bytes(_), Key::Bytes(_))
        );
        if ok {
            Ok(())
        } else {
            Err(IcebergIndexError::Malformed(format!(
                "summary `{}` has bound type {} but query key is {}",
                self.name,
                bound.variant_name(),
                key_variant_name(query),
            )))
        }
    }
}

/// `lhs < rhs` under the appropriate per-type ordering. Mixed-type
/// inputs are a precondition violation — callers must
/// [`IndexSummary::check_type`] first. Defaults to `false` on
/// mismatch so misuse is conservative (no false pruning) even if a
/// check is missed.
fn key_lt(lhs: &Key<'_>, rhs: &Key<'_>) -> bool {
    match (lhs, rhs) {
        (Key::I64(a), Key::I64(b)) => a < b,
        (Key::I32(a), Key::I32(b)) => a < b,
        (Key::Bytes(a), Key::Bytes(b)) => a < b,
        _ => false,
    }
}

fn key_gt(lhs: &Key<'_>, rhs: &Key<'_>) -> bool {
    match (lhs, rhs) {
        (Key::I64(a), Key::I64(b)) => a > b,
        (Key::I32(a), Key::I32(b)) => a > b,
        (Key::Bytes(a), Key::Bytes(b)) => a > b,
        _ => false,
    }
}

fn key_variant_name(k: &Key<'_>) -> &'static str {
    match k {
        Key::I64(_) => "I64",
        Key::I32(_) => "I32",
        Key::Bytes(_) => "Bytes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bounds_could_contain_anything() {
        let s = IndexSummary::new("idx");
        assert!(s.could_contain_eq(&Key::I64(42)).unwrap());
        assert!(s
            .could_contain_range(Some(&Key::I64(0)), Some(&Key::I64(100)))
            .unwrap());
    }

    #[test]
    fn i64_eq_in_and_out_of_range() {
        let s = IndexSummary::new("idx").with_range(SummaryKey::I64(10), SummaryKey::I64(100));
        assert!(s.could_contain_eq(&Key::I64(50)).unwrap());
        assert!(s.could_contain_eq(&Key::I64(10)).unwrap()); // inclusive low
        assert!(s.could_contain_eq(&Key::I64(100)).unwrap()); // inclusive high
        assert!(!s.could_contain_eq(&Key::I64(9)).unwrap());
        assert!(!s.could_contain_eq(&Key::I64(101)).unwrap());
    }

    #[test]
    fn i32_eq_in_and_out_of_range() {
        let s = IndexSummary::new("idx").with_range(SummaryKey::I32(-5), SummaryKey::I32(5));
        assert!(s.could_contain_eq(&Key::I32(0)).unwrap());
        assert!(!s.could_contain_eq(&Key::I32(-6)).unwrap());
        assert!(!s.could_contain_eq(&Key::I32(6)).unwrap());
    }

    #[test]
    fn bytes_eq_lexicographic_range() {
        let s = IndexSummary::new("idx").with_range(
            SummaryKey::Bytes(b"alpha".to_vec()),
            SummaryKey::Bytes(b"omega".to_vec()),
        );
        assert!(s.could_contain_eq(&Key::Bytes(b"hotel")).unwrap());
        assert!(s.could_contain_eq(&Key::Bytes(b"alpha")).unwrap()); // inclusive
        assert!(s.could_contain_eq(&Key::Bytes(b"omega")).unwrap()); // inclusive
        assert!(!s.could_contain_eq(&Key::Bytes(b"aardvark")).unwrap()); // < alpha
        assert!(!s.could_contain_eq(&Key::Bytes(b"zebra")).unwrap()); // > omega
    }

    #[test]
    fn range_overlap_yes_no() {
        let s = IndexSummary::new("idx").with_range(SummaryKey::I64(100), SummaryKey::I64(200));
        // overlap: query range straddles summary
        assert!(s
            .could_contain_range(Some(&Key::I64(150)), Some(&Key::I64(250)))
            .unwrap());
        // overlap: summary fully inside query
        assert!(s
            .could_contain_range(Some(&Key::I64(0)), Some(&Key::I64(1000)))
            .unwrap());
        // no overlap: query entirely below
        assert!(!s
            .could_contain_range(Some(&Key::I64(0)), Some(&Key::I64(50)))
            .unwrap());
        // no overlap: query entirely above
        assert!(!s
            .could_contain_range(Some(&Key::I64(300)), Some(&Key::I64(400)))
            .unwrap());
        // boundary kiss (high == min) is INCLUSIVE → considered overlap.
        assert!(s
            .could_contain_range(Some(&Key::I64(0)), Some(&Key::I64(100)))
            .unwrap());
        assert!(s
            .could_contain_range(Some(&Key::I64(200)), Some(&Key::I64(500)))
            .unwrap());
    }

    #[test]
    fn open_ended_query_range() {
        let s = IndexSummary::new("idx").with_range(SummaryKey::I64(100), SummaryKey::I64(200));
        // No low bound (-∞, 150] — overlaps.
        assert!(s.could_contain_range(None, Some(&Key::I64(150))).unwrap());
        // No low bound (-∞, 50] — below summary, no overlap.
        assert!(!s.could_contain_range(None, Some(&Key::I64(50))).unwrap());
        // No high bound [150, +∞) — overlaps.
        assert!(s.could_contain_range(Some(&Key::I64(150)), None).unwrap());
        // No high bound [500, +∞) — above summary, no overlap.
        assert!(!s.could_contain_range(Some(&Key::I64(500)), None).unwrap());
        // Fully unbounded — always overlaps.
        assert!(s.could_contain_range(None, None).unwrap());
    }

    #[test]
    fn type_mismatch_errors_loudly() {
        let s = IndexSummary::new("idx").with_range(SummaryKey::I64(10), SummaryKey::I64(20));
        let err = s.could_contain_eq(&Key::I32(15)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("I64") && msg.contains("I32"), "got: {msg}");
    }

    #[test]
    fn one_sided_summary_bounds() {
        // Only `min_key` set: anything < min prunes; anything ≥ min is "maybe".
        let s = IndexSummary {
            name: "idx".into(),
            min_key: Some(SummaryKey::I64(50)),
            max_key: None,
            dataset_bloom: None,
        };
        assert!(!s.could_contain_eq(&Key::I64(0)).unwrap());
        assert!(s.could_contain_eq(&Key::I64(50)).unwrap());
        assert!(s.could_contain_eq(&Key::I64(1_000_000)).unwrap());

        // Only `max_key` set: anything > max prunes; anything ≤ max is "maybe".
        let s = IndexSummary {
            name: "idx".into(),
            min_key: None,
            max_key: Some(SummaryKey::I64(100)),
            dataset_bloom: None,
        };
        assert!(s.could_contain_eq(&Key::I64(-1_000_000)).unwrap());
        assert!(s.could_contain_eq(&Key::I64(100)).unwrap());
        assert!(!s.could_contain_eq(&Key::I64(101)).unwrap());
    }

    #[test]
    fn bloom_present_is_round_tripped_but_not_yet_consulted() {
        // Π.21a invariant: a Bloom blob is opaque storage. The
        // pruning predicate still returns the conservative `true`
        // when the only signal is the bloom — wiring its probe
        // happens in a follow-up phase.
        let s = IndexSummary::new("idx").with_bloom(vec![0xFFu8; 64]);
        assert!(s.could_contain_eq(&Key::I64(42)).unwrap());
        assert_eq!(s.dataset_bloom.as_ref().unwrap().len(), 64);
    }
}
