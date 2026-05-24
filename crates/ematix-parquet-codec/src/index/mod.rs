//! Sidecar indexes for ematix-parquet (Π.17..Π.20).
//!
//! Postgres-style indexes stored in a separate file
//! (`mydata.parquet.idx` next to `mydata.parquet`) so existing parquet
//! files can gain indexes without being rewritten. The sidecar is
//! itself a parquet file — every codec optimization we already ship
//! (predicate-bitmap, dict-preserved reads, page-level skipping) also
//! applies to index lookup.
//!
//! Each index targets `(row_group, page_within_chunk, row_bitmap)`
//! triples so a lookup result feeds straight into the existing
//! `read_column_*_masked_into` family. Pages whose mask is zero are
//! skipped before decompression via the v0.14.0 popcount-mask-range
//! lever — selectivity-aware reads come free.
//!
//! ## Status
//!
//! - **Π.17a (this module): scaffolding + manifest + source-file
//!   fingerprint.** No index logic yet — this PR locks in the file
//!   format, the `KeyValueMetadata` manifest schema, and the contract
//!   that a sidecar reader rejects on source-file mismatch.
//! - Π.17b: sorted-index builder + reader for `INT64`.
//! - Π.18: sorted index for `INT32` + `BYTE_ARRAY`; range queries.
//! - Π.19: per-page Bloom + composite (leading-prefix) index.
//! - Π.20: inverted (text) index + tokenizer trait.
//!
//! ## File layout
//!
//! A `.parquet.idx` is a parquet file. Each row group holds **one
//! index**; the schema is chosen by the index type. The sidecar's
//! footer `KeyValueMetadata` carries an [`IndexManifest`] under the
//! key [`MANIFEST_KEY`], naming each index and the row group that
//! holds it.
//!
//! Readers MUST verify the embedded [`SourceFingerprint`] against the
//! source `.parquet` before answering any lookup. Sidecars are tied
//! to a specific footer state; any rewrite of the source invalidates
//! every sidecar built against it.

pub mod fingerprint;
pub mod manifest;

pub use fingerprint::{compute_source_fingerprint, crc32_ieee};
pub use manifest::{
    IndexEntry, IndexKind, IndexManifest, ManifestError, PhysicalType, SourceFingerprint,
    Tokenizer, MANIFEST_KEY, MANIFEST_VERSION,
};
