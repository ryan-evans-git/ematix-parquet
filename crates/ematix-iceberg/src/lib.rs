//! Iceberg integration for ematix-parquet sidecar indexes (Π.21a).
//!
//! ## What this crate is
//!
//! `ematix-iceberg` defines the **extension fields** that an Iceberg
//! writer appends to each `data_file` manifest entry so that an
//! Iceberg-aware query planner can:
//!
//! 1. **Prune at the file level** before opening any per-file sidecar
//!    (`mydata.parquet.idx`), via per-index `(min_key, max_key)` and
//!    optional dataset-level Bloom blobs.
//! 2. **Locate the per-file sidecar** via a relative path stored in
//!    the manifest entry alongside the data file itself.
//!
//! This crate carries **only the contract** — the on-wire JSON shape,
//! the typed Rust struct, and the pruning predicate. Wiring into
//! `iceberg-rust` (reading/writing real Iceberg manifests) lives in
//! Π.21b. Once that lands the scaffolding here gets one more user:
//! the `iceberg::spec::DataFile` builder fills in the extension JSON
//! via the standard `key_metadata` or properties channel.
//!
//! ## Why this layer exists
//!
//! Without it, an Iceberg query for `customer_id = 42` reads every
//! file in the table (Iceberg's native column min/max only helps if
//! the column is partitioned or has tight per-file stats — and even
//! then it doesn't reach into pages). With it:
//!
//! ```text
//! query: customer_id = 42
//!   → load manifests
//!   → for each data_file, check ematix_index_summaries:
//!       summary.could_contain_eq(&Key::I64(42))  // O(1) per file
//!   → for the small surviving set, open .parquet.idx,
//!     do page-granular lookup via ematix-parquet-codec::index
//!   → masked-decode the source pages with the resulting rowsets
//! ```
//!
//! Partition keys become a runtime choice: pick any indexed column,
//! pay the per-file summary check, win.
//!
//! ## Layering
//!
//! - [`summary`] — [`IndexSummary`] / [`SummaryKey`] / pruning predicates
//!   ([`IndexSummary::could_contain_eq`], [`IndexSummary::could_contain_range`]).
//! - [`extension`] — [`EmatixDataFileExtension`] (the full per-`data_file`
//!   blob: sidecar relative path + summaries vec), JSON encode/decode,
//!   and the Iceberg property key constants ([`EMATIX_EXTENSION_KEY`]).
//! - [`error`] — [`IcebergIndexError`], returned by JSON decode.
//!
//! ## Wire format (stable as of Π.21)
//!
//! JSON. Same reason as [`ematix_parquet_codec::index`]: matches the
//! shape Iceberg / Spark / Hudi already produce, no new dep for
//! external readers, and the per-file overhead is tiny. The full
//! shape — including the breaking-change-version suffix — lives in
//! the [`extension`] module docs.

pub mod error;
pub mod extension;
pub mod summary;

pub use error::IcebergIndexError;
pub use extension::{EmatixDataFileExtension, EMATIX_EXTENSION_KEY, EMATIX_EXTENSION_VERSION};
pub use summary::{IndexSummary, SummaryKey};
