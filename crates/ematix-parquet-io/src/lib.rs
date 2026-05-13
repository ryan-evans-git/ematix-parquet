//! ematix-parquet-io — file handles + byte-range reads + page iteration.
//!
//! Sits between `ematix-parquet-format` (pure decoding of in-memory
//! byte slices) and the future page-body codecs. Owns:
//!   - The `File` handle for a parquet file
//!   - The cached footer bytes (so `metadata()` doesn't re-read)
//!   - Synchronous byte-range reads for column-chunk bodies
//!   - A `PageWalker` that turns column-chunk bytes into an iterator
//!     of `(PageHeader, body_slice)` pairs

pub mod file;
pub mod pages;
pub mod error;

pub use error::{IoError, Result};
pub use file::ParquetFile;
pub use pages::PageWalker;
