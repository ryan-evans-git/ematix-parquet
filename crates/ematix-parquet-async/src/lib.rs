//! ematix-parquet-async — async / object-store integration for the
//! ematix-parquet codec.
//!
//! Where the sync `ematix-parquet-io::ParquetFile` operates on a
//! local `std::fs::File`, this crate exposes `AsyncParquetFile` over
//! any `object_store::ObjectStore` — S3, GCS, Azure, HTTP, local FS,
//! in-memory. Footer parse uses the suffix-range trick to cold-open
//! a remote file in ≤2 round trips; per-column reads issue one
//! GET per chunk.
//!
//! The sync crate is untouched and dep-free. This crate pulls in
//! tokio + object_store only when imported. See
//! `docs/plans/PI-11-async-design.md` for the architecture decisions.
//!
//! ## Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use object_store::local::LocalFileSystem;
//! use object_store::path::Path;
//! use ematix_parquet_async::AsyncParquetFile;
//!
//! # async fn example() -> ematix_parquet_async::Result<()> {
//! let store = Arc::new(LocalFileSystem::new());
//! let path = Path::from("path/to/file.parquet");
//! let file = AsyncParquetFile::open(store, path).await?;
//! let metadata = file.metadata()?;
//! println!("row groups: {}", metadata.row_groups.len());
//! # Ok(()) }
//! ```

pub mod error;
pub mod file;

pub use error::{AsyncError, Result};
pub use file::AsyncParquetFile;

// Re-export the ObjectStore trait so consumers don't have to take a
// second dependency to construct one.
pub use object_store;
