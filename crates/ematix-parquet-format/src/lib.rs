//! ematix-parquet-format — Apache Thrift compact-protocol reader and
//! Parquet metadata types.
//!
//! The crate is intentionally I/O-free: every reader consumes a `&[u8]`
//! slice and advances a cursor. Higher layers (`ematix-parquet-io`)
//! own file handles, range fetchers, and async paging.
//!
//! Layering target (in priority order):
//! 1. Compact-protocol primitives  (varint, zigzag, field headers)
//! 2. Parquet enums + simple structs  (Type, Encoding, Statistics, …)
//! 3. Compound structures           (SchemaElement, ColumnChunk, RowGroup)
//! 4. File-level metadata           (FileMetaData, OffsetIndex, ColumnIndex)
//!
//! Correctness oracle: every decoder is pinned by a byte-identical
//! round-trip against `parquet-rs` for the same input bytes.

pub mod compact;
pub mod error;
pub mod types;
