//! ematix-parquet-codec — page-body decompression + per-(encoding,
//! physical type) decoders.
//!
//! Layering:
//!   1. `compression`  — wrappers around codec libraries (Snappy now,
//!                       Zstd/Gzip/Brotli/LZ4 later) producing
//!                       decompressed page-body bytes.
//!   2. `plain`        — PLAIN encoding decoders, one per physical
//!                       type (i32/i64/f32/f64/byte_array/...).
//!                       This is the foundation: dictionary pages
//!                       use PLAIN, and v2 fast-paths use it too.
//!   3. `dict`         — RLE_DICTIONARY data-page indices ↔ PLAIN
//!                       dict values. Where most real columns end
//!                       up after writers run a dictionary pass.
//!   4. `delta`        — DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,
//!                       DELTA_BYTE_ARRAY. (Not yet.)
//!   5. `byte_stream_split` — Parquet v2 numeric encoding. (Not yet.)

pub mod column;
pub mod compression;
pub mod dict;
pub mod error;
pub mod plain;
pub mod rle;

pub use error::{CodecError, Result};
