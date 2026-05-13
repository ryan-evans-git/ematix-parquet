//! Minimum-viable Parquet writer.
//!
//! `write_{i32,i64,f64,bool,byte_array}_column_to_path(path, name, values)`
//! each produce a complete `.parquet` file with one row group, one
//! column, uncompressed PLAIN encoding, no dictionary, no rep/def
//! levels (REQUIRED column).
//!
//! Layout produced:
//!
//! ```text
//!   [PAR1]                       4 bytes
//!   [DataPage Header (thrift)]   variable
//!   [DataPage Body (PLAIN T)]    variable
//!   [FileMetaData (thrift)]      variable
//!   [footer length (u32 LE)]     4 bytes
//!   [PAR1]                       4 bytes
//! ```
//!
//! Round-trip oracle: write a known `Vec<T>` → read back with
//! `parquet-rs` AND with our own Π.1 façade → assert equality.
//! See `tests/write_*_oracle`.
//!
//! Π.2a/Π.2b focus on the smallest shape that proves the protocol-
//! level machinery is correct. Compression (Snappy/Zstd), dictionary
//! encoding, and multi-column writes follow in Π.2c+.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ematix_parquet_format::metadata::{
    ColumnChunk, ColumnMetaData, DataPageHeader, FileMetaData, PageHeader, RowGroup,
    SchemaElement,
};
use ematix_parquet_format::metadata_writer::{write_file_metadata, write_page_header};
use ematix_parquet_format::types::{
    CompressionCodec, Encoding, FieldRepetitionType, PageType, ParquetType,
};

use crate::compression::{compress_snappy, compress_zstd};
use crate::error::{CodecError, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

// ---- Public per-type entry points (uncompressed defaults) ----------

/// Write a single-column INT64 file (PLAIN, uncompressed, REQUIRED).
pub fn write_i64_column_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i64],
) -> Result<()> {
    write_i64_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

pub fn write_i32_column_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
) -> Result<()> {
    write_i32_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

pub fn write_f64_column_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[f64],
) -> Result<()> {
    write_f64_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

pub fn write_bool_column_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[bool],
) -> Result<()> {
    write_bool_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

pub fn write_byte_array_column_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[&[u8]],
) -> Result<()> {
    write_byte_array_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

// ---- Public per-type entry points with explicit codec --------------

pub fn write_i64_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i64],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_i64(values);
    write_to_path(path, name, ParquetType::Int64, values.len(), body, codec)
}

pub fn write_i32_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_i32(values);
    write_to_path(path, name, ParquetType::Int32, values.len(), body, codec)
}

pub fn write_f64_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[f64],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_f64(values);
    write_to_path(path, name, ParquetType::Double, values.len(), body, codec)
}

pub fn write_bool_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[bool],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_bool(values);
    write_to_path(path, name, ParquetType::Boolean, values.len(), body, codec)
}

pub fn write_byte_array_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[&[u8]],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_byte_array(values);
    write_to_path(path, name, ParquetType::ByteArray, values.len(), body, codec)
}

// ---- Public `Write`-sink variants (in-memory friendly) --------------

pub fn write_i64_column<W: Write>(out: &mut W, name: &str, values: &[i64]) -> Result<()> {
    let body = encode_plain_i64(values);
    write_single_column(
        out,
        name,
        ParquetType::Int64,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
    )
}

pub fn write_i32_column<W: Write>(out: &mut W, name: &str, values: &[i32]) -> Result<()> {
    let body = encode_plain_i32(values);
    write_single_column(
        out,
        name,
        ParquetType::Int32,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
    )
}

pub fn write_f64_column<W: Write>(out: &mut W, name: &str, values: &[f64]) -> Result<()> {
    let body = encode_plain_f64(values);
    write_single_column(
        out,
        name,
        ParquetType::Double,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
    )
}

pub fn write_bool_column<W: Write>(out: &mut W, name: &str, values: &[bool]) -> Result<()> {
    let body = encode_plain_bool(values);
    write_single_column(
        out,
        name,
        ParquetType::Boolean,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
    )
}

pub fn write_byte_array_column<W: Write>(out: &mut W, name: &str, values: &[&[u8]]) -> Result<()> {
    let body = encode_plain_byte_array(values);
    write_single_column(
        out,
        name,
        ParquetType::ByteArray,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
    )
}

// ---- File-path wrapper ----------------------------------------------

fn write_to_path(
    path: impl AsRef<Path>,
    name: &str,
    parquet_type: ParquetType,
    num_values: usize,
    body: Vec<u8>,
    codec: CompressionCodec,
) -> Result<()> {
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column(&mut w, name, parquet_type, num_values, body, codec)?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

// ---- Shared file-skeleton writer ------------------------------------

/// Stamps PAR1 + one DataPage + footer + footer-len + PAR1 into `out`.
/// Type-agnostic: callers pass the encoded PLAIN body, the matching
/// `ParquetType` for the schema leaf, and the compression codec to
/// apply to the body.
fn write_single_column<W: Write>(
    out: &mut W,
    column_name: &str,
    parquet_type: ParquetType,
    num_values: usize,
    body: Vec<u8>,
    codec: CompressionCodec,
) -> Result<()> {
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    let mut written: u64 = 4;

    let uncompressed_size = body.len();
    let compressed_body = compress_body(&body, codec)?;
    let compressed_size = compressed_body.len();

    let data_page_header = PageHeader {
        page_type: PageType::DataPage,
        uncompressed_page_size: uncompressed_size as i32,
        compressed_page_size: compressed_size as i32,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: num_values as i32,
            encoding: Encoding::Plain,
            // Rep/def encodings are declared as RLE per spec, even for
            // REQUIRED columns where no level bytes are emitted.
            definition_level_encoding: Encoding::Rle,
            repetition_level_encoding: Encoding::Rle,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };
    let header_bytes = write_page_header(&data_page_header);

    let data_page_offset = written as i64;
    out.write_all(&header_bytes).map_err(io_to_codec)?;
    written += header_bytes.len() as u64;
    out.write_all(&compressed_body).map_err(io_to_codec)?;
    written += compressed_body.len() as u64;

    let total_uncompressed_size = (header_bytes.len() + uncompressed_size) as i64;
    let total_compressed_size = (header_bytes.len() + compressed_size) as i64;
    let column_name_bytes = column_name.as_bytes();

    let root = SchemaElement {
        column_type: None,
        type_length: None,
        repetition_type: None,
        name: b"schema",
        num_children: Some(1),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    };
    let leaf = SchemaElement {
        column_type: Some(parquet_type),
        type_length: None,
        repetition_type: Some(FieldRepetitionType::Required),
        name: column_name_bytes,
        num_children: None,
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    };
    let cm = ColumnMetaData {
        column_type: parquet_type,
        encodings: vec![Encoding::Plain, Encoding::Rle],
        path_in_schema: vec![column_name_bytes],
        codec,
        num_values: num_values as i64,
        total_uncompressed_size,
        total_compressed_size,
        key_value_metadata: None,
        data_page_offset,
        index_page_offset: None,
        dictionary_page_offset: None,
        statistics: None,
        encoding_stats: None,
        bloom_filter_offset: None,
        bloom_filter_length: None,
        size_statistics: None,
    };
    let cc = ColumnChunk {
        file_path: None,
        file_offset: data_page_offset,
        meta_data: Some(cm),
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
    };
    let rg = RowGroup {
        columns: vec![cc],
        total_byte_size: total_uncompressed_size,
        num_rows: num_values as i64,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: None,
    };
    let md = FileMetaData {
        version: 1,
        schema: vec![root, leaf],
        num_rows: num_values as i64,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: Some(b"ematix-parquet 0.0.1"),
        column_orders: None,
    };
    let footer = write_file_metadata(&md);
    let footer_len = footer.len() as u32;
    out.write_all(&footer).map_err(io_to_codec)?;
    written += footer.len() as u64;

    out.write_all(&footer_len.to_le_bytes()).map_err(io_to_codec)?;
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    written += 8;

    let _ = written;
    Ok(())
}

// ---- PLAIN body encoders --------------------------------------------

fn encode_plain_i64(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_plain_i32(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_plain_f64(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// PLAIN boolean: bit-packed, LSB-first within each byte, padded to
/// `ceil(n / 8)` bytes. Matches `decode_plain_bool`.
fn encode_plain_bool(values: &[bool]) -> Vec<u8> {
    let n_bytes = values.len().div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for (i, &v) in values.iter().enumerate() {
        if v {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// PLAIN byte_array: each value is `u32_le length` + raw bytes.
fn encode_plain_byte_array(values: &[&[u8]]) -> Vec<u8> {
    let total: usize = 4 * values.len() + values.iter().map(|v| v.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    for v in values {
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        out.extend_from_slice(v);
    }
    out
}

fn io_to_codec(e: std::io::Error) -> CodecError {
    CodecError::InvalidInput(format!("io: {e}"))
}

/// Dispatch to the right compressor (or no-op for Uncompressed).
fn compress_body(body: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(body.to_vec()),
        CompressionCodec::Snappy => compress_snappy(body),
        CompressionCodec::Zstd => compress_zstd(body),
        other => Err(CodecError::Unsupported(format!(
            "compression codec not yet supported on the write path: {other:?}"
        ))),
    }
}
