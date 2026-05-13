//! Minimum-viable Parquet writer.
//!
//! `write_i64_column_to_path(path, name, values)` produces a complete
//! `.parquet` file with one row group, one column, uncompressed PLAIN
//! encoding, no dictionary, no rep/def levels (required column).
//!
//! Layout produced:
//!
//! ```text
//!   [PAR1]                       4 bytes
//!   [DataPage Header (thrift)]   variable
//!   [DataPage Body (PLAIN i64)]  values.len() * 8 bytes
//!   [FileMetaData (thrift)]      variable
//!   [footer length (u32 LE)]     4 bytes
//!   [PAR1]                       4 bytes
//! ```
//!
//! Round-trip oracle: write a known `Vec<i64>` → read back with
//! `parquet-rs` → assert byte-for-byte equality. See `tests/write_*`.
//!
//! Π.2a focuses on the smallest shape that proves the protocol-level
//! machinery is correct. Compression, dictionary encoding, multiple
//! columns, multiple row groups, and the f64 / i32 / bool / byte_array
//! variants follow as additive work in Π.2b / Π.2c.

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

use crate::error::{CodecError, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Write a single-column Parquet file containing `values` as a
/// REQUIRED INT64 column named `column_name`. PLAIN encoding,
/// uncompressed, no dictionary, single row group.
pub fn write_i64_column_to_path(
    path: impl AsRef<Path>,
    column_name: &str,
    values: &[i64],
) -> Result<()> {
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_i64_column(&mut w, column_name, values)?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// Same as `write_i64_column_to_path` but writes to an arbitrary
/// `Write` sink. Useful for in-memory round-trip tests.
pub fn write_i64_column<W: Write>(
    out: &mut W,
    column_name: &str,
    values: &[i64],
) -> Result<()> {
    // ---- 1) Magic header ----
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    let mut written: u64 = 4;

    // ---- 2) Data page ----
    // Body: PLAIN i64 = each value as little-endian 8 bytes.
    let mut body = Vec::with_capacity(values.len() * 8);
    for &v in values {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let body_len = body.len() as i32;

    // Header for the data page. Encodings on rep/def levels are
    // declared as RLE per spec even when no levels are emitted; the
    // body simply has zero bytes of level data ahead of the value
    // stream. For a REQUIRED column, parquet-rs accepts the body
    // beginning directly with values.
    let data_page_header = PageHeader {
        page_type: PageType::DataPage,
        uncompressed_page_size: body_len,
        compressed_page_size: body_len,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: values.len() as i32,
            encoding: Encoding::Plain,
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
    out.write_all(&body).map_err(io_to_codec)?;
    written += body.len() as u64;

    // ---- 3) FileMetaData ----
    let total_uncompressed_size = (header_bytes.len() + body.len()) as i64;
    let total_compressed_size = total_uncompressed_size;

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
        column_type: Some(ParquetType::Int64),
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
        column_type: ParquetType::Int64,
        encodings: vec![Encoding::Plain, Encoding::Rle],
        path_in_schema: vec![column_name_bytes],
        codec: CompressionCodec::Uncompressed,
        num_values: values.len() as i64,
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
        num_rows: values.len() as i64,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: None,
    };
    let md = FileMetaData {
        version: 1,
        schema: vec![root, leaf],
        num_rows: values.len() as i64,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: Some(b"ematix-parquet 0.0.1"),
        column_orders: None,
    };
    let footer = write_file_metadata(&md);
    let footer_len = footer.len() as u32;
    out.write_all(&footer).map_err(io_to_codec)?;
    written += footer.len() as u64;

    // ---- 4) Footer length (LE u32) + trailing magic ----
    out.write_all(&footer_len.to_le_bytes()).map_err(io_to_codec)?;
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    written += 8;

    let _ = written;
    Ok(())
}

fn io_to_codec(e: std::io::Error) -> CodecError {
    CodecError::InvalidInput(format!("io: {e}"))
}
