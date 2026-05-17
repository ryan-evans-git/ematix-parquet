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
    ColumnChunk, ColumnMetaData, DataPageHeader, DataPageHeaderV2, DictionaryPageHeader,
    FileMetaData, PageHeader, RowGroup, SchemaElement, Statistics,
};
use ematix_parquet_format::metadata_writer::{write_file_metadata, write_page_header};
use ematix_parquet_format::types::{
    CompressionCodec, Encoding, FieldRepetitionType, PageType, ParquetType,
};

use crate::rle::{encode_rle_bit_packed, min_bit_width_for_dict};

use crate::compression::{
    compress_brotli, compress_gzip, compress_lz4_raw, compress_snappy, compress_zstd,
};
use crate::error::{CodecError, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Magic used at the file trailer in **encrypted-footer** PME mode.
/// Signals that the FileMetaData itself is encrypted and a
/// FileCryptoMetaData trailer sits just ahead of it.
#[cfg(feature = "encryption")]
const PARQUET_MAGIC_ENCRYPTED: &[u8; 4] = b"PARE";

// ---- Multi-column table writer -------------------------------------

/// Type-erased column payload for `write_table_to_path` / `write_table`.
/// Each variant borrows the caller's slice for the duration of the
/// write; nothing is copied except into the on-disk page body.
pub enum ColumnData<'a> {
    I32(&'a [i32]),
    I64(&'a [i64]),
    F64(&'a [f64]),
    Bool(&'a [bool]),
    ByteArray(&'a [&'a [u8]]),
}

impl<'a> ColumnData<'a> {
    fn row_count(&self) -> usize {
        match self {
            ColumnData::I32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::ByteArray(v) => v.len(),
        }
    }

    fn parquet_type(&self) -> ParquetType {
        match self {
            ColumnData::I32(_) => ParquetType::Int32,
            ColumnData::I64(_) => ParquetType::Int64,
            ColumnData::F64(_) => ParquetType::Double,
            ColumnData::Bool(_) => ParquetType::Boolean,
            ColumnData::ByteArray(_) => ParquetType::ByteArray,
        }
    }

    fn encode_plain(&self) -> Vec<u8> {
        match self {
            ColumnData::I32(v) => encode_plain_i32(v),
            ColumnData::I64(v) => encode_plain_i64(v),
            ColumnData::F64(v) => encode_plain_f64(v),
            ColumnData::Bool(v) => encode_plain_bool(v),
            ColumnData::ByteArray(v) => encode_plain_byte_array(v),
        }
    }

    /// Borrow a sub-range as a new `ColumnData` over the same
    /// underlying slice. Used by the multi-row-group writer to walk
    /// each column in `row_group_size` chunks without copying data.
    fn slice(&self, range: std::ops::Range<usize>) -> ColumnData<'a> {
        match self {
            ColumnData::I32(v) => ColumnData::I32(&v[range]),
            ColumnData::I64(v) => ColumnData::I64(&v[range]),
            ColumnData::F64(v) => ColumnData::F64(&v[range]),
            ColumnData::Bool(v) => ColumnData::Bool(&v[range]),
            ColumnData::ByteArray(v) => ColumnData::ByteArray(&v[range]),
        }
    }
}

/// Write a multi-column single-row-group Parquet file.
///
/// All columns must have the same row count. Each column is written
/// as one uncompressed-or-compressed PLAIN-encoded data page. The
/// row group's columns appear in the same order as `columns`.
///
/// `codec` applies to every column. Per-column codec selection lands
/// later — for now, the common case (one writer = one codec) is the
/// only thing wired.
pub fn write_table_to_path<P: AsRef<Path>>(
    path: P,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
) -> Result<()> {
    write_table_to_path_with_row_group_size(path, columns, codec, usize::MAX)
}

/// Same as `write_table_to_path` but writes to an arbitrary `Write` sink.
pub fn write_table<W: Write>(
    out: &mut W,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
) -> Result<()> {
    write_table_with_row_group_size(out, columns, codec, usize::MAX)
}

/// Page-format selector. V1 is the historical default and what every
/// `write_table*` entry point produces unless explicitly opted into V2.
/// V2 is the current parquet-mr / parquet-rs default for new files;
/// choose it for interop with newer ecosystem readers that might
/// short-circuit V1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageVersion {
    V1,
    V2,
}

/// Multi-row-group variant. `row_group_size` is the row count at
/// which the writer cuts a new row group. `usize::MAX` produces a
/// single row group regardless of input size (the default shape).
///
/// Each row group carries its own page headers and its own per-column
/// `Statistics`, computed against the rows in that group only. This
/// is what makes per-row-group predicate pushdown effective on the
/// reader side.
pub fn write_table_to_path_with_row_group_size<P: AsRef<Path>>(
    path: P,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
    row_group_size: usize,
) -> Result<()> {
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_table_with_row_group_size(&mut w, columns, codec, row_group_size)?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// V2-page variant of `write_table_to_path_with_row_group_size`.
/// Same semantics, just emits `PageType::DataPageV2` data pages.
/// REQUIRED columns only — rep/def levels are zero, so the V2 body
/// is just the (compressed) values, identical bytes to V1 except for
/// the page-header type.
pub fn write_table_to_path_v2<P: AsRef<Path>>(
    path: P,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
    row_group_size: usize,
) -> Result<()> {
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_table_inner(
        &mut w,
        columns,
        codec,
        row_group_size,
        PageVersion::V2,
        None,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// Multi-column / multi-row-group writer with **per-column bloom
/// filters**. `bloom_fpps[i]` is the target false-positive
/// probability for column `i`; `None` means no bloom filter for
/// that column. Slice length must equal `columns.len()`.
///
/// Each (row-group, column) pair that has bloom enabled gets its
/// own Split-Block Bloom Filter built from the values in that RG
/// slice. The filter bytes are emitted inline after the column's
/// data page; `bloom_filter_offset` and `bloom_filter_length` are
/// set on the corresponding `ColumnMetaData` so downstream readers
/// (ours + the upstream Rust Parquet reader) discover the filter
/// via the spec.
///
/// Hash input follows the Parquet spec — XXHash64 seed=0 of the
/// value's PLAIN-encoded bytes (LE for scalar, raw bytes for
/// byte_array without length prefix).
pub fn write_table_with_blooms_to_path<P: AsRef<Path>>(
    path: P,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
    row_group_size: usize,
    bloom_fpps: &[Option<f64>],
) -> Result<()> {
    if bloom_fpps.len() != columns.len() {
        return Err(CodecError::InvalidInput(format!(
            "write_table_with_blooms: bloom_fpps length {} != columns length {}",
            bloom_fpps.len(),
            columns.len()
        )));
    }
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_table_inner(
        &mut w,
        columns,
        codec,
        row_group_size,
        PageVersion::V1,
        Some(bloom_fpps),
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// `write_table_to_path_with_row_group_size` for an arbitrary
/// `Write` sink.
pub fn write_table_with_row_group_size<W: Write>(
    out: &mut W,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
    row_group_size: usize,
) -> Result<()> {
    write_table_inner(out, columns, codec, row_group_size, PageVersion::V1, None)
}

fn write_table_inner<W: Write>(
    out: &mut W,
    columns: &[(&str, ColumnData<'_>)],
    codec: CompressionCodec,
    row_group_size: usize,
    page_version: PageVersion,
    bloom_fpps: Option<&[Option<f64>]>,
) -> Result<()> {
    if columns.is_empty() {
        return Err(CodecError::InvalidInput(
            "write_table requires at least one column".into(),
        ));
    }
    if row_group_size == 0 {
        return Err(CodecError::InvalidInput(
            "row_group_size must be > 0".into(),
        ));
    }
    let total_rows = columns[0].1.row_count();
    for (name, c) in &columns[1..] {
        if c.row_count() != total_rows {
            return Err(CodecError::InvalidInput(format!(
                "column {name:?} has row count {} but first column has {}",
                c.row_count(),
                total_rows
            )));
        }
    }

    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    let mut written: u64 = 4;

    /// Per-column descriptor inside one row group: where its single
    /// data page landed and its (owned) stats bytes. Bloom offset/
    /// length are `Some` iff this (RG, column) had a bloom filter
    /// written immediately after its data page.
    struct PreparedColumn {
        data_page_offset: i64,
        total_uncompressed_size: i64,
        total_compressed_size: i64,
        stats: ColumnStats,
        bloom_filter_offset: Option<i64>,
        bloom_filter_length: Option<i32>,
    }

    // One `PreparedColumn` vec per row group, owned for the lifetime
    // of the footer construction below (the `Statistics<'_>` in each
    // `ColumnMetaData` borrows from these).
    let mut prepared_groups: Vec<(usize, Vec<PreparedColumn>)> = Vec::new();

    // Empty input still emits a single empty row group — what the
    // existing `write_empty_column` oracle relies on.
    let mut row_start = 0usize;
    let mut emitted_any = false;
    while row_start < total_rows || !emitted_any {
        let row_end = if total_rows == 0 {
            0
        } else {
            (row_start + row_group_size).min(total_rows)
        };
        let chunk_rows = row_end - row_start;

        let mut prepared: Vec<PreparedColumn> = Vec::with_capacity(columns.len());
        for (col_ix, (_name, col)) in columns.iter().enumerate() {
            let col_slice = col.slice(row_start..row_end);
            let stats = compute_stats(&col_slice);
            let body = col_slice.encode_plain();
            let uncompressed_size = body.len();
            let compressed_body = compress_body(&body, codec)?;
            let compressed_size = compressed_body.len();

            // Build the bloom filter (if requested for this column)
            // from the RG slice — per-(RG, column) granularity, the
            // spec-correct shape for row-group pruning.
            let bloom_bytes: Option<Vec<u8>> = bloom_fpps
                .and_then(|fpps| fpps[col_ix])
                .map(|fpp| build_bloom_for_column(&col_slice, fpp));

            let header = match page_version {
                PageVersion::V1 => PageHeader {
                    page_type: PageType::DataPage,
                    uncompressed_page_size: uncompressed_size as i32,
                    compressed_page_size: compressed_size as i32,
                    crc: None,
                    data_page_header: Some(DataPageHeader {
                        num_values: chunk_rows as i32,
                        encoding: Encoding::Plain,
                        definition_level_encoding: Encoding::Rle,
                        repetition_level_encoding: Encoding::Rle,
                        statistics: Some(stats.as_statistics()),
                    }),
                    index_page_header: None,
                    dictionary_page_header: None,
                    data_page_header_v2: None,
                },
                PageVersion::V2 => PageHeader {
                    page_type: PageType::DataPageV2,
                    uncompressed_page_size: uncompressed_size as i32,
                    compressed_page_size: compressed_size as i32,
                    crc: None,
                    data_page_header: None,
                    index_page_header: None,
                    dictionary_page_header: None,
                    // REQUIRED columns: no rep/def levels, no nulls.
                    // The V2 body is therefore just the (compressed)
                    // value bytes — identical layout to V1.
                    data_page_header_v2: Some(DataPageHeaderV2 {
                        num_values: chunk_rows as i32,
                        num_nulls: 0,
                        num_rows: chunk_rows as i32,
                        encoding: Encoding::Plain,
                        definition_levels_byte_length: 0,
                        repetition_levels_byte_length: 0,
                        is_compressed: codec != CompressionCodec::Uncompressed,
                        statistics: Some(stats.as_statistics()),
                    }),
                },
            };
            let header_bytes = write_page_header(&header);

            let data_page_offset = written as i64;
            out.write_all(&header_bytes).map_err(io_to_codec)?;
            written += header_bytes.len() as u64;
            out.write_all(&compressed_body).map_err(io_to_codec)?;
            written += compressed_body.len() as u64;

            // Write the bloom filter (if any) immediately after the
            // data page; record the offset/length for ColumnMetaData.
            let (bloom_filter_offset, bloom_filter_length) =
                if let Some(blob) = bloom_bytes.as_ref() {
                    let off = written as i64;
                    out.write_all(blob).map_err(io_to_codec)?;
                    let len = blob.len() as i32;
                    written += blob.len() as u64;
                    (Some(off), Some(len))
                } else {
                    (None, None)
                };

            prepared.push(PreparedColumn {
                data_page_offset,
                total_uncompressed_size: (header_bytes.len() + uncompressed_size) as i64,
                total_compressed_size: (header_bytes.len() + compressed_size) as i64,
                stats,
                bloom_filter_offset,
                bloom_filter_length,
            });
        }
        prepared_groups.push((chunk_rows, prepared));
        emitted_any = true;
        row_start = row_end;
    }

    // ---- Build the footer schema + row groups ----
    let root = SchemaElement {
        column_type: None,
        type_length: None,
        repetition_type: None,
        name: b"schema",
        num_children: Some(columns.len() as i32),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    };
    let mut schema: Vec<SchemaElement<'_>> = Vec::with_capacity(columns.len() + 1);
    schema.push(root);
    for (name, col) in columns {
        schema.push(SchemaElement {
            column_type: Some(col.parquet_type()),
            type_length: None,
            repetition_type: Some(FieldRepetitionType::Required),
            name: name.as_bytes(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        });
    }

    let mut row_groups: Vec<RowGroup<'_>> = Vec::with_capacity(prepared_groups.len());
    for (rg_ix, (chunk_rows, prepared)) in prepared_groups.iter().enumerate() {
        let mut row_group_columns: Vec<ColumnChunk<'_>> = Vec::with_capacity(columns.len());
        let mut total_byte_size: i64 = 0;
        for ((name, col), prep) in columns.iter().zip(prepared.iter()) {
            let cm = ColumnMetaData {
                column_type: col.parquet_type(),
                encodings: vec![Encoding::Plain, Encoding::Rle],
                path_in_schema: vec![name.as_bytes()],
                codec,
                num_values: *chunk_rows as i64,
                total_uncompressed_size: prep.total_uncompressed_size,
                total_compressed_size: prep.total_compressed_size,
                key_value_metadata: None,
                data_page_offset: prep.data_page_offset,
                index_page_offset: None,
                dictionary_page_offset: None,
                statistics: Some(prep.stats.as_statistics()),
                encoding_stats: None,
                bloom_filter_offset: prep.bloom_filter_offset,
                bloom_filter_length: prep.bloom_filter_length,
                size_statistics: None,
            };
            row_group_columns.push(ColumnChunk {
                file_path: None,
                file_offset: prep.data_page_offset,
                meta_data: Some(cm),
                offset_index_offset: None,
                offset_index_length: None,
                column_index_offset: None,
                column_index_length: None,
                crypto_metadata: None,
                encrypted_column_metadata: None,
            });
            total_byte_size += prep.total_uncompressed_size;
        }
        row_groups.push(RowGroup {
            columns: row_group_columns,
            total_byte_size,
            num_rows: *chunk_rows as i64,
            sorting_columns: None,
            file_offset: None,
            total_compressed_size: None,
            ordinal: Some(rg_ix as i16),
        });
    }

    let md = FileMetaData {
        version: 1,
        schema,
        num_rows: total_rows as i64,
        row_groups,
        key_value_metadata: None,
        created_by: Some(b"ematix-parquet 0.0.1"),
        column_orders: None,
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    };
    let footer = write_file_metadata(&md);
    let footer_len = footer.len() as u32;
    out.write_all(&footer).map_err(io_to_codec)?;
    out.write_all(&footer_len.to_le_bytes())
        .map_err(io_to_codec)?;
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;

    Ok(())
}

// ---- Public per-type entry points (uncompressed defaults) ----------

/// Write a single-column INT64 file (PLAIN, uncompressed, REQUIRED).
pub fn write_i64_column_to_path(path: impl AsRef<Path>, name: &str, values: &[i64]) -> Result<()> {
    write_i64_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

pub fn write_i32_column_to_path(path: impl AsRef<Path>, name: &str, values: &[i32]) -> Result<()> {
    write_i32_column_to_path_with_codec(path, name, values, CompressionCodec::Uncompressed)
}

/// PME plaintext-footer mode: write a single i32 column encrypted
/// under `key`. The footer stays plaintext but advertises
/// `EncryptionAlgorithm::AesGcmV1`; the single column chunk's
/// data page header + body are both AES-GCM-sealed using the
/// wire frame `[length: u32 LE][nonce: 12B][ct+tag]`.
///
/// Footer-key mode only (no per-column keys; `ColumnChunk.crypto_metadata
/// = EncryptionWithFooterKey`). 16-byte random `aad_file_unique` is
/// generated per file. The caller-supplied `aad_prefix` is NOT embedded
/// in the metadata — readers must supply it themselves
/// (`supply_aad_prefix = true`).
#[cfg(feature = "encryption")]
pub fn write_i32_column_to_path_encrypted(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    key: &ematix_parquet_crypto::key::Key,
    aad_prefix: Option<&[u8]>,
) -> Result<()> {
    let body = encode_plain_i32(values);
    let stats = stats_i32(values);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_encrypted(
        &mut w,
        name,
        ParquetType::Int32,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
        key,
        aad_prefix,
        FooterMode::Plaintext,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// PME **encrypted-footer mode**: write a single i32 column with both
/// page data AND the FileMetaData itself encrypted under `key`.
///
/// File trailer layout (vs plaintext-footer's PAR1 mode):
///
/// ```text
///   [ ... encrypted pages ... ]
///   [ FileCryptoMetaData (Thrift) ]
///   [ encrypted FileMetaData wire frame ]
///   [ footer_len: u32 LE ]    -- covers both Thrift + encrypted frame
///   [ PARE ]                  -- magic distinguishes from PAR1
/// ```
///
/// Footer-key mode only (same as `write_i32_column_to_path_encrypted`).
/// Caller must supply the same `key` to the reader.
#[cfg(feature = "encryption")]
pub fn write_i32_column_to_path_encrypted_footer(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    key: &ematix_parquet_crypto::key::Key,
    aad_prefix: Option<&[u8]>,
) -> Result<()> {
    let body = encode_plain_i32(values);
    let stats = stats_i32(values);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_encrypted(
        &mut w,
        name,
        ParquetType::Int32,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
        key,
        aad_prefix,
        FooterMode::Encrypted,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// PME footer mode for the encrypted-write helpers. Plaintext keeps
/// the readable PAR1-magic footer + per-page encryption; Encrypted
/// additionally seals the FileMetaData itself and emits PARE magic.
#[cfg(feature = "encryption")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FooterMode {
    Plaintext,
    Encrypted,
}

pub fn write_f64_column_to_path(path: impl AsRef<Path>, name: &str, values: &[f64]) -> Result<()> {
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
    let stats = stats_i64(values);
    write_to_path(
        path,
        name,
        ParquetType::Int64,
        values.len(),
        body,
        codec,
        stats,
    )
}

pub fn write_i32_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_i32(values);
    let stats = stats_i32(values);
    write_to_path(
        path,
        name,
        ParquetType::Int32,
        values.len(),
        body,
        codec,
        stats,
    )
}

pub fn write_f64_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[f64],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_f64(values);
    let stats = stats_f64(values);
    write_to_path(
        path,
        name,
        ParquetType::Double,
        values.len(),
        body,
        codec,
        stats,
    )
}

pub fn write_bool_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[bool],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_bool(values);
    let stats = stats_bool(values);
    write_to_path(
        path,
        name,
        ParquetType::Boolean,
        values.len(),
        body,
        codec,
        stats,
    )
}

pub fn write_byte_array_column_to_path_with_codec(
    path: impl AsRef<Path>,
    name: &str,
    values: &[&[u8]],
    codec: CompressionCodec,
) -> Result<()> {
    let body = encode_plain_byte_array(values);
    let stats = stats_byte_array(values);
    write_to_path(
        path,
        name,
        ParquetType::ByteArray,
        values.len(),
        body,
        codec,
        stats,
    )
}

// ---- Public `Write`-sink variants (in-memory friendly) --------------

pub fn write_i64_column<W: Write>(out: &mut W, name: &str, values: &[i64]) -> Result<()> {
    let body = encode_plain_i64(values);
    let stats = stats_i64(values);
    write_single_column(
        out,
        name,
        ParquetType::Int64,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
    )
}

pub fn write_i32_column<W: Write>(out: &mut W, name: &str, values: &[i32]) -> Result<()> {
    let body = encode_plain_i32(values);
    let stats = stats_i32(values);
    write_single_column(
        out,
        name,
        ParquetType::Int32,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
    )
}

pub fn write_f64_column<W: Write>(out: &mut W, name: &str, values: &[f64]) -> Result<()> {
    let body = encode_plain_f64(values);
    let stats = stats_f64(values);
    write_single_column(
        out,
        name,
        ParquetType::Double,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
    )
}

pub fn write_bool_column<W: Write>(out: &mut W, name: &str, values: &[bool]) -> Result<()> {
    let body = encode_plain_bool(values);
    let stats = stats_bool(values);
    write_single_column(
        out,
        name,
        ParquetType::Boolean,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
    )
}

pub fn write_byte_array_column<W: Write>(out: &mut W, name: &str, values: &[&[u8]]) -> Result<()> {
    let body = encode_plain_byte_array(values);
    let stats = stats_byte_array(values);
    write_single_column(
        out,
        name,
        ParquetType::ByteArray,
        values.len(),
        body,
        CompressionCodec::Uncompressed,
        stats,
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
    stats: ColumnStats,
) -> Result<()> {
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column(&mut w, name, parquet_type, num_values, body, codec, stats)?;
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
    stats: ColumnStats,
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
            statistics: Some(stats.as_statistics()),
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
        statistics: Some(stats.as_statistics()),
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
        crypto_metadata: None,
        encrypted_column_metadata: None,
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
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    };
    let footer = write_file_metadata(&md);
    let footer_len = footer.len() as u32;
    out.write_all(&footer).map_err(io_to_codec)?;
    written += footer.len() as u64;

    out.write_all(&footer_len.to_le_bytes())
        .map_err(io_to_codec)?;
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    written += 8;

    let _ = written;
    Ok(())
}

/// Encrypted-write counterpart of `write_single_column`. Emits a
/// single-column-chunk file whose data page header + body are both
/// AES-GCM-sealed under `key`. Footer stays plaintext (PAR1 magic)
/// but advertises `EncryptionAlgorithm::AesGcmV1`.
#[cfg(feature = "encryption")]
#[allow(clippy::too_many_arguments)]
fn write_single_column_encrypted<W: Write>(
    out: &mut W,
    column_name: &str,
    parquet_type: ParquetType,
    num_values: usize,
    body: Vec<u8>,
    codec: CompressionCodec,
    stats: ColumnStats,
    key: &ematix_parquet_crypto::key::Key,
    aad_prefix: Option<&[u8]>,
    footer_mode: FooterMode,
) -> Result<()> {
    use crate::encrypted::{encrypt_module, ColumnEncryptContext};
    use ematix_parquet_crypto::aad::ModuleType;
    use ematix_parquet_crypto::nonce::RandomNonceSource;
    use ematix_parquet_format::metadata::{AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm};

    // 16-byte random file-unique per spec recommendation.
    let file_unique = ematix_parquet_crypto::nonce::random_bytes::<16>()
        .map_err(|e| CodecError::Decompress(format!("PME encrypt: rng: {e}")))?;

    let mut nonces = RandomNonceSource;
    let mut ctx = ColumnEncryptContext {
        key: key.clone(),
        aad_prefix,
        aad_file_unique: &file_unique,
        rg_ordinal: 0,
        col_ordinal: 0,
        nonces: &mut nonces,
    };

    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    let mut written: u64 = 4;

    let uncompressed_size = body.len();
    let compressed_body = compress_body(&body, codec)?;

    // Seal the body FIRST so we know the encrypted-frame size; that's
    // what goes into compressed_page_size per PME spec (the reader uses
    // compressed_page_size to know how many on-disk bytes to read for
    // the body; it must be the encrypted-frame size, not the plaintext
    // compressed size). parquet-rs writes pages this way too.
    let encrypted_body = encrypt_module(&compressed_body, &mut ctx, ModuleType::DataPage, Some(0))?;

    let data_page_header = PageHeader {
        page_type: PageType::DataPage,
        uncompressed_page_size: uncompressed_size as i32,
        compressed_page_size: encrypted_body.len() as i32,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: num_values as i32,
            encoding: Encoding::Plain,
            definition_level_encoding: Encoding::Rle,
            repetition_level_encoding: Encoding::Rle,
            statistics: Some(stats.as_statistics()),
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };
    let plaintext_header = write_page_header(&data_page_header);

    // Seal the page header bytes (DataPageHeader module, page_ord=0).
    let encrypted_header = encrypt_module(
        &plaintext_header,
        &mut ctx,
        ModuleType::DataPageHeader,
        Some(0),
    )?;

    let data_page_offset = written as i64;
    out.write_all(&encrypted_header).map_err(io_to_codec)?;
    written += encrypted_header.len() as u64;
    out.write_all(&encrypted_body).map_err(io_to_codec)?;
    written += encrypted_body.len() as u64;

    // The on-disk *encrypted* sizes go into ColumnMetaData. The
    // uncompressed size is still the original plaintext body length
    // (spec convention — encrypted_size is implicit from the wire
    // frame length prefix that the reader walks).
    let total_uncompressed_size = (plaintext_header.len() + uncompressed_size) as i64;
    let total_compressed_size = (encrypted_header.len() + encrypted_body.len()) as i64;
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
        statistics: Some(stats.as_statistics()),
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
        crypto_metadata: Some(ColumnCryptoMetaData::EncryptionWithFooterKey),
        encrypted_column_metadata: None,
    };
    let rg = RowGroup {
        columns: vec![cc],
        total_byte_size: total_uncompressed_size,
        num_rows: num_values as i64,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: Some(0),
    };
    // FileMetaData differs between the two footer modes:
    //   - Plaintext footer: include encryption_algorithm so the reader
    //     can build the AAD on decrypt of column pages.
    //   - Encrypted footer: encryption_algorithm lives in the
    //     FileCryptoMetaData trailer instead; the (encrypted)
    //     FileMetaData does NOT repeat it (matches parquet-rs).
    let md = FileMetaData {
        version: 1,
        schema: vec![root, leaf],
        num_rows: num_values as i64,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: Some(b"ematix-parquet 0.0.1"),
        column_orders: None,
        encryption_algorithm: match footer_mode {
            FooterMode::Plaintext => Some(EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
                aad_prefix: None,
                aad_file_unique: Some(&file_unique),
                supply_aad_prefix: aad_prefix.map(|_| true),
            })),
            FooterMode::Encrypted => None,
        },
        footer_signing_key_metadata: None,
    };
    let footer = write_file_metadata(&md);

    use ematix_parquet_crypto::aad::build_module_aad;
    use ematix_parquet_crypto::aead::seal;
    use ematix_parquet_crypto::nonce::NonceSource;

    match footer_mode {
        FooterMode::Plaintext => {
            // Plaintext-footer-mode signature per spec:
            //   [ FileMetaData ][ nonce: 12B ][ tag: 16B ]
            // GCM tag over the FileMetaData bytes with the footer key
            // and Footer-module AAD. parquet-rs verifies this on read.
            let footer_aad =
                build_module_aad(aad_prefix, &file_unique, ModuleType::Footer, 0, 0, None);
            let footer_nonce = ctx
                .nonces
                .next()
                .map_err(|e| CodecError::Decompress(format!("PME encrypt: footer nonce: {e}")))?;
            let ct_and_tag = seal(key, &footer_nonce, &footer_aad, &footer)
                .map_err(|e| CodecError::Decompress(format!("PME encrypt: footer seal: {e}")))?;
            let tag = &ct_and_tag[ct_and_tag.len() - 16..];

            let footer_len = (footer.len() + 12 + 16) as u32;
            out.write_all(&footer).map_err(io_to_codec)?;
            written += footer.len() as u64;
            out.write_all(&footer_nonce).map_err(io_to_codec)?;
            written += 12;
            out.write_all(tag).map_err(io_to_codec)?;
            written += 16;

            out.write_all(&footer_len.to_le_bytes())
                .map_err(io_to_codec)?;
            out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
            written += 8;
        }
        FooterMode::Encrypted => {
            // Encrypted-footer trailer layout (PARE magic):
            //   [ FileCryptoMetaData Thrift ]
            //   [ encrypted FileMetaData wire frame ]
            //   [ footer_len: u32 LE ]   -- covers both above
            //   [ PARE ]
            use crate::encrypted::encrypt_footer;
            use ematix_parquet_format::metadata::FileCryptoMetaData;
            use ematix_parquet_format::metadata_writer::write_file_crypto_metadata;

            let fcm = FileCryptoMetaData {
                encryption_algorithm: Some(EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
                    aad_prefix: None,
                    aad_file_unique: Some(&file_unique),
                    supply_aad_prefix: aad_prefix.map(|_| true),
                })),
                key_metadata: None,
            };
            let fcm_bytes = write_file_crypto_metadata(&fcm);
            let encrypted_md_frame =
                encrypt_footer(&footer, key, aad_prefix, &file_unique, ctx.nonces)?;

            let footer_len = (fcm_bytes.len() + encrypted_md_frame.len()) as u32;
            out.write_all(&fcm_bytes).map_err(io_to_codec)?;
            written += fcm_bytes.len() as u64;
            out.write_all(&encrypted_md_frame).map_err(io_to_codec)?;
            written += encrypted_md_frame.len() as u64;

            out.write_all(&footer_len.to_le_bytes())
                .map_err(io_to_codec)?;
            out.write_all(PARQUET_MAGIC_ENCRYPTED)
                .map_err(io_to_codec)?;
            written += 8;
        }
    }

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

// ---- Statistics ----------------------------------------------------
//
// Owned per-column min/max/null_count, computed during encoding and
// borrowed by the `Statistics<'a>` struct that lands in
// `ColumnMetaData.statistics`. Bytes are the spec-defined PLAIN
// encoding of a single value (LE for fixed-width, raw bytes for
// BYTE_ARRAY — without the u32 length prefix that PLAIN body uses).

#[derive(Default)]
struct ColumnStats {
    min: Option<Vec<u8>>,
    max: Option<Vec<u8>>,
    null_count: i64,
}

impl ColumnStats {
    /// Build the borrowed `Statistics<'_>` view fed to the metadata
    /// writer. Emits both deprecated (`min`/`max`) and current
    /// (`min_value`/`max_value`) field pairs — they're identical
    /// bytes, and writing both maximises reader compatibility.
    fn as_statistics(&self) -> Statistics<'_> {
        Statistics {
            max: self.max.as_deref(),
            min: self.min.as_deref(),
            null_count: Some(self.null_count),
            distinct_count: None,
            max_value: self.max.as_deref(),
            min_value: self.min.as_deref(),
            is_max_value_exact: self.max.as_ref().map(|_| true),
            is_min_value_exact: self.min.as_ref().map(|_| true),
        }
    }
}

/// Compute min/max/null_count for a column. No-nulls today, so
/// `null_count = 0` always; the field is still emitted because
/// downstream readers branch on its presence for pushdown.
fn compute_stats(col: &ColumnData<'_>) -> ColumnStats {
    match col {
        ColumnData::I32(v) => stats_i32(v),
        ColumnData::I64(v) => stats_i64(v),
        ColumnData::F64(v) => stats_f64(v),
        ColumnData::Bool(v) => stats_bool(v),
        ColumnData::ByteArray(v) => stats_byte_array(v),
    }
}

/// Build the serialised SBBF bytes (header + bitset) for one
/// column's value slice at the requested false-positive rate.
///
/// Per Parquet spec, the hash input is the value's PLAIN-encoded
/// bytes — LE for scalar types, raw bytes for byte_array without
/// the length prefix. Bool is hashed as a single byte (0/1).
///
/// The filter is sized via `optimal_num_blocks` using the **value
/// count** as the distinct-count estimate. Conservative: when many
/// values are duplicates the actual fpp ends up better than the
/// target. Could be tightened by deduplicating first (extra pass),
/// but the size win is bounded and not worth the per-write cost
/// here — high-cardinality columns are where bloom filters matter
/// most.
fn build_bloom_for_column(col: &ColumnData<'_>, target_fpp: f64) -> Vec<u8> {
    use crate::bloom::{optimal_num_blocks, SplitBlockBloomFilterBuilder};
    let n = col.row_count();
    let num_blocks = optimal_num_blocks(n, target_fpp);
    let mut b = SplitBlockBloomFilterBuilder::new(num_blocks);
    match col {
        ColumnData::I32(vs) => {
            let mut le = [0u8; 4];
            for &v in *vs {
                le.copy_from_slice(&v.to_le_bytes());
                b.insert_bytes(&le);
            }
        }
        ColumnData::I64(vs) => {
            let mut le = [0u8; 8];
            for &v in *vs {
                le.copy_from_slice(&v.to_le_bytes());
                b.insert_bytes(&le);
            }
        }
        ColumnData::F64(vs) => {
            let mut le = [0u8; 8];
            for &v in *vs {
                le.copy_from_slice(&v.to_le_bytes());
                b.insert_bytes(&le);
            }
        }
        ColumnData::Bool(vs) => {
            for &v in *vs {
                b.insert_bytes(&[v as u8]);
            }
        }
        ColumnData::ByteArray(vs) => {
            for v in *vs {
                b.insert_bytes(v);
            }
        }
    }
    b.into_bytes()
}

fn stats_i32(v: &[i32]) -> ColumnStats {
    let mut it = v.iter().copied();
    let Some(first) = it.next() else {
        return ColumnStats::default();
    };
    let (mut mn, mut mx) = (first, first);
    for x in it {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    ColumnStats {
        min: Some(mn.to_le_bytes().to_vec()),
        max: Some(mx.to_le_bytes().to_vec()),
        null_count: 0,
    }
}

fn stats_i64(v: &[i64]) -> ColumnStats {
    let mut it = v.iter().copied();
    let Some(first) = it.next() else {
        return ColumnStats::default();
    };
    let (mut mn, mut mx) = (first, first);
    for x in it {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    ColumnStats {
        min: Some(mn.to_le_bytes().to_vec()),
        max: Some(mx.to_le_bytes().to_vec()),
        null_count: 0,
    }
}

/// f64 stats per Parquet spec:
///   * NaN excluded from min/max.
///   * If the computed min is +0.0, store -0.0; if max is -0.0,
///     store +0.0. Keeps range queries that span zero from
///     incorrectly skipping pages.
///   * If every value is NaN, no min/max are emitted.
fn stats_f64(v: &[f64]) -> ColumnStats {
    let mut mn: Option<f64> = None;
    let mut mx: Option<f64> = None;
    for &x in v {
        if x.is_nan() {
            continue;
        }
        mn = Some(mn.map_or(x, |m| if x < m { x } else { m }));
        mx = Some(mx.map_or(x, |m| if x > m { x } else { m }));
    }
    let (min_b, max_b) = match (mn, mx) {
        (Some(mn), Some(mx)) => {
            let mn = if mn.to_bits() == 0u64 { -0.0_f64 } else { mn };
            let mx = if mx.to_bits() == 0x8000_0000_0000_0000u64 {
                0.0_f64
            } else {
                mx
            };
            (
                Some(mn.to_le_bytes().to_vec()),
                Some(mx.to_le_bytes().to_vec()),
            )
        }
        _ => (None, None),
    };
    ColumnStats {
        min: min_b,
        max: max_b,
        null_count: 0,
    }
}

/// Boolean min/max with `false < true`. Encoded as a single byte
/// (0 or 1), matching how parquet-rs writes BOOLEAN statistics.
fn stats_bool(v: &[bool]) -> ColumnStats {
    if v.is_empty() {
        return ColumnStats::default();
    }
    let any_true = v.iter().any(|&b| b);
    let any_false = v.iter().any(|&b| !b);
    let mn = !any_false; // false present → min is false; otherwise min is true
    let mx = any_true; // true present  → max is true;  otherwise max is false
    ColumnStats {
        min: Some(vec![mn as u8]),
        max: Some(vec![mx as u8]),
        null_count: 0,
    }
}

/// BYTE_ARRAY min/max under unsigned lexicographic ordering. Stored
/// as the raw bytes (NOT length-prefixed — that's a PLAIN-body
/// detail, not a Statistics detail).
fn stats_byte_array(v: &[&[u8]]) -> ColumnStats {
    let mut it = v.iter().copied();
    let Some(first) = it.next() else {
        return ColumnStats::default();
    };
    let mut mn: &[u8] = first;
    let mut mx: &[u8] = first;
    for x in it {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    ColumnStats {
        min: Some(mn.to_vec()),
        max: Some(mx.to_vec()),
        null_count: 0,
    }
}

/// Dispatch to the right compressor (or no-op for Uncompressed).
fn compress_body(body: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(body.to_vec()),
        CompressionCodec::Snappy => compress_snappy(body),
        CompressionCodec::Zstd => compress_zstd(body),
        CompressionCodec::Gzip => compress_gzip(body),
        CompressionCodec::Brotli => compress_brotli(body),
        CompressionCodec::Lz4Raw => compress_lz4_raw(body),
        other => Err(CodecError::Unsupported(format!(
            "compression codec not yet supported on the write path: {other:?}"
        ))),
    }
}

// ---- Dictionary encoding (Π.4c) -------------------------------------
//
// Layout produced for a dict-encoded single-column file:
//
//   [PAR1]
//   [DictPage Header]    page_type=DictionaryPage, encoding=PLAIN
//   [DictPage Body]      PLAIN-encoded unique values
//   [DataPage Header]    page_type=DataPage, encoding=PLAIN_DICTIONARY
//   [DataPage Body]      [bit_width: u8] [RLE/bit-pack indices]
//   [FileMetaData]       ColumnMetaData.dictionary_page_offset set,
//                        encodings = [Plain, Rle, PlainDictionary]
//   [footer length]
//   [PAR1]
//
// Matches what parquet-rs writes for low-cardinality columns. The
// data-page encoding `PLAIN_DICTIONARY` (id=2) is the legacy id;
// modern readers (including ours) accept both `PLAIN_DICTIONARY` and
// `RLE_DICTIONARY` for the data page. We pick the legacy id for
// maximum reader compatibility.

/// Build a unique-value dictionary preserving first-occurrence order
/// + per-input indices. Same shape used by every per-type dict
///   builder below; the closure converts each input element to a
///   hashable key.
fn build_dict_with<T, K>(values: &[T], key_of: impl Fn(&T) -> K) -> (Vec<usize>, Vec<u32>)
where
    K: Eq + std::hash::Hash,
{
    let mut map: std::collections::HashMap<K, u32> = std::collections::HashMap::new();
    let mut dict_positions: Vec<usize> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(values.len());
    for (ix, v) in values.iter().enumerate() {
        let key = key_of(v);
        if let Some(&existing) = map.get(&key) {
            indices.push(existing);
        } else {
            let new_ix = dict_positions.len() as u32;
            dict_positions.push(ix);
            map.insert(key, new_ix);
            indices.push(new_ix);
        }
    }
    (dict_positions, indices)
}

fn build_dict_i64(values: &[i64]) -> (Vec<i64>, Vec<u32>) {
    let (positions, indices) = build_dict_with(values, |&x| x);
    (positions.iter().map(|&p| values[p]).collect(), indices)
}
fn build_dict_i32(values: &[i32]) -> (Vec<i32>, Vec<u32>) {
    let (positions, indices) = build_dict_with(values, |&x| x);
    (positions.iter().map(|&p| values[p]).collect(), indices)
}
fn build_dict_f64(values: &[f64]) -> (Vec<f64>, Vec<u32>) {
    // f64 isn't `Hash` — bucket by bit pattern instead. NaN with the
    // same bit pattern collapses to one dict slot; different NaN bit
    // patterns get separate slots, which matches parquet-rs.
    let (positions, indices) = build_dict_with(values, |&x| x.to_bits());
    (positions.iter().map(|&p| values[p]).collect(), indices)
}
fn build_dict_byte_array<'a>(values: &[&'a [u8]]) -> (Vec<&'a [u8]>, Vec<u32>) {
    let (positions, indices) = build_dict_with(values, |&x| x);
    (positions.iter().map(|&p| values[p]).collect(), indices)
}

/// Stamp PAR1 + dict page + data page + footer into `out`. Same
/// skeleton as `write_single_column`, but the column-chunk has two
/// pages and `dictionary_page_offset` is set.
#[allow(clippy::too_many_arguments)]
fn write_single_column_dict<W: Write>(
    out: &mut W,
    column_name: &str,
    parquet_type: ParquetType,
    num_values: usize,
    dict_body: Vec<u8>,
    dict_len: usize,
    indices: &[u32],
    codec: CompressionCodec,
    stats: ColumnStats,
) -> Result<()> {
    write_single_column_dict_inner(
        out,
        column_name,
        parquet_type,
        num_values,
        dict_body,
        dict_len,
        indices,
        codec,
        stats,
        None,
    )
}

/// Same as `write_single_column_dict` but allows attaching a
/// pre-serialised Split-Block Bloom Filter blob (header + bitset)
/// to the column chunk. Written immediately after the data page;
/// records `bloom_filter_offset` + `bloom_filter_length` on the
/// ColumnMetaData so downstream readers can find it via the spec.
#[allow(clippy::too_many_arguments)]
fn write_single_column_dict_inner<W: Write>(
    out: &mut W,
    column_name: &str,
    parquet_type: ParquetType,
    num_values: usize,
    dict_body: Vec<u8>,
    dict_len: usize,
    indices: &[u32],
    codec: CompressionCodec,
    stats: ColumnStats,
    bloom_bytes: Option<&[u8]>,
) -> Result<()> {
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    let mut written: u64 = 4;

    // ---- Dictionary page ----
    let dict_uncomp_size = dict_body.len();
    let dict_compressed = compress_body(&dict_body, codec)?;
    let dict_comp_size = dict_compressed.len();

    let dict_header = PageHeader {
        page_type: PageType::DictionaryPage,
        uncompressed_page_size: dict_uncomp_size as i32,
        compressed_page_size: dict_comp_size as i32,
        crc: None,
        data_page_header: None,
        index_page_header: None,
        dictionary_page_header: Some(DictionaryPageHeader {
            num_values: dict_len as i32,
            encoding: Encoding::Plain,
            is_sorted: Some(false),
        }),
        data_page_header_v2: None,
    };
    let dict_header_bytes = write_page_header(&dict_header);
    let dict_page_offset = written as i64;
    out.write_all(&dict_header_bytes).map_err(io_to_codec)?;
    written += dict_header_bytes.len() as u64;
    out.write_all(&dict_compressed).map_err(io_to_codec)?;
    written += dict_compressed.len() as u64;

    // ---- Data page (RLE_DICTIONARY indices) ----
    let bit_width = min_bit_width_for_dict(dict_len);
    let mut data_body = Vec::with_capacity(1 + indices.len());
    data_body.push(bit_width);
    data_body.extend_from_slice(&encode_rle_bit_packed(indices, bit_width));

    let data_uncomp_size = data_body.len();
    let data_compressed = compress_body(&data_body, codec)?;
    let data_comp_size = data_compressed.len();

    let data_header = PageHeader {
        page_type: PageType::DataPage,
        uncompressed_page_size: data_uncomp_size as i32,
        compressed_page_size: data_comp_size as i32,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: num_values as i32,
            encoding: Encoding::PlainDictionary,
            definition_level_encoding: Encoding::Rle,
            repetition_level_encoding: Encoding::Rle,
            statistics: Some(stats.as_statistics()),
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };
    let data_header_bytes = write_page_header(&data_header);
    let data_page_offset = written as i64;
    out.write_all(&data_header_bytes).map_err(io_to_codec)?;
    written += data_header_bytes.len() as u64;
    out.write_all(&data_compressed).map_err(io_to_codec)?;
    written += data_compressed.len() as u64;

    // ---- Optional Split-Block Bloom Filter ----
    // Per spec, the bloom filter for a column chunk lives
    // immediately after that chunk's data; the absolute byte
    // offset goes into ColumnMetaData.bloom_filter_offset.
    let (bloom_filter_offset, bloom_filter_length) = if let Some(blob) = bloom_bytes {
        let offset = written as i64;
        out.write_all(blob).map_err(io_to_codec)?;
        let length = blob.len() as i32;
        written += blob.len() as u64;
        (Some(offset), Some(length))
    } else {
        (None, None)
    };

    let total_uncompressed_size =
        (dict_header_bytes.len() + dict_uncomp_size + data_header_bytes.len() + data_uncomp_size)
            as i64;
    let total_compressed_size =
        (dict_header_bytes.len() + dict_comp_size + data_header_bytes.len() + data_comp_size)
            as i64;

    // ---- Footer ----
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
        encodings: vec![Encoding::Plain, Encoding::Rle, Encoding::PlainDictionary],
        path_in_schema: vec![column_name_bytes],
        codec,
        num_values: num_values as i64,
        total_uncompressed_size,
        total_compressed_size,
        key_value_metadata: None,
        data_page_offset,
        index_page_offset: None,
        dictionary_page_offset: Some(dict_page_offset),
        statistics: Some(stats.as_statistics()),
        encoding_stats: None,
        bloom_filter_offset,
        bloom_filter_length,
        size_statistics: None,
    };
    let cc = ColumnChunk {
        file_path: None,
        // file_offset traditionally points at the dict page when
        // present (the column chunk's first byte on disk).
        file_offset: dict_page_offset,
        meta_data: Some(cm),
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
        crypto_metadata: None,
        encrypted_column_metadata: None,
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
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    };
    let footer = write_file_metadata(&md);
    let footer_len = footer.len() as u32;
    out.write_all(&footer).map_err(io_to_codec)?;
    written += footer.len() as u64;
    out.write_all(&footer_len.to_le_bytes())
        .map_err(io_to_codec)?;
    out.write_all(PARQUET_MAGIC).map_err(io_to_codec)?;
    written += 8;

    let _ = written;
    Ok(())
}

// ---- Public dict-encoded entry points -------------------------------

/// Write a single-column INT64 file using PLAIN_DICTIONARY encoding.
/// The dictionary holds the unique values; the data page holds
/// RLE/bit-pack-encoded indices into it.
pub fn write_i64_column_dict_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i64],
    codec: CompressionCodec,
) -> Result<()> {
    let stats = stats_i64(values);
    let (dict, indices) = build_dict_i64(values);
    let dict_body = encode_plain_i64(&dict);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict(
        &mut w,
        name,
        ParquetType::Int64,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

pub fn write_i32_column_dict_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    codec: CompressionCodec,
) -> Result<()> {
    let stats = stats_i32(values);
    let (dict, indices) = build_dict_i32(values);
    let dict_body = encode_plain_i32(&dict);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict(
        &mut w,
        name,
        ParquetType::Int32,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// Same as `write_i32_column_dict_to_path` but also emits a
/// Split-Block Bloom Filter built from the column's distinct
/// values, and records its offset/length in the ColumnMetaData
/// so downstream readers can consult it via the spec.
///
/// `target_fpp` sizes the filter via `bloom::optimal_num_blocks`.
/// 0.01 is a reasonable default; smaller is more selective but
/// uses more bytes (filter size scales linearly with -log fpp).
pub fn write_i32_column_dict_with_bloom_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i32],
    codec: CompressionCodec,
    target_fpp: f64,
) -> Result<()> {
    use crate::bloom::{optimal_num_blocks, SplitBlockBloomFilterBuilder};

    let stats = stats_i32(values);
    let (dict, indices) = build_dict_i32(values);
    let dict_body = encode_plain_i32(&dict);

    // Bloom filter is built over the *distinct* values (the dict).
    // Each i32 is hashed as its 4-byte LE encoding — the same shape
    // every other Parquet writer uses for the spec's "PLAIN-encoded
    // bytes" hash input.
    let num_blocks = optimal_num_blocks(dict.len(), target_fpp);
    let mut builder = SplitBlockBloomFilterBuilder::new(num_blocks);
    let mut le_buf = [0u8; 4];
    for &v in &dict {
        le_buf.copy_from_slice(&v.to_le_bytes());
        builder.insert_bytes(&le_buf);
    }
    let bloom_bytes = builder.into_bytes();

    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict_inner(
        &mut w,
        name,
        ParquetType::Int32,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
        Some(&bloom_bytes),
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// `write_i64_column_dict_to_path` with an SBBF built over the
/// distinct values (8-byte LE hash input).
pub fn write_i64_column_dict_with_bloom_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[i64],
    codec: CompressionCodec,
    target_fpp: f64,
) -> Result<()> {
    use crate::bloom::{optimal_num_blocks, SplitBlockBloomFilterBuilder};
    let stats = stats_i64(values);
    let (dict, indices) = build_dict_i64(values);
    let dict_body = encode_plain_i64(&dict);

    let num_blocks = optimal_num_blocks(dict.len(), target_fpp);
    let mut builder = SplitBlockBloomFilterBuilder::new(num_blocks);
    let mut le_buf = [0u8; 8];
    for &v in &dict {
        le_buf.copy_from_slice(&v.to_le_bytes());
        builder.insert_bytes(&le_buf);
    }
    let bloom_bytes = builder.into_bytes();

    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict_inner(
        &mut w,
        name,
        ParquetType::Int64,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
        Some(&bloom_bytes),
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// `write_f64_column_dict_to_path` with an SBBF built over the
/// distinct values (8-byte LE hash input — same shape parquet-rs
/// uses for double columns).
pub fn write_f64_column_dict_with_bloom_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[f64],
    codec: CompressionCodec,
    target_fpp: f64,
) -> Result<()> {
    use crate::bloom::{optimal_num_blocks, SplitBlockBloomFilterBuilder};
    let stats = stats_f64(values);
    let (dict, indices) = build_dict_f64(values);
    let dict_body = encode_plain_f64(&dict);

    let num_blocks = optimal_num_blocks(dict.len(), target_fpp);
    let mut builder = SplitBlockBloomFilterBuilder::new(num_blocks);
    let mut le_buf = [0u8; 8];
    for &v in &dict {
        le_buf.copy_from_slice(&v.to_le_bytes());
        builder.insert_bytes(&le_buf);
    }
    let bloom_bytes = builder.into_bytes();

    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict_inner(
        &mut w,
        name,
        ParquetType::Double,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
        Some(&bloom_bytes),
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// `write_byte_array_column_dict_to_path` with an SBBF built over
/// the distinct values. Hash input is the raw bytes — **without**
/// any length prefix (per the Parquet spec).
pub fn write_byte_array_column_dict_with_bloom_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[&[u8]],
    codec: CompressionCodec,
    target_fpp: f64,
) -> Result<()> {
    use crate::bloom::{optimal_num_blocks, SplitBlockBloomFilterBuilder};
    let stats = stats_byte_array(values);
    let (dict, indices) = build_dict_byte_array(values);
    let dict_body = encode_plain_byte_array(&dict);

    let num_blocks = optimal_num_blocks(dict.len(), target_fpp);
    let mut builder = SplitBlockBloomFilterBuilder::new(num_blocks);
    for v in &dict {
        builder.insert_bytes(v);
    }
    let bloom_bytes = builder.into_bytes();

    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict_inner(
        &mut w,
        name,
        ParquetType::ByteArray,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
        Some(&bloom_bytes),
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

pub fn write_f64_column_dict_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[f64],
    codec: CompressionCodec,
) -> Result<()> {
    let stats = stats_f64(values);
    let (dict, indices) = build_dict_f64(values);
    let dict_body = encode_plain_f64(&dict);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict(
        &mut w,
        name,
        ParquetType::Double,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}

/// BYTE_ARRAY is the canonical dict use case: low-cardinality string
/// columns shrink dramatically when the unique strings are stored
/// once and the data page only carries indices.
pub fn write_byte_array_column_dict_to_path(
    path: impl AsRef<Path>,
    name: &str,
    values: &[&[u8]],
    codec: CompressionCodec,
) -> Result<()> {
    let stats = stats_byte_array(values);
    let (dict, indices) = build_dict_byte_array(values);
    let dict_body = encode_plain_byte_array(&dict);
    let f = File::create(path.as_ref()).map_err(io_to_codec)?;
    let mut w = BufWriter::new(f);
    write_single_column_dict(
        &mut w,
        name,
        ParquetType::ByteArray,
        values.len(),
        dict_body,
        dict.len(),
        &indices,
        codec,
        stats,
    )?;
    w.flush().map_err(io_to_codec)?;
    Ok(())
}
