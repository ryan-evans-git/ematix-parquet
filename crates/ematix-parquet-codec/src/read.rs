//! High-level read façade.
//!
//! Wraps the low-level `ParquetFile` → `PageWalker` → decompress →
//! decode pipeline into one call per scalar type. The low-level
//! decoders are still public and unaffected — this module just
//! removes the ~30 lines of boilerplate every consumer would
//! otherwise repeat.
//!
//! Dispatch rules per page:
//!   - First page in the chunk: if `PageType::DictionaryPage`, decode
//!     as PLAIN of `T` to build the dictionary; otherwise treat as a
//!     data page.
//!   - Data pages: encoding `Plain` → decode_plain_*; encoding
//!     `RleDictionary` / `PlainDictionary` → decode_rle_dictionary_into
//!     against the dictionary built above.
//!
//! Other encodings (DELTA_*, BYTE_STREAM_SPLIT) are not yet dispatched
//! by the façade — call the low-level decoders directly for those.
//! That gap is tracked in the v1.0 roadmap.

use ematix_parquet_format::types::{CompressionCodec, Encoding, PageType};
use ematix_parquet_io::{PageWalker, ParquetFile};

use crate::compression::{
    decompress_brotli_into, decompress_gzip_into, decompress_lz4_raw_into,
    decompress_snappy_into, decompress_zstd_into,
};
use crate::dict::decode_rle_dictionary_into;
use crate::error::{CodecError, Result};
use crate::plain::{
    decode_plain_byte_array, decode_plain_f64, decode_plain_i32, decode_plain_i64,
};

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<i64>`. Requires the column's physical type to be INT64.
pub fn read_column_i64(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<i64>> {
    decode_chunk(file, row_group, column, |bytes| decode_plain_i64(bytes))
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<i32>`. Requires the column's physical type to be INT32.
pub fn read_column_i32(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<i32>> {
    decode_chunk(file, row_group, column, |bytes| decode_plain_i32(bytes))
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<f64>`. Requires the column's physical type to be DOUBLE.
pub fn read_column_f64(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<f64>> {
    decode_chunk(file, row_group, column, |bytes| decode_plain_f64(bytes))
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<Vec<u8>>`. Requires the column's physical type to be
/// BYTE_ARRAY.
///
/// Values are copied off the decompressed page buffers so callers
/// don't have to manage page lifetimes. If you need zero-copy
/// access, call `decode_plain_byte_array` against `PageWalker`
/// pages directly.
pub fn read_column_byte_array(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<Vec<u8>>> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<Vec<u8>> = Vec::new();
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        decompress_into(codec, body, &mut decomp)?;

        match hdr.page_type {
            PageType::DictionaryPage => {
                let slices = decode_plain_byte_array(&decomp)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let dph = hdr
                    .data_page_header
                    .as_ref()
                    .ok_or_else(|| CodecError::InvalidInput("data page missing header".into()))?;
                let n = dph.num_values as usize;
                match dph.encoding {
                    Encoding::Plain => {
                        let slices = decode_plain_byte_array(&decomp)?;
                        out.extend(slices.into_iter().map(|s| s.to_vec()));
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        // We can't use decode_rle_dictionary_into directly
                        // for byte_array (Vec<Vec<u8>> isn't Copy). Decode
                        // indices then gather.
                        let indices =
                            crate::dict::decode_rle_dictionary_indices(&decomp, n)?;
                        out.reserve(n);
                        for idx in indices {
                            let v = dict.get(idx as usize).ok_or_else(|| {
                                CodecError::InvalidInput(
                                    "dictionary index out of range".into(),
                                )
                            })?;
                            out.push(v.clone());
                        }
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "encoding not yet dispatched by façade: {other:?}"
                        )));
                    }
                }
            }
            PageType::IndexPage => {
                // Ignored — index pages aren't part of the value stream.
            }
        }
        if out.len() >= total_values {
            break;
        }
    }

    Ok(out)
}

// ---- internals -------------------------------------------------------------

/// Generic chunk-decode for `Copy` scalar types. `decode_plain` knows
/// how to turn a bytes slice into a `Vec<T>` via the PLAIN encoding.
fn decode_chunk<T: Copy>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    decode_plain: impl Fn(&[u8]) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<T> = Vec::new();
    let mut out: Vec<T> = Vec::with_capacity(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        decompress_into(codec, body, &mut decomp)?;

        match hdr.page_type {
            PageType::DictionaryPage => {
                dict = decode_plain(&decomp)?;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let dph = hdr
                    .data_page_header
                    .as_ref()
                    .ok_or_else(|| CodecError::InvalidInput("data page missing header".into()))?;
                let n = dph.num_values as usize;
                match dph.encoding {
                    Encoding::Plain => {
                        out.extend(decode_plain(&decomp)?);
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        decode_rle_dictionary_into(&decomp, &dict, n, &mut out)?;
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "encoding not yet dispatched by façade: {other:?}"
                        )));
                    }
                }
            }
            PageType::IndexPage => {}
        }
        if out.len() >= total_values {
            break;
        }
    }

    Ok(out)
}

/// Pull the raw column-chunk bytes (compressed pages, dictionary
/// page first if present) plus the total value count and codec.
fn read_chunk_raw(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<(Vec<u8>, usize, CompressionCodec)> {
    let md = file.metadata().map_err(io_to_codec)?;
    let rg = md.row_groups.get(row_group).ok_or_else(|| {
        CodecError::InvalidInput(format!("row group {row_group} out of range"))
    })?;
    let col = rg.columns.get(column).ok_or_else(|| {
        CodecError::InvalidInput(format!("column {column} out of range"))
    })?;
    let cm = col
        .meta_data
        .as_ref()
        .ok_or_else(|| CodecError::InvalidInput("column missing inline meta_data".into()))?;
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let bytes = file.read_range(start, length).map_err(io_to_codec)?;
    Ok((bytes, cm.num_values as usize, cm.codec))
}

fn decompress_into(codec: CompressionCodec, body: &[u8], out: &mut Vec<u8>) -> Result<()> {
    match codec {
        CompressionCodec::Uncompressed => {
            out.clear();
            out.extend_from_slice(body);
            Ok(())
        }
        CompressionCodec::Snappy => decompress_snappy_into(body, out),
        CompressionCodec::Zstd => decompress_zstd_into(body, out),
        CompressionCodec::Gzip => decompress_gzip_into(body, out),
        CompressionCodec::Brotli => decompress_brotli_into(body, out),
        CompressionCodec::Lz4Raw => decompress_lz4_raw_into(body, out),
        other => Err(CodecError::Unsupported(format!(
            "compression codec not yet wired in façade: {other:?}"
        ))),
    }
}

fn io_to_codec(e: ematix_parquet_io::IoError) -> CodecError {
    CodecError::InvalidInput(format!("io: {e}"))
}
