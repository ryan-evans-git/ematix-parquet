//! Async read façade — mirror of `ematix_parquet_codec::read` with
//! async I/O at the chunk-fetch boundary.
//!
//! The page walk + per-encoding decode is the same byte-slice work
//! the sync crate does; once the chunk's compressed bytes are in
//! memory, no more `await` points. This keeps the async surface
//! narrow (one GET per chunk) and lets us reuse every existing
//! decoder.

use bytes::Bytes;
use ematix_parquet_codec::compression::{
    decompress_brotli_into, decompress_gzip_into, decompress_lz4_raw_into, decompress_snappy_into,
    decompress_zstd_into,
};
use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, decode_rle_dictionary_into};
use ematix_parquet_codec::plain::{
    decode_plain_byte_array, decode_plain_f64, decode_plain_i32, decode_plain_i64,
};
use ematix_parquet_format::metadata::PageHeader;
use ematix_parquet_format::types::{CompressionCodec, Encoding, PageType};
use ematix_parquet_io::PageWalker;

use crate::error::{AsyncError, Result};
use crate::file::AsyncParquetFile;

/// Read INT64 column chunk asynchronously. Issues one GET for the
/// chunk bytes, then walks pages + decodes in memory.
pub async fn read_column_i64_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    read_column_i64_async_into(file, row_group, column, &mut out).await?;
    Ok(out)
}

/// `read_column_i64_async` writing into a caller-provided buffer.
pub async fn read_column_i64_async_into(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<i64>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw_async(file, row_group, column).await?;
    decode_chunk_into(&chunk_bytes, total_values, codec, out, decode_plain_i64)
}

/// Read INT32 column chunk asynchronously.
pub async fn read_column_i32_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    read_column_i32_async_into(file, row_group, column, &mut out).await?;
    Ok(out)
}

pub async fn read_column_i32_async_into(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<i32>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw_async(file, row_group, column).await?;
    decode_chunk_into(&chunk_bytes, total_values, codec, out, decode_plain_i32)
}

/// Read DOUBLE column chunk asynchronously.
pub async fn read_column_f64_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    read_column_f64_async_into(file, row_group, column, &mut out).await?;
    Ok(out)
}

pub async fn read_column_f64_async_into(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<f64>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw_async(file, row_group, column).await?;
    decode_chunk_into(&chunk_bytes, total_values, codec, out, decode_plain_f64)
}

// ============================================================
// byte_array — Vec<Vec<u8>> shape
// ============================================================

/// Read BYTE_ARRAY column chunk asynchronously, returning owned
/// `Vec<u8>` per row. For low-cardinality dict-encoded columns this
/// is much slower than the offsets variant (per-row allocation);
/// prefer `read_column_byte_array_offsets_async` when downstream
/// can consume the Arrow-style flat layout.
pub async fn read_column_byte_array_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    read_column_byte_array_async_into(file, row_group, column, &mut out).await?;
    Ok(out)
}

pub async fn read_column_byte_array_async_into(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    out.clear();
    let (chunk_bytes, total_values, codec) = read_chunk_raw_async(file, row_group, column).await?;

    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();
    let mut dict: Vec<Vec<u8>> = Vec::new();
    out.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_async)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp).map_err(codec_to_async)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        let slices =
                            decode_plain_byte_array(info.values).map_err(codec_to_async)?;
                        out.extend(slices.into_iter().map(|s| s.to_vec()));
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(AsyncError::Format(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        let indices = decode_rle_dictionary_indices(info.values, info.num_values)
                            .map_err(codec_to_async)?;
                        out.reserve(info.num_values);
                        for idx in indices {
                            let v = dict.get(idx as usize).ok_or_else(|| {
                                AsyncError::Format("dictionary index out of range".into())
                            })?;
                            out.push(v.clone());
                        }
                    }
                    other => {
                        return Err(AsyncError::Format(format!(
                            "encoding not yet dispatched by async façade: {other:?}"
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
    Ok(())
}

// ============================================================
// byte_array — Arrow-style flat bytes + offsets
// ============================================================

/// Async equivalent of `read_column_byte_array_offsets`. Returns
/// `(values, offsets)` where row `i` is `values[offsets[i] as usize
/// .. offsets[i + 1] as usize]`. The trailing offset is the total
/// values length (Arrow BinaryArray convention).
pub async fn read_column_byte_array_offsets_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<(Vec<u8>, Vec<u32>)> {
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    read_column_byte_array_offsets_async_into(file, row_group, column, &mut bytes, &mut offsets)
        .await?;
    Ok((bytes, offsets))
}

pub async fn read_column_byte_array_offsets_async_into(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    out_bytes: &mut Vec<u8>,
    out_offsets: &mut Vec<u32>,
) -> Result<()> {
    out_bytes.clear();
    out_offsets.clear();

    let (chunk_bytes, total_values, codec) = read_chunk_raw_async(file, row_group, column).await?;

    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    // Dictionary as flat bytes + offsets (same shape as the output).
    let mut dict_bytes: Vec<u8> = Vec::new();
    let mut dict_offsets: Vec<u32> = vec![0];

    out_offsets.reserve(total_values + 1);
    out_offsets.push(0);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_async)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp).map_err(codec_to_async)?;
                let total: usize = slices.iter().map(|s| s.len()).sum();
                dict_bytes.clear();
                dict_bytes.reserve(total);
                dict_offsets.clear();
                dict_offsets.reserve(slices.len() + 1);
                dict_offsets.push(0);
                let mut acc: u32 = 0;
                for s in &slices {
                    dict_bytes.extend_from_slice(s);
                    acc += s.len() as u32;
                    dict_offsets.push(acc);
                }
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        // Inline PLAIN walk: u32_le length + bytes per value.
                        let body = info.values;
                        let mut i = 0usize;
                        let mut emitted = 0usize;
                        let cap_target = info.num_values;
                        out_bytes.reserve(body.len());
                        out_offsets.reserve(cap_target);
                        let mut acc: u32 = *out_offsets.last().unwrap();
                        while i < body.len() && emitted < cap_target {
                            if i + 4 > body.len() {
                                return Err(AsyncError::Format(
                                    "PLAIN byte_array: truncated length prefix".into(),
                                ));
                            }
                            let len =
                                u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) as usize;
                            i += 4;
                            if i + len > body.len() {
                                return Err(AsyncError::Format(
                                    "PLAIN byte_array: value runs past page end".into(),
                                ));
                            }
                            out_bytes.extend_from_slice(&body[i..i + len]);
                            i += len;
                            acc += len as u32;
                            out_offsets.push(acc);
                            emitted += 1;
                        }
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict_offsets.len() < 2 {
                            return Err(AsyncError::Format(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        let indices = decode_rle_dictionary_indices(info.values, info.num_values)
                            .map_err(codec_to_async)?;
                        let dict_n = dict_offsets.len() - 1;
                        out_offsets.reserve(indices.len());
                        let mut acc: u32 = *out_offsets.last().unwrap();
                        for idx in indices {
                            let i = idx as usize;
                            if i >= dict_n {
                                return Err(AsyncError::Format(
                                    "dictionary index out of range".into(),
                                ));
                            }
                            let lo = dict_offsets[i] as usize;
                            let hi = dict_offsets[i + 1] as usize;
                            out_bytes.extend_from_slice(&dict_bytes[lo..hi]);
                            acc += (hi - lo) as u32;
                            out_offsets.push(acc);
                        }
                    }
                    other => {
                        return Err(AsyncError::Format(format!(
                            "encoding not yet dispatched by async façade: {other:?}"
                        )));
                    }
                }
            }
            PageType::IndexPage => {}
        }
        // Stop once we've emitted total_values values (offsets has +1).
        if out_offsets.len() >= total_values + 1 {
            break;
        }
    }
    Ok(())
}

// ============================================================
// internal: async chunk fetch + sync chunk decode
// ============================================================

/// Compute the chunk's `(start, length)` from metadata, issue one
/// async GET, return the bytes + the chunk's total value count + codec.
async fn read_chunk_raw_async(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
) -> Result<(Bytes, usize, CompressionCodec)> {
    let md = file.metadata()?;
    let rg = md
        .row_groups
        .get(row_group)
        .ok_or_else(|| AsyncError::Format(format!("row group {row_group} out of range")))?;
    let col = rg
        .columns
        .get(column)
        .ok_or_else(|| AsyncError::Format(format!("column {column} out of range")))?;
    let cm = col
        .meta_data
        .as_ref()
        .ok_or_else(|| AsyncError::Format("column missing inline meta_data".into()))?;
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let bytes = file.read_range(start, length).await?;
    Ok((bytes, cm.num_values as usize, cm.codec))
}

/// Sync chunk decode that mirrors `ematix_parquet_codec::read::
/// decode_chunk_into` exactly — walks dictionary then data pages,
/// dispatches Plain vs RleDictionary, writes into the caller's
/// buffer. Duplicated here (rather than re-exported from codec)
/// because the codec version expects a sync `ParquetFile` to fetch
/// the chunk; we already have the bytes in hand.
fn decode_chunk_into<T: Copy>(
    chunk_bytes: &[u8],
    total_values: usize,
    codec: CompressionCodec,
    out: &mut Vec<T>,
    decode_plain: impl Fn(&[u8]) -> ematix_parquet_codec::error::Result<Vec<T>>,
) -> Result<()> {
    out.clear();
    let mut walker = PageWalker::new(chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<T> = Vec::new();
    out.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_async)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                dict = decode_plain(&decomp).map_err(codec_to_async)?;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        out.extend(decode_plain(info.values).map_err(codec_to_async)?);
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(AsyncError::Format(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        decode_rle_dictionary_into(info.values, &dict, info.num_values, out)
                            .map_err(codec_to_async)?;
                    }
                    other => {
                        return Err(AsyncError::Format(format!(
                            "encoding not yet dispatched by async façade: {other:?}"
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

    Ok(())
}

/// Mirror of `ematix_parquet_codec::read::DataPageInfo` — local copy
/// because that type is private. The shape and rules match exactly.
struct DataPageInfo<'a> {
    num_values: usize,
    encoding: Encoding,
    values: &'a [u8],
}

/// Mirror of codec's `data_page_view`. V1 decompresses the whole
/// body; V2 keeps rep/def levels uncompressed and only decompresses
/// the values portion (gated by `is_compressed`).
fn data_page_view<'a>(
    hdr: &'a PageHeader<'a>,
    body: &'a [u8],
    chunk_codec: CompressionCodec,
    decomp: &'a mut Vec<u8>,
) -> Result<DataPageInfo<'a>> {
    if let Some(ref dph) = hdr.data_page_header {
        decompress_into(chunk_codec, body, decomp)?;
        Ok(DataPageInfo {
            num_values: dph.num_values as usize,
            encoding: dph.encoding,
            values: decomp.as_slice(),
        })
    } else if let Some(ref dph) = hdr.data_page_header_v2 {
        let rep_len = dph.repetition_levels_byte_length as usize;
        let def_len = dph.definition_levels_byte_length as usize;
        let prefix = rep_len + def_len;
        if body.len() < prefix {
            return Err(AsyncError::Format(format!(
                "DataPageV2 body too short: {} bytes, need {} for rep+def",
                body.len(),
                prefix
            )));
        }
        let value_bytes = &body[prefix..];
        let values: &[u8] = if dph.is_compressed && chunk_codec != CompressionCodec::Uncompressed {
            decompress_into(chunk_codec, value_bytes, decomp)?;
            decomp.as_slice()
        } else {
            value_bytes
        };
        Ok(DataPageInfo {
            num_values: dph.num_values as usize,
            encoding: dph.encoding,
            values,
        })
    } else {
        Err(AsyncError::Format(
            "data page missing both V1 and V2 header".into(),
        ))
    }
}

fn decompress_into(codec: CompressionCodec, body: &[u8], out: &mut Vec<u8>) -> Result<()> {
    match codec {
        CompressionCodec::Uncompressed => {
            out.clear();
            out.extend_from_slice(body);
            Ok(())
        }
        CompressionCodec::Snappy => decompress_snappy_into(body, out).map_err(codec_to_async),
        CompressionCodec::Zstd => decompress_zstd_into(body, out).map_err(codec_to_async),
        CompressionCodec::Gzip => decompress_gzip_into(body, out).map_err(codec_to_async),
        CompressionCodec::Brotli => decompress_brotli_into(body, out).map_err(codec_to_async),
        CompressionCodec::Lz4Raw => decompress_lz4_raw_into(body, out).map_err(codec_to_async),
        other => Err(AsyncError::Format(format!(
            "compression codec not yet wired in async façade: {other:?}"
        ))),
    }
}

fn io_to_async(e: ematix_parquet_io::IoError) -> AsyncError {
    AsyncError::Format(format!("io: {e}"))
}

fn codec_to_async(e: ematix_parquet_codec::error::CodecError) -> AsyncError {
    AsyncError::Format(format!("codec: {e}"))
}

// ============================================================
// Π.11d — async streaming Stream API
// ============================================================
//
// `read_column_*_async_stream(file, rg, col, batch_size)` returns
// a `Stream<Item = Result<Vec<T>>>` that yields decoded values in
// batches of (mostly) `batch_size` rows. Mirrors the sync
// `ematix_parquet_codec::read::ColumnBatchIter` shape, lifted into
// an async Stream so callers can `.next().await` and integrate
// with `futures::stream` adapters (map, take_while, throttle, etc.).
//
// Shape: one async GET to fetch the chunk bytes, then sync
// page-walk + decode produces an in-memory `Vec<T>` of all the
// chunk's values, which the stream then yields in slices of
// `batch_size`. This is simpler than yielding per-page (which
// would tangle Stream poll semantics with page-walker state)
// and adequate for the common consumer use case (Arrow
// RecordBatch sizing).

use futures_core::Stream;

/// Stream INT64 batches asynchronously. Internally fetches the
/// chunk once via one GET, decodes fully in memory, then yields
/// `batch_size`-sized `Vec<i64>` until exhausted. The final batch
/// may be shorter.
///
/// Pair with `futures::StreamExt` to consume: `s.next().await`,
/// `s.try_collect().await`, etc.
pub fn read_column_i64_async_stream(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> impl Stream<Item = Result<Vec<i64>>> + '_ {
    async_stream::try_stream! {
        if batch_size == 0 {
            Err(AsyncError::Format("batch_size must be > 0".into()))?;
        }
        let mut full: Vec<i64> = Vec::new();
        read_column_i64_async_into(file, row_group, column, &mut full).await?;
        let mut cursor = 0usize;
        while cursor < full.len() {
            let end = (cursor + batch_size).min(full.len());
            let batch = full[cursor..end].to_vec();
            cursor = end;
            yield batch;
        }
    }
}

/// Stream INT32 batches asynchronously.
pub fn read_column_i32_async_stream(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> impl Stream<Item = Result<Vec<i32>>> + '_ {
    async_stream::try_stream! {
        if batch_size == 0 {
            Err(AsyncError::Format("batch_size must be > 0".into()))?;
        }
        let mut full: Vec<i32> = Vec::new();
        read_column_i32_async_into(file, row_group, column, &mut full).await?;
        let mut cursor = 0usize;
        while cursor < full.len() {
            let end = (cursor + batch_size).min(full.len());
            let batch = full[cursor..end].to_vec();
            cursor = end;
            yield batch;
        }
    }
}

/// Stream DOUBLE batches asynchronously.
pub fn read_column_f64_async_stream(
    file: &AsyncParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> impl Stream<Item = Result<Vec<f64>>> + '_ {
    async_stream::try_stream! {
        if batch_size == 0 {
            Err(AsyncError::Format("batch_size must be > 0".into()))?;
        }
        let mut full: Vec<f64> = Vec::new();
        read_column_f64_async_into(file, row_group, column, &mut full).await?;
        let mut cursor = 0usize;
        while cursor < full.len() {
            let end = (cursor + batch_size).min(full.len());
            let batch = full[cursor..end].to_vec();
            cursor = end;
            yield batch;
        }
    }
}
