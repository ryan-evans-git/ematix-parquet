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

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_column_index, PageHeader};
use ematix_parquet_format::types::{CompressionCodec, Encoding, PageType};
use ematix_parquet_io::{PageWalker, ParquetFile};

use crate::compression::{
    decompress_brotli_into, decompress_gzip_into, decompress_lz4_raw_into, decompress_snappy_into,
    decompress_zstd_into,
};
use crate::dict::{decode_rle_dictionary_into, gather_dict_at_bitmap_into};
use crate::error::{CodecError, Result};
use crate::page_index::{select_pages_overlapping_i32, select_pages_overlapping_i64};
use crate::plain::{
    decode_plain_byte_array, decode_plain_f64, decode_plain_fixed_len_byte_array, decode_plain_i32,
    decode_plain_i64, decode_plain_int96, plain_sparse_decode_byte_array_into,
    plain_sparse_decode_byte_array_offsets_into, plain_sparse_decode_f64_into,
    plain_sparse_decode_i32_into, plain_sparse_decode_i64_into, Int96,
};

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<i64>`. Requires the column's physical type to be INT64.
pub fn read_column_i64(file: &ParquetFile, row_group: usize, column: usize) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    read_column_i64_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_i64` writing into a caller-provided buffer (cleared
/// then filled). Reuse the same `Vec` across calls to avoid the
/// per-read allocation; steady-state cost is whatever the chunk
/// itself requires.
pub fn read_column_i64_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<i64>,
) -> Result<()> {
    decode_chunk_into(file, row_group, column, out, |bytes| {
        decode_plain_i64(bytes)
    })
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<i32>`. Requires the column's physical type to be INT32.
pub fn read_column_i32(file: &ParquetFile, row_group: usize, column: usize) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    read_column_i32_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_i32` writing into a caller-provided buffer.
pub fn read_column_i32_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<i32>,
) -> Result<()> {
    decode_chunk_into(file, row_group, column, out, |bytes| {
        decode_plain_i32(bytes)
    })
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<f64>`. Requires the column's physical type to be DOUBLE.
pub fn read_column_f64(file: &ParquetFile, row_group: usize, column: usize) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    read_column_f64_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_f64` writing into a caller-provided buffer.
pub fn read_column_f64_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<f64>,
) -> Result<()> {
    decode_chunk_into(file, row_group, column, out, |bytes| {
        decode_plain_f64(bytes)
    })
}

// ============================================================
// Π.10a — masked-decode read façade (late materialization)
// ============================================================
//
// `read_column_*_masked_into(file, rg, col, mask, &mut out)` decodes
// only the rows where `mask` (packed bitmap, 1 bit per row) is set,
// appending matched values to `out` in row order. The dict path
// reuses Π.5/Π.9-era `gather_dict_at_bitmap_into`; the PLAIN path
// uses the new `plain_sparse_decode_*_into` primitives. Per-page
// popcount-skip drops fully-dead pages without decompression.
//
// `mask` is a packed bitmap covering the chunk's row 0..num_values
// address space. Bit `i` of byte `k` is row `8k + i`. Caller is
// responsible for sizing the mask correctly (≥ ceil(num_values/8)
// bytes) — undersized masks return `InvalidInput`.
//
// **Appends.** Unlike the `_into` variants on the allocating reads,
// these do NOT clear `out` — callers can build a contiguous output
// across multiple row-groups via a single Vec. Call `out.clear()`
// up front if you want full-replace semantics.

/// Decode INT64 only at rows where `mask` is set. Appends to `out`.
pub fn read_column_i64_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out: &mut Vec<i64>,
) -> Result<()> {
    decode_chunk_row_masked_into(
        file,
        row_group,
        column,
        mask,
        out,
        decode_plain_i64,
        plain_sparse_decode_i64_into,
    )
}

/// Decode INT32 only at rows where `mask` is set. Appends to `out`.
pub fn read_column_i32_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out: &mut Vec<i32>,
) -> Result<()> {
    decode_chunk_row_masked_into(
        file,
        row_group,
        column,
        mask,
        out,
        decode_plain_i32,
        plain_sparse_decode_i32_into,
    )
}

/// Decode DOUBLE only at rows where `mask` is set. Appends to `out`.
pub fn read_column_f64_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out: &mut Vec<f64>,
) -> Result<()> {
    decode_chunk_row_masked_into(
        file,
        row_group,
        column,
        mask,
        out,
        decode_plain_f64,
        plain_sparse_decode_f64_into,
    )
}

/// Decode BYTE_ARRAY only at rows where `mask` is set, returning
/// owned `Vec<u8>` per matched value. Appends to `out`.
///
/// For an allocation-light path use the offsets variant.
pub fn read_column_byte_array_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;

    let required_mask_bytes = total_values.div_ceil(8);
    if mask.len() < required_mask_bytes {
        return Err(CodecError::InvalidInput(format!(
            "mask too small: {} bytes for {} rows (need ≥ {})",
            mask.len(),
            total_values,
            required_mask_bytes,
        )));
    }

    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();
    let mut dict: Vec<Vec<u8>> = Vec::new();
    let mut row_cursor: usize = 0;

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                let page_n = info.num_values;
                let matched_in_page = popcount_mask_range(mask, row_cursor, row_cursor + page_n);
                if matched_in_page == 0 {
                    row_cursor += page_n;
                    if row_cursor >= total_values {
                        break;
                    }
                    continue;
                }
                match info.encoding {
                    Encoding::Plain => {
                        plain_sparse_decode_byte_array_into(
                            info.values,
                            page_n,
                            mask,
                            row_cursor,
                            out,
                        )?;
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        // gather_dict_at_bitmap_into is now T: Clone
                        // (was T: Copy in Π.9) — accepts Vec<u8>.
                        crate::dict::gather_dict_at_bitmap_into(
                            info.values,
                            page_n,
                            mask,
                            row_cursor,
                            &dict,
                            out,
                        )?;
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "encoding not yet dispatched by façade: {other:?}"
                        )));
                    }
                }
                row_cursor += page_n;
                if row_cursor >= total_values {
                    break;
                }
            }
            PageType::IndexPage => {}
        }
    }
    Ok(())
}

/// Arrow-style BYTE_ARRAY masked-decode: appends matched values'
/// bytes to `out_bytes` and pushes one offset per matched value to
/// `out_offsets`. If `out_offsets` is empty on entry, the initial
/// `0` offset is pushed automatically; otherwise this continues
/// from the existing trailing offset (multi-chunk concatenation
/// works naturally — call this per row-group with the same
/// `(out_bytes, out_offsets)` pair).
pub fn read_column_byte_array_offsets_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out_bytes: &mut Vec<u8>,
    out_offsets: &mut Vec<u32>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;

    let required_mask_bytes = total_values.div_ceil(8);
    if mask.len() < required_mask_bytes {
        return Err(CodecError::InvalidInput(format!(
            "mask too small: {} bytes for {} rows (need ≥ {})",
            mask.len(),
            total_values,
            required_mask_bytes,
        )));
    }

    if out_offsets.is_empty() {
        out_offsets.push(0);
    }

    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();
    // Dict stored as owned Vec<u8> to outlive the page-buffer scope.
    let mut dict: Vec<Vec<u8>> = Vec::new();
    let mut row_cursor: usize = 0;

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                let page_n = info.num_values;
                let matched_in_page = popcount_mask_range(mask, row_cursor, row_cursor + page_n);
                if matched_in_page == 0 {
                    row_cursor += page_n;
                    if row_cursor >= total_values {
                        break;
                    }
                    continue;
                }
                match info.encoding {
                    Encoding::Plain => {
                        plain_sparse_decode_byte_array_offsets_into(
                            info.values,
                            page_n,
                            mask,
                            row_cursor,
                            out_bytes,
                            out_offsets,
                        )?;
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        // Walk dict indices via the existing index
                        // decoder, gather sparsely into the offsets
                        // shape. The Copy-based fused gather doesn't
                        // produce offsets directly, so we expand here.
                        let indices =
                            crate::dict::decode_rle_dictionary_indices(info.values, page_n)?;
                        let mut running = *out_offsets.last().unwrap();
                        for (row, idx) in indices.iter().enumerate() {
                            let bit_pos = row_cursor + row;
                            let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
                            if bit == 1 {
                                let i = *idx as usize;
                                let v = dict.get(i).ok_or(CodecError::DictIndexOutOfRange {
                                    index: *idx,
                                    dict_size: dict.len(),
                                })?;
                                out_bytes.extend_from_slice(v);
                                running = running.checked_add(v.len() as u32).ok_or_else(|| {
                                    CodecError::InvalidInput(
                                        "byte_array masked-decode: offset overflow > u32::MAX"
                                            .into(),
                                    )
                                })?;
                                out_offsets.push(running);
                            }
                        }
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "encoding not yet dispatched by façade: {other:?}"
                        )));
                    }
                }
                row_cursor += page_n;
                if row_cursor >= total_values {
                    break;
                }
            }
            PageType::IndexPage => {}
        }
    }
    Ok(())
}

/// Build a packed bitmap of length `ceil(num_rows / 8)` bytes where
/// bit `i` of byte `k` is `1` iff `pred(8k + i)` returns true. Useful
/// when callers don't already have a bitmap-shaped mask (e.g. when
/// the predicate runs over already-decoded scalar values).
pub fn build_packed_mask(num_rows: usize, pred: impl Fn(usize) -> bool) -> Vec<u8> {
    let mut mask = vec![0u8; num_rows.div_ceil(8)];
    for row in 0..num_rows {
        if pred(row) {
            mask[row / 8] |= 1u8 << (row % 8);
        }
    }
    mask
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
    let mut out = Vec::new();
    read_column_byte_array_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_byte_array` writing into a caller-provided buffer
/// (cleared then filled). Note: the per-row `Vec<u8>` allocations
/// are still incurred — for an allocation-light path use
/// `read_column_byte_array_offsets` or its `_into` variant.
pub fn read_column_byte_array_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    out.clear();
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<Vec<u8>> = Vec::new();
    out.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        let slices = decode_plain_byte_array(info.values)?;
                        out.extend(slices.into_iter().map(|s| s.to_vec()));
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        // Vec<Vec<u8>> isn't Copy, so we go via the
                        // index decoder and gather by hand.
                        let indices = crate::dict::decode_rle_dictionary_indices(
                            info.values,
                            info.num_values,
                        )?;
                        out.reserve(info.num_values);
                        for idx in indices {
                            let v = dict.get(idx as usize).ok_or_else(|| {
                                CodecError::InvalidInput("dictionary index out of range".into())
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

    Ok(())
}

/// Dict-preserving decode of a BYTE_ARRAY column chunk.
///
/// Returns the parquet dictionary (as flat bytes + offsets) plus the
/// raw `Vec<u32>` of per-row indices — no per-row materialisation.
/// Lets Arrow consumers build a `DictionaryArray<UInt32, Utf8/Binary>`
/// directly, preserving the dict structure end-to-end so downstream
/// operators (filter / group-by / join) can stay on dict codes rather
/// than paying the gather + hash at every operator boundary.
///
/// Errors if the column chunk has no `DictionaryPage` (pure-PLAIN
/// encoding) or if any data page falls back to `PLAIN` after a
/// dictionary page — in both cases the chunk cannot be represented as
/// `(dict, indices)` and the caller must fall back to one of the
/// materialising entry points (`read_column_byte_array_offsets` /
/// `read_column_byte_array`).
#[derive(Debug, Clone, Default)]
pub struct DictPreservedColumn {
    /// Concatenated dictionary entries, addressed by `dict_offsets`.
    pub dict_bytes: Vec<u8>,
    /// Offset i,i+1 delimit entry i in `dict_bytes`. Length = dict_len + 1.
    pub dict_offsets: Vec<u32>,
    /// One u32 per row in the chunk. `indices[row]` < dict_len.
    pub indices: Vec<u32>,
}

/// See `DictPreservedColumn`. Allocates a fresh column; for hot paths
/// use `read_column_byte_array_dict_preserved_into` to reuse buffers.
pub fn read_column_byte_array_dict_preserved(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<DictPreservedColumn> {
    let mut col = DictPreservedColumn::default();
    read_column_byte_array_dict_preserved_into(
        file,
        row_group,
        column,
        &mut col.dict_bytes,
        &mut col.dict_offsets,
        &mut col.indices,
    )?;
    Ok(col)
}

/// `read_column_byte_array_dict_preserved` writing into caller-
/// provided buffers (each cleared then filled). Mirrors the
/// `_offsets_into` shape so steady-state hot paths are
/// zero-allocation once buffers have grown.
pub fn read_column_byte_array_dict_preserved_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    dict_bytes: &mut Vec<u8>,
    dict_offsets: &mut Vec<u32>,
    indices: &mut Vec<u32>,
) -> Result<()> {
    dict_bytes.clear();
    dict_offsets.clear();
    indices.clear();

    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut have_dict = false;
    dict_offsets.push(0);
    indices.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_byte_array(&decomp)?;
                let total_dict_bytes: usize = slices.iter().map(|s| s.len()).sum();
                dict_bytes.reserve(total_dict_bytes);
                dict_offsets.reserve(slices.len());
                let mut acc: u32 = 0;
                for s in &slices {
                    dict_bytes.extend_from_slice(s);
                    acc = acc.checked_add(s.len() as u32).ok_or_else(|| {
                        CodecError::InvalidInput("dict bytes exceed u32::MAX".into())
                    })?;
                    dict_offsets.push(acc);
                }
                have_dict = true;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                if !have_dict {
                    return Err(CodecError::InvalidInput(
                        "dict-preserved read: data page before dictionary (column has no DictionaryPage)".into(),
                    ));
                }
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        let mut page_indices = crate::dict::decode_rle_dictionary_indices(
                            info.values,
                            info.num_values,
                        )?;
                        // Validate every index lands inside the dict.
                        // The downstream Arrow consumer relies on this
                        // invariant; checking once here keeps the hot
                        // path on the gather side branch-free.
                        let dict_len = dict_offsets.len() - 1;
                        if let Some(bad) = page_indices.iter().find(|&&i| (i as usize) >= dict_len)
                        {
                            return Err(CodecError::InvalidInput(format!(
                                "dictionary index {bad} out of range (dict_len = {dict_len})"
                            )));
                        }
                        indices.append(&mut page_indices);
                    }
                    Encoding::Plain => {
                        return Err(CodecError::InvalidInput(
                            "dict-preserved read: data page is PLAIN-encoded (writer fell back from dict — chunk cannot be expressed as one dict + indices)".into(),
                        ));
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "dict-preserved read: data page encoding not supported: {other:?}"
                        )));
                    }
                }
            }
            PageType::IndexPage => {}
        }
        if indices.len() >= total_values {
            break;
        }
    }

    if !have_dict {
        return Err(CodecError::InvalidInput(
            "dict-preserved read: column chunk has no DictionaryPage".into(),
        ));
    }

    Ok(())
}

// ============================================================
// Π.14e — adaptive-dispatch read façade
// ============================================================
//
// `read_column_*_predicate_adaptive(file, rg, col, predicate, opts,
//  telemetry)` opens the chunk, decompresses every page, hands the
// page bodies + the dict + a mask built from `predicate` to the
// `adaptive::run_adaptive_dict_chunk` runner. The runner probes the
// first `opts.probe_pages` pages with the fused kernel; if observed
// selectivity > `opts.threshold` it commits to materialised values
// for the whole chunk, otherwise it stays on the fused bitmap.
//
// `predicate` is applied to dict entries (not row values), so it
// runs at most `dict.len()` times per chunk — typically 1-3K
// for analytic columns.
//
// **Dict-only.** These entry points require the column to be
// dict-encoded across every data page. Chunks with PLAIN-encoded
// data pages return `InvalidInput` — callers should fall back to
// `read_column_*_masked_into` for those.

use crate::adaptive::{
    run_adaptive_dict_chunk, AdaptiveChunkOutput, AdaptiveDictPredicate, AdaptiveDispatchOptions,
    AdaptivePageInput, SelectivityProbe,
};
use crate::dict::build_dict_predicate_mask;

/// Pull the chunk's dict + every decompressed data-page body into
/// owned buffers, in order. Returns `(dict, pages, bit_width)`.
///
/// All pages must be dict-encoded — `PLAIN` data pages return
/// `InvalidInput`. The runner needs every body live at once so it
/// can re-decode the probed pages on a `Materialized` dispatch.
fn pull_dict_chunk<T, F>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    decode_dict_plain: F,
) -> Result<(Vec<T>, Vec<(usize, Vec<u8>)>, u8)>
where
    F: Fn(&[u8]) -> Result<Vec<T>>,
{
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut dict_decoded: Option<Vec<T>> = None;
    let mut pages: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut rows_collected: usize = 0;
    let mut bit_width: u8 = 0;

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                let mut decomp = Vec::new();
                decompress_into(codec, body, &mut decomp)?;
                dict_decoded = Some(decode_dict_plain(&decomp)?);
            }
            PageType::DataPage | PageType::DataPageV2 => {
                if dict_decoded.is_none() {
                    return Err(CodecError::InvalidInput(
                        "adaptive read: data page before dictionary (column has no DictionaryPage)"
                            .into(),
                    ));
                }
                let mut owned = Vec::new();
                let info = data_page_view(&hdr, body, codec, &mut owned)?;
                match info.encoding {
                    Encoding::RleDictionary | Encoding::PlainDictionary => {}
                    Encoding::Plain => {
                        return Err(CodecError::InvalidInput(
                            "adaptive read: data page is PLAIN-encoded (writer fell back from dict — use read_column_*_masked_into instead)".into(),
                        ));
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "adaptive read: data page encoding not supported: {other:?}"
                        )));
                    }
                }
                debug_assert!(!info.values.is_empty(), "dict-encoded page body empty");
                if bit_width == 0 {
                    bit_width = info.values[0];
                }
                // info.values borrows from `owned` (V1) or `body` (V2).
                // Promote to a standalone owned Vec so the runner can
                // hold every page body live simultaneously.
                pages.push((info.num_values, info.values.to_vec()));
                rows_collected += info.num_values;
            }
            PageType::IndexPage => {}
        }
        if rows_collected >= total_values {
            break;
        }
    }

    let dict = dict_decoded.ok_or_else(|| {
        CodecError::InvalidInput("adaptive read: column chunk has no DictionaryPage".into())
    })?;
    Ok((dict, pages, bit_width))
}

fn run_facade<T: Copy, F, P>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    predicate: P,
    opts: AdaptiveDispatchOptions,
    telemetry: Option<&mut dyn FnMut(SelectivityProbe)>,
    decode_dict_plain: F,
) -> Result<AdaptiveChunkOutput<T>>
where
    F: Fn(&[u8]) -> Result<Vec<T>>,
    P: Fn(&T) -> bool,
{
    let (dict, pages, bit_width) = pull_dict_chunk(file, row_group, column, decode_dict_plain)?;
    let dict_mask = build_dict_predicate_mask(&dict, bit_width, predicate)?;
    let cfg = AdaptiveDictPredicate {
        dict_mask,
        threshold: opts.threshold,
        probe_pages: opts.probe_pages,
    };
    let inputs: Vec<AdaptivePageInput<'_>> = pages
        .iter()
        .map(|(n, b)| AdaptivePageInput {
            body: b.as_slice(),
            num_values: *n,
        })
        .collect();
    run_adaptive_dict_chunk::<T>(&inputs, &dict, &cfg, telemetry)
}

/// Adaptive predicate dispatch for an INT32 column.
///
/// Probes the first `opts.probe_pages` pages with the fused
/// `decode_rle_dictionary_predicate_bitmap` kernel; if observed
/// selectivity > `opts.threshold` returns
/// `AdaptiveOutputKind::Values(Vec<i32>)`, else returns
/// `AdaptiveOutputKind::Bitmap { bitmap, set_bits }`.
///
/// `telemetry`, if `Some`, is called once with the per-chunk
/// `SelectivityProbe` summarising the dispatch decision.
pub fn read_column_i32_predicate_adaptive<P: Fn(&i32) -> bool>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    predicate: P,
    opts: AdaptiveDispatchOptions,
    telemetry: Option<&mut dyn FnMut(SelectivityProbe)>,
) -> Result<AdaptiveChunkOutput<i32>> {
    run_facade::<i32, _, _>(
        file,
        row_group,
        column,
        predicate,
        opts,
        telemetry,
        decode_plain_i32,
    )
}

/// Adaptive predicate dispatch for an INT64 column. See
/// `read_column_i32_predicate_adaptive` for the full contract.
pub fn read_column_i64_predicate_adaptive<P: Fn(&i64) -> bool>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    predicate: P,
    opts: AdaptiveDispatchOptions,
    telemetry: Option<&mut dyn FnMut(SelectivityProbe)>,
) -> Result<AdaptiveChunkOutput<i64>> {
    run_facade::<i64, _, _>(
        file,
        row_group,
        column,
        predicate,
        opts,
        telemetry,
        decode_plain_i64,
    )
}

/// Adaptive predicate dispatch for a DOUBLE column. See
/// `read_column_i32_predicate_adaptive` for the full contract.
pub fn read_column_f64_predicate_adaptive<P: Fn(&f64) -> bool>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    predicate: P,
    opts: AdaptiveDispatchOptions,
    telemetry: Option<&mut dyn FnMut(SelectivityProbe)>,
) -> Result<AdaptiveChunkOutput<f64>> {
    run_facade::<f64, _, _>(
        file,
        row_group,
        column,
        predicate,
        opts,
        telemetry,
        decode_plain_f64,
    )
}

/// Read a BYTE_ARRAY column into Arrow-style flat bytes + offsets.
///
/// Returns `(values, offsets)` where row `i` is the byte slice
/// `values[offsets[i] as usize .. offsets[i + 1] as usize]`. Offsets
/// has length `num_rows + 1` (the trailing offset is the total
/// values length, matching Arrow's BinaryArray convention).
///
/// This is the zero-allocation-per-row alternative to
/// `read_column_byte_array`. For low-cardinality dict-encoded columns
/// (e.g. l_returnflag with 3 distinct one-byte values × 1M rows),
/// the per-row allocation in `Vec<Vec<u8>>.push(dict[i].clone())` is
/// the dominant cost; this entry point amortises into a single
/// growing `Vec<u8>`.
///
/// `u32` offsets cap a single chunk at 4 GiB of decoded byte_array
/// content — adequate for any reasonable parquet column. Use
/// `read_column_byte_array` if you need owned per-row Vecs.
pub fn read_column_byte_array_offsets(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<(Vec<u8>, Vec<u32>)> {
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    read_column_byte_array_offsets_into(file, row_group, column, &mut bytes, &mut offsets)?;
    Ok((bytes, offsets))
}

/// `read_column_byte_array_offsets` writing into caller-provided
/// buffers (cleared then filled). The bytes buffer's capacity is
/// reused across calls; for steady-state hot paths this is a
/// zero-allocation read once both buffers have grown to the
/// largest chunk size.
pub fn read_column_byte_array_offsets_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out_bytes: &mut Vec<u8>,
    out_offsets: &mut Vec<u32>,
) -> Result<()> {
    out_bytes.clear();
    out_offsets.clear();

    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    // Dictionary as flat bytes + per-entry offsets. dict_offsets has
    // length dict_len + 1; entry i is dict_bytes[dict_offsets[i] .. dict_offsets[i+1]].
    let mut dict_bytes: Vec<u8> = Vec::new();
    let mut dict_offsets: Vec<u32> = vec![0];

    out_offsets.reserve(total_values + 1);
    out_offsets.push(0);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                // Flatten the dict slices into our own bytes+offsets.
                let slices = decode_plain_byte_array(&decomp)?;
                let total_dict_bytes: usize = slices.iter().map(|s| s.len()).sum();
                dict_bytes.clear();
                dict_bytes.reserve(total_dict_bytes);
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
                        // PLAIN body: u32_le length + bytes, repeated.
                        // Walk it inline rather than going through
                        // decode_plain_byte_array (which allocates a
                        // Vec<&[u8]>).
                        let body = info.values;
                        let mut i = 0usize;
                        let mut emitted = 0usize;
                        let cap_target = info.num_values;
                        out_bytes.reserve(body.len()); // upper bound
                        out_offsets.reserve(cap_target);
                        let mut acc: u32 = *out_offsets.last().unwrap();
                        while i < body.len() && emitted < cap_target {
                            if i + 4 > body.len() {
                                return Err(CodecError::InvalidInput(
                                    "PLAIN byte_array: truncated length prefix".into(),
                                ));
                            }
                            let len =
                                u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) as usize;
                            i += 4;
                            if i + len > body.len() {
                                return Err(CodecError::InvalidInput(
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
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        let indices = crate::dict::decode_rle_dictionary_indices(
                            info.values,
                            info.num_values,
                        )?;
                        // Pre-compute total bytes needed to do a
                        // single allocation grow.
                        let dict_len = dict_offsets.len() - 1;
                        let mut total_bytes_needed: usize = 0;
                        for &idx in &indices {
                            let i = idx as usize;
                            if i >= dict_len {
                                return Err(CodecError::InvalidInput(
                                    "dictionary index out of range".into(),
                                ));
                            }
                            let start = dict_offsets[i] as usize;
                            let end = dict_offsets[i + 1] as usize;
                            total_bytes_needed += end - start;
                        }
                        out_bytes.reserve(total_bytes_needed);
                        out_offsets.reserve(indices.len());

                        // Hot loop: bounds were validated in the
                        // pre-pass above, so we can use unchecked
                        // accesses here. SAFETY: every idx < dict_len
                        // by the loop above.
                        let mut acc: u32 = *out_offsets.last().unwrap();
                        let dict_offsets_ptr = dict_offsets.as_ptr();
                        let dict_bytes_ptr = dict_bytes.as_ptr();
                        unsafe {
                            for &idx in &indices {
                                let i = idx as usize;
                                let start = *dict_offsets_ptr.add(i) as usize;
                                let end = *dict_offsets_ptr.add(i + 1) as usize;
                                let len = end - start;
                                let src = dict_bytes_ptr.add(start);
                                let dst = out_bytes.as_mut_ptr().add(out_bytes.len());
                                std::ptr::copy_nonoverlapping(src, dst, len);
                                out_bytes.set_len(out_bytes.len() + len);
                                acc += len as u32;
                                out_offsets.push(acc);
                            }
                        }
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
        if out_offsets.len() > total_values {
            break;
        }
    }

    Ok(())
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<Int96>`. Requires the column's physical type to be INT96.
///
/// INT96 is mostly used for legacy nanosecond timestamps (pre-2018
/// Hive output). Modern files use INT64 with a `Timestamp` logical
/// type instead — but reading old files is still a load-bearing
/// requirement.
pub fn read_column_int96(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<Int96>> {
    let mut out = Vec::new();
    read_column_int96_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_int96` writing into a caller-provided buffer.
pub fn read_column_int96_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<Int96>,
) -> Result<()> {
    decode_chunk_into(file, row_group, column, out, |bytes| {
        decode_plain_int96(bytes)
    })
}

/// Read the entire column chunk at (`row_group`, `column`) into a
/// `Vec<Vec<u8>>`. Requires the column's physical type to be
/// FIXED_LEN_BYTE_ARRAY. The fixed value width is taken from the
/// schema's `type_length`.
///
/// Common shapes: UUIDs (`type_length`=16), DECIMAL(N, S) where the
/// binary form is fixed-width, opaque BLOBs.
pub fn read_column_flba(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    read_column_flba_into(file, row_group, column, &mut out)?;
    Ok(out)
}

/// `read_column_flba` writing into a caller-provided buffer
/// (cleared then filled). Per-row `Vec<u8>` allocations still
/// happen — there's no offsets variant for FLBA today.
pub fn read_column_flba_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    out.clear();
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let type_length = type_length_for(file, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    // FLBA values are owned per row in the output; the dictionary,
    // if present, is also owned (Vec<Vec<u8>>) so values can be
    // copied out by index without holding onto page buffers.
    let mut dict: Vec<Vec<u8>> = Vec::new();
    out.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                let slices = decode_plain_fixed_len_byte_array(&decomp, type_length)?;
                dict = slices.into_iter().map(|s| s.to_vec()).collect();
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        let slices = decode_plain_fixed_len_byte_array(info.values, type_length)?;
                        out.extend(slices.into_iter().map(|s| s.to_vec()));
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        let indices = crate::dict::decode_rle_dictionary_indices(
                            info.values,
                            info.num_values,
                        )?;
                        out.reserve(info.num_values);
                        for idx in indices {
                            let v = dict.get(idx as usize).ok_or_else(|| {
                                CodecError::InvalidInput("dictionary index out of range".into())
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
            PageType::IndexPage => {}
        }
        if out.len() >= total_values {
            break;
        }
    }
    Ok(())
}

// ---- Page-index pruning entry points (Π.5a) -------------------------
//
// `*_with_range(file, rg, col, lo, hi)` returns values from data
// pages whose [page_min, page_max] intersect [lo, hi]. Pages that
// can't possibly contain a value in [lo, hi] are skipped without
// being decompressed or decoded.
//
// Caller still applies the final value-level predicate — pruning is a
// page-granularity optimisation, not row-granularity. The win is the
// pages we never touched; the returned vec may include some
// non-matching values from kept pages.
//
// If the column has no `ColumnIndex` in the footer, the entry point
// falls back to a full chunk read (same result as `read_column_*`).

/// `read_column_i64` with page-index pruning by `[lo, hi]` (inclusive).
pub fn read_column_i64_with_range(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i64,
    hi: i64,
) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    read_column_i64_with_range_into(file, row_group, column, lo, hi, &mut out)?;
    Ok(out)
}

/// `read_column_i64_with_range` writing into a caller-provided buffer.
pub fn read_column_i64_with_range_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i64,
    hi: i64,
    out: &mut Vec<i64>,
) -> Result<()> {
    let mask = page_mask_i64(file, row_group, column, lo, hi)?;
    decode_chunk_masked_into(file, row_group, column, mask, out, |bytes| {
        decode_plain_i64(bytes)
    })
}

/// `read_column_i32` with page-index pruning by `[lo, hi]` (inclusive).
pub fn read_column_i32_with_range(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i32,
    hi: i32,
) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    read_column_i32_with_range_into(file, row_group, column, lo, hi, &mut out)?;
    Ok(out)
}

/// `read_column_i32_with_range` writing into a caller-provided buffer.
pub fn read_column_i32_with_range_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i32,
    hi: i32,
    out: &mut Vec<i32>,
) -> Result<()> {
    let mask = page_mask_i32(file, row_group, column, lo, hi)?;
    decode_chunk_masked_into(file, row_group, column, mask, out, |bytes| {
        decode_plain_i32(bytes)
    })
}

fn page_mask_i64(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i64,
    hi: i64,
) -> Result<Option<Vec<bool>>> {
    let Some(ci_bytes) = read_column_index_bytes(file, row_group, column)? else {
        return Ok(None);
    };
    let mut cur = Cursor::new(&ci_bytes);
    let ci = read_column_index(&mut cur).map_err(format_to_codec)?;
    Ok(Some(select_pages_overlapping_i64(&ci, lo, hi)?))
}

fn page_mask_i32(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    lo: i32,
    hi: i32,
) -> Result<Option<Vec<bool>>> {
    let Some(ci_bytes) = read_column_index_bytes(file, row_group, column)? else {
        return Ok(None);
    };
    let mut cur = Cursor::new(&ci_bytes);
    let ci = read_column_index(&mut cur).map_err(format_to_codec)?;
    Ok(Some(select_pages_overlapping_i32(&ci, lo, hi)?))
}

fn read_column_index_bytes(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<Option<Vec<u8>>> {
    let md = file.metadata().map_err(io_to_codec)?;
    let rg = md
        .row_groups
        .get(row_group)
        .ok_or_else(|| CodecError::InvalidInput(format!("row group {row_group} out of range")))?;
    let col = rg
        .columns
        .get(column)
        .ok_or_else(|| CodecError::InvalidInput(format!("column {column} out of range")))?;
    match (col.column_index_offset, col.column_index_length) {
        (Some(off), Some(len)) => {
            let bytes = file
                .read_range(off as u64, len as u64)
                .map_err(io_to_codec)?;
            Ok(Some(bytes))
        }
        _ => Ok(None),
    }
}

// ---- internals -------------------------------------------------------------

/// Same as `decode_chunk_into` but skips data pages whose `page_mask`
/// entry is `false`. `page_mask` is indexed by data-page ordinal
/// (dictionary pages are not counted). `None` means "no pruning"
/// — equivalent to `decode_chunk_into`.
fn decode_chunk_masked_into<T: Copy>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    page_mask: Option<Vec<bool>>,
    out: &mut Vec<T>,
    decode_plain: impl Fn(&[u8]) -> Result<Vec<T>>,
) -> Result<()> {
    out.clear();
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<T> = Vec::new();
    out.reserve(total_values);
    let mut data_page_ix: usize = 0;
    let total_kept_data_pages = page_mask.as_ref().map(|m| m.iter().filter(|b| **b).count());

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                dict = decode_plain(&decomp)?;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let keep = page_mask
                    .as_ref()
                    .map_or(true, |m| m.get(data_page_ix).copied().unwrap_or(true));
                data_page_ix += 1;
                if !keep {
                    continue;
                }
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        out.extend(decode_plain(info.values)?);
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        decode_rle_dictionary_into(info.values, &dict, info.num_values, out)?;
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
        // Without pruning, exit early once we've decoded the chunk's
        // total. With pruning, walker exhaustion is the natural end.
        if total_kept_data_pages.is_none() && out.len() >= total_values {
            break;
        }
    }
    Ok(())
}

fn format_to_codec(e: ematix_parquet_format::error::FormatError) -> CodecError {
    CodecError::InvalidInput(format!("format: {e}"))
}

/// Per-data-page view that abstracts V1 vs V2 layout differences.
struct DataPageInfo<'a> {
    num_values: usize,
    encoding: Encoding,
    /// Decompressed value bytes. For V1, the entire decompressed
    /// page body. For V2 with rep+def levels, the slice past the
    /// (uncompressed) rep+def prefix, then decompressed if the
    /// header's `is_compressed` flag is set.
    values: &'a [u8],
}

/// Extract the value bytes from a data page (V1 or V2). Decompression
/// happens into `decomp` (which the caller owns and reuses across
/// pages); the returned `DataPageInfo` borrows from either `decomp`
/// or `body` depending on which path was taken.
fn data_page_view<'a>(
    hdr: &'a PageHeader<'a>,
    body: &'a [u8],
    chunk_codec: CompressionCodec,
    decomp: &'a mut Vec<u8>,
) -> Result<DataPageInfo<'a>> {
    if let Some(ref dph) = hdr.data_page_header {
        // ---- DataPageV1: whole body is one compressed unit ----
        decompress_into(chunk_codec, body, decomp)?;
        Ok(DataPageInfo {
            num_values: dph.num_values as usize,
            encoding: dph.encoding,
            values: decomp.as_slice(),
        })
    } else if let Some(ref dph) = hdr.data_page_header_v2 {
        // ---- DataPageV2: rep + def prefixes are uncompressed ----
        let rep_len = dph.repetition_levels_byte_length as usize;
        let def_len = dph.definition_levels_byte_length as usize;
        let prefix = rep_len + def_len;
        if body.len() < prefix {
            return Err(CodecError::InvalidInput(format!(
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
        Err(CodecError::InvalidInput(
            "data page missing both V1 and V2 header".into(),
        ))
    }
}

fn type_length_for(file: &ParquetFile, column: usize) -> Result<i32> {
    let md = file.metadata().map_err(io_to_codec)?;
    // Schema is depth-first; column N is the (N+1)-th leaf — but in
    // practice for flat REQUIRED columns the schema has [root, leaf0,
    // leaf1, ...]. Walk leaves to find the Nth.
    let mut leaf_seen = 0usize;
    for se in &md.schema {
        if se.column_type.is_some() {
            if leaf_seen == column {
                return se.type_length.ok_or_else(|| {
                    CodecError::InvalidInput(
                        "FIXED_LEN_BYTE_ARRAY column missing type_length".into(),
                    )
                });
            }
            leaf_seen += 1;
        }
    }
    Err(CodecError::InvalidInput(format!(
        "column {column} not found in schema"
    )))
}

/// Generic chunk-decode for `Copy` scalar types. `decode_plain` knows
/// how to turn a bytes slice into a `Vec<T>` via the PLAIN encoding.
/// Writes into the caller's `out` buffer (cleared then filled).
fn decode_chunk_into<T: Copy>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    out: &mut Vec<T>,
    decode_plain: impl Fn(&[u8]) -> Result<Vec<T>>,
) -> Result<()> {
    out.clear();
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();

    let mut dict: Vec<T> = Vec::new();
    out.reserve(total_values);

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                dict = decode_plain(&decomp)?;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                match info.encoding {
                    Encoding::Plain => {
                        out.extend(decode_plain(info.values)?);
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        decode_rle_dictionary_into(info.values, &dict, info.num_values, out)?;
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

    Ok(())
}

/// Row-masked chunk decode for scalar `Copy` types. Walks pages
/// page-by-page; for each data page:
///
/// 1. Popcount the mask over the page's row range. If zero, skip
///    the page entirely (no decompression).
/// 2. Otherwise, dispatch to PLAIN sparse-decode or
///    `gather_dict_at_bitmap_into` depending on encoding.
///
/// Appends to `out` — does NOT clear.
///
/// `plain_full_decode` is used for the DictionaryPage itself
/// (we need every dict entry). `plain_sparse_decode` is the
/// per-page-data sparse path.
fn decode_chunk_row_masked_into<T: Copy>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],
    out: &mut Vec<T>,
    plain_full_decode: impl Fn(&[u8]) -> Result<Vec<T>>,
    plain_sparse_decode: impl Fn(&[u8], usize, &[u8], usize, &mut Vec<T>) -> Result<()>,
) -> Result<()> {
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;

    let required_mask_bytes = total_values.div_ceil(8);
    if mask.len() < required_mask_bytes {
        return Err(CodecError::InvalidInput(format!(
            "mask too small: {} bytes for {} rows (need ≥ {})",
            mask.len(),
            total_values,
            required_mask_bytes,
        )));
    }

    let mut walker = PageWalker::new(&chunk_bytes);
    let mut decomp: Vec<u8> = Vec::new();
    let mut dict: Vec<T> = Vec::new();
    let mut row_cursor: usize = 0;

    while let Some((hdr, body)) = walker.next_page().map_err(io_to_codec)? {
        match hdr.page_type {
            PageType::DictionaryPage => {
                decompress_into(codec, body, &mut decomp)?;
                dict = plain_full_decode(&decomp)?;
            }
            PageType::DataPage | PageType::DataPageV2 => {
                let info = data_page_view(&hdr, body, codec, &mut decomp)?;
                let page_n = info.num_values;
                // Per-page popcount: if zero, skip decode entirely.
                let matched_in_page = popcount_mask_range(mask, row_cursor, row_cursor + page_n);
                if matched_in_page == 0 {
                    row_cursor += page_n;
                    if row_cursor >= total_values {
                        break;
                    }
                    continue;
                }
                match info.encoding {
                    Encoding::Plain => {
                        plain_sparse_decode(info.values, page_n, mask, row_cursor, out)?;
                    }
                    Encoding::RleDictionary | Encoding::PlainDictionary => {
                        if dict.is_empty() {
                            return Err(CodecError::InvalidInput(
                                "dict-encoded data page before dictionary".into(),
                            ));
                        }
                        gather_dict_at_bitmap_into(
                            info.values,
                            page_n,
                            mask,
                            row_cursor,
                            &dict,
                            out,
                        )?;
                    }
                    other => {
                        return Err(CodecError::Unsupported(format!(
                            "encoding not yet dispatched by façade: {other:?}"
                        )));
                    }
                }
                row_cursor += page_n;
                if row_cursor >= total_values {
                    break;
                }
            }
            PageType::IndexPage => {}
        }
    }
    Ok(())
}

/// Count set bits in `bitmap[start_bit..end_bit]`. Used by per-page
/// skip in row-masked decode. Local copy of `dict::popcount_range`
/// (not exported there).
fn popcount_mask_range(bitmap: &[u8], start_bit: usize, end_bit: usize) -> usize {
    if start_bit >= end_bit {
        return 0;
    }
    let mut bit = start_bit;
    let mut total: usize = 0;
    while bit < end_bit && bit % 8 != 0 {
        if (bitmap[bit / 8] >> (bit % 8)) & 1 == 1 {
            total += 1;
        }
        bit += 1;
    }
    while bit + 8 <= end_bit {
        total += bitmap[bit / 8].count_ones() as usize;
        bit += 8;
    }
    while bit < end_bit {
        if (bitmap[bit / 8] >> (bit % 8)) & 1 == 1 {
            total += 1;
        }
        bit += 1;
    }
    total
}

/// Pull the raw column-chunk bytes (compressed pages, dictionary
/// page first if present) plus the total value count and codec.
fn read_chunk_raw(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
) -> Result<(Vec<u8>, usize, CompressionCodec)> {
    let md = file.metadata().map_err(io_to_codec)?;
    let rg = md
        .row_groups
        .get(row_group)
        .ok_or_else(|| CodecError::InvalidInput(format!("row group {row_group} out of range")))?;
    let col = rg
        .columns
        .get(column)
        .ok_or_else(|| CodecError::InvalidInput(format!("column {column} out of range")))?;
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

// ============================================================
// Π.9c — Streaming / batched decode API
// ============================================================
//
// `ColumnBatchIter<T, F>` walks a column chunk page-by-page and
// yields `Vec<T>` batches of (mostly) `batch_size` rows. Lets ematix-
// flow pipeline (decode batch N+1 while the engine consumes batch N)
// and bounds working-set memory for huge row groups.
//
// Memory shape: the chunk's compressed bytes live for the lifetime of
// the iterator (single allocation). Decoded values land in `carry`,
// which holds at most `max(batch_size, page_values)` between emits —
// the upper bound is one page's worth + the batch, since we decode a
// page at a time and emit `batch_size` slices from it. For typical
// parquet pages (~20K values) and batch_size in the 1K–64K range,
// this is near-optimal.
//
// Dispatch matches `decode_chunk_into`: PLAIN data pages decode via
// the per-type `decode_plain` callback; dict pages build the chunk's
// dictionary; RLE_DICTIONARY data pages call `decode_rle_dictionary_into`.

/// Iterator yielding `Vec<T>` batches from a single column chunk.
/// Constructed via `read_column_*_batches` helpers below.
pub struct ColumnBatchIter<T: Copy, F: Fn(&[u8]) -> Result<Vec<T>>> {
    chunk_bytes: Vec<u8>,
    walker_pos: usize,
    codec: CompressionCodec,
    total_values: usize,
    emitted_to_carry: usize,
    batch_size: usize,
    dict: Vec<T>,
    decomp: Vec<u8>,
    carry: Vec<T>,
    carry_pos: usize,
    decode_plain: F,
    finished: bool,
}

impl<T: Copy, F: Fn(&[u8]) -> Result<Vec<T>>> ColumnBatchIter<T, F> {
    /// Pull and decode exactly one page from the chunk, appending
    /// the decoded values to `carry`. Returns `Ok(true)` if a page
    /// was consumed, `Ok(false)` if the chunk is exhausted.
    fn fill_one_page(&mut self) -> Result<bool> {
        if self.emitted_to_carry >= self.total_values {
            return Ok(false);
        }
        let mut walker = PageWalker::new(&self.chunk_bytes[self.walker_pos..]);
        loop {
            let pair = walker.next_page().map_err(io_to_codec)?;
            let (hdr, body) = match pair {
                Some(p) => p,
                None => {
                    self.walker_pos = self.chunk_bytes.len();
                    return Ok(false);
                }
            };
            let consumed = walker.position();
            self.walker_pos += consumed;

            match hdr.page_type {
                PageType::DictionaryPage => {
                    decompress_into(self.codec, body, &mut self.decomp)?;
                    self.dict = (self.decode_plain)(&self.decomp)?;
                    // Loop to find a data page.
                    walker = PageWalker::new(&self.chunk_bytes[self.walker_pos..]);
                    continue;
                }
                PageType::DataPage | PageType::DataPageV2 => {
                    let info = data_page_view(&hdr, body, self.codec, &mut self.decomp)?;
                    let before = self.carry.len();
                    match info.encoding {
                        Encoding::Plain => {
                            self.carry.extend((self.decode_plain)(info.values)?);
                        }
                        Encoding::RleDictionary | Encoding::PlainDictionary => {
                            if self.dict.is_empty() {
                                return Err(CodecError::InvalidInput(
                                    "dict-encoded data page before dictionary".into(),
                                ));
                            }
                            decode_rle_dictionary_into(
                                info.values,
                                &self.dict,
                                info.num_values,
                                &mut self.carry,
                            )?;
                        }
                        other => {
                            return Err(CodecError::Unsupported(format!(
                                "encoding not yet dispatched by façade: {other:?}"
                            )));
                        }
                    }
                    let added = self.carry.len() - before;
                    self.emitted_to_carry += added;
                    return Ok(true);
                }
                PageType::IndexPage => {
                    walker = PageWalker::new(&self.chunk_bytes[self.walker_pos..]);
                    continue;
                }
            }
        }
    }

    fn drain_one_batch(&mut self) -> Option<Result<Vec<T>>> {
        if self.finished {
            return None;
        }
        // Pull pages until we have enough for a batch, or the chunk is
        // exhausted.
        while self.carry.len() - self.carry_pos < self.batch_size
            && self.emitted_to_carry < self.total_values
        {
            match self.fill_one_page() {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
        let available = self.carry.len() - self.carry_pos;
        if available == 0 {
            self.finished = true;
            return None;
        }
        let n = available.min(self.batch_size);
        let batch = self.carry[self.carry_pos..self.carry_pos + n].to_vec();
        self.carry_pos += n;
        // Compact when fully drained to keep memory bounded.
        if self.carry_pos == self.carry.len() {
            self.carry.clear();
            self.carry_pos = 0;
        }
        Some(Ok(batch))
    }
}

impl<T: Copy, F: Fn(&[u8]) -> Result<Vec<T>>> Iterator for ColumnBatchIter<T, F> {
    type Item = Result<Vec<T>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.drain_one_batch()
    }
}

fn batch_iter_new<T: Copy, F: Fn(&[u8]) -> Result<Vec<T>>>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
    decode_plain: F,
) -> Result<ColumnBatchIter<T, F>> {
    if batch_size == 0 {
        return Err(CodecError::InvalidInput("batch_size must be > 0".into()));
    }
    let (chunk_bytes, total_values, codec) = read_chunk_raw(file, row_group, column)?;
    Ok(ColumnBatchIter {
        chunk_bytes,
        walker_pos: 0,
        codec,
        total_values,
        emitted_to_carry: 0,
        batch_size,
        dict: Vec::new(),
        decomp: Vec::new(),
        carry: Vec::with_capacity(batch_size),
        carry_pos: 0,
        decode_plain,
        finished: false,
    })
}

/// Stream INT64 batches of (mostly) `batch_size` rows from a column
/// chunk. The final batch may be shorter.
pub fn read_column_i64_batches(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> Result<ColumnBatchIter<i64, impl Fn(&[u8]) -> Result<Vec<i64>>>> {
    batch_iter_new(file, row_group, column, batch_size, |bytes| {
        decode_plain_i64(bytes)
    })
}

/// Stream INT32 batches.
pub fn read_column_i32_batches(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> Result<ColumnBatchIter<i32, impl Fn(&[u8]) -> Result<Vec<i32>>>> {
    batch_iter_new(file, row_group, column, batch_size, |bytes| {
        decode_plain_i32(bytes)
    })
}

/// Stream DOUBLE batches.
pub fn read_column_f64_batches(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    batch_size: usize,
) -> Result<ColumnBatchIter<f64, impl Fn(&[u8]) -> Result<Vec<f64>>>> {
    batch_iter_new(file, row_group, column, batch_size, |bytes| {
        decode_plain_f64(bytes)
    })
}

// Note: BYTE_ARRAY batched API is not provided in this iteration.
// `Vec<u8>` is not `Copy`, and the dict-encoded path needs a separate
// index-then-gather-then-clone strategy. Callers needing chunked
// byte_array can use `read_column_byte_array_offsets_into` with their
// own slicing for now.
