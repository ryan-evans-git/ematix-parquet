//! Per-page row layout helper for a column chunk.
//!
//! The Parquet writer in this codec does not emit `OffsetIndex` (the
//! native per-page first-row-index table), so the sidecar builder
//! and reader re-derive it from the column-chunk page headers. The
//! walk is cheap: page headers are uncompressed thrift and tiny
//! (~50–200 bytes each), so an O(num_pages) header read is
//! milliseconds even for huge chunks. **No body decompression
//! happens here.**
//!
//! The same walker shape is reused by Π.18+ when we widen to other
//! physical types — page layout is encoding-agnostic.

use ematix_parquet_format::metadata::PageHeader;
use ematix_parquet_format::types::PageType;
use ematix_parquet_io::{PageWalker, ParquetFile};

use crate::error::{CodecError, Result};

/// One data-page-shaped tuple emitted by [`walk_data_pages`].
#[derive(Debug, Clone, Copy)]
pub struct DataPageLayout {
    /// Zero-based ordinal among **data** pages in this chunk. The
    /// dictionary page (if any) is not counted.
    pub page_idx: u32,
    /// Row index, within the row group, of the first row in this
    /// page. Equal to the sum of `num_values` of every preceding
    /// data page in the same chunk.
    pub first_row: usize,
    /// `num_values` from the page header — V1 + V2 alike. For
    /// REQUIRED non-nested columns (all current TPC-H reference
    /// shapes), this equals the row count.
    pub num_values: usize,
}

/// Walk the data pages of one column chunk and emit a [`DataPageLayout`]
/// per data page. Dictionary pages are skipped (their `num_values`
/// is the dictionary cardinality, which is not a row count).
///
/// `visit` is called once per data page in ordinal order. Errors
/// returned from `visit` propagate.
///
/// I/O: one range read for the whole compressed chunk, then header
/// walk over the same bytes. Bodies are never decompressed.
pub fn walk_data_pages<F>(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(DataPageLayout) -> Result<()>,
{
    let md = file
        .metadata()
        .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
    let rg = md
        .row_groups
        .get(row_group)
        .ok_or_else(|| CodecError::InvalidInput(format!("row group {row_group} out of range")))?;
    let col = rg.columns.get(column).ok_or_else(|| {
        CodecError::InvalidInput(format!("column {column} out of range in rg {row_group}"))
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
    let bytes = file
        .read_range(start, length)
        .map_err(|e| CodecError::InvalidInput(format!("read chunk bytes: {e}")))?;

    let mut walker = PageWalker::new(&bytes);
    let mut page_idx: u32 = 0;
    let mut first_row: usize = 0;

    while let Some((hdr, _body)) = walker
        .next_page()
        .map_err(|e| CodecError::InvalidInput(format!("walk page: {e}")))?
    {
        match hdr.page_type {
            PageType::DataPage | PageType::DataPageV2 => {
                let n = data_page_num_values(&hdr)?;
                visit(DataPageLayout {
                    page_idx,
                    first_row,
                    num_values: n,
                })?;
                first_row += n;
                page_idx += 1;
            }
            PageType::DictionaryPage | PageType::IndexPage => {
                // Dict / index pages don't contribute rows. Skip.
            }
        }
    }
    Ok(())
}

/// Extract `num_values` from a data-page header (V1 or V2). Mirror
/// of the private helper in `read.rs`; duplicated here so the index
/// module stays self-contained and `read.rs` doesn't need a public
/// surface change.
fn data_page_num_values(hdr: &PageHeader<'_>) -> Result<usize> {
    if let Some(ref dph) = hdr.data_page_header {
        Ok(dph.num_values as usize)
    } else if let Some(ref dph) = hdr.data_page_header_v2 {
        Ok(dph.num_values as usize)
    } else {
        Err(CodecError::InvalidInput(
            "data page missing both V1 and V2 header".into(),
        ))
    }
}
