//! Sidecar-index builder. Π.17b ships **sorted INT64** only; Π.18
//! widens to INT32 + BYTE_ARRAY by adding sibling `write_sorted_i32`
//! / `write_sorted_byte_array` entry points that share the same
//! per-page → packed-bitmap pipeline.
//!
//! ## Algorithm
//!
//! For each (row_group, data_page) in the source column:
//! 1. Decode the page's values once.
//! 2. Bucket each value into a `HashMap<(value, rg, page), Vec<row_within_page>>`.
//!    Same value seen multiple times in the same page accumulates
//!    row positions in the same bucket.
//! 3. After the walk, sort the buckets by `(value, rg, page)` and
//!    convert each `Vec<row_within_page>` to a packed bitmap sized
//!    to the page's `num_values`.
//! 4. Emit a parquet file with schema
//!    `(value INT64, target_rg INT32, target_page INT32, target_rowset BYTE_ARRAY)`
//!    sorted ASC by `value` — exactly the shape `ParquetIndex::lookup_eq`
//!    expects.
//!
//! ## Memory shape
//!
//! Peak memory is proportional to the number of distinct
//! `(value, rg, page)` triples — bounded above by `num_rows`, but
//! typically far smaller for repeating columns (a date column over
//! TPC-H lineitem has ~2500 distinct values spread over ~200
//! pages → ~500K triples even at 6 M rows). Each triple holds a
//! `Vec<u32>` of at most `page_num_values` row positions; for the
//! common low-selectivity case those vectors are short.
//!
//! The MVP is *in-memory*. Petabyte-scale sources will need an
//! external-sort pass (Π.18+) — gated on real usage, not
//! pre-optimization.

use std::collections::HashMap;
use std::path::Path;

use ematix_parquet_format::types::{CompressionCodec, ParquetType};
use ematix_parquet_io::ParquetFile;

use crate::error::{CodecError, Result};
use crate::index::fingerprint::compute_source_fingerprint;
use crate::index::manifest::{IndexEntry, IndexKind, IndexManifest, PhysicalType, MANIFEST_KEY};
use crate::index::page_layout::{walk_data_pages, DataPageLayout};
use crate::read::read_column_i64;
use crate::write::{write_table_with_options_to_path, ColumnData, WriteOptions};

/// Builder for the sorted-i64 sidecar. Multi-index sidecars come
/// later; today each call to a `write_*` method emits one sidecar
/// file containing one index.
pub struct IndexBuilder<'a> {
    source: &'a ParquetFile,
}

impl<'a> IndexBuilder<'a> {
    /// Wrap a source `.parquet`. The builder does no I/O until
    /// `write_sorted_i64` is called.
    pub fn new(source: &'a ParquetFile) -> Self {
        Self { source }
    }

    /// Build a sorted INT64 index on `source_column` and write the
    /// sidecar to `out_path`.
    ///
    /// `index_name` is the logical handle the reader uses to address
    /// this index (e.g. `"idx_orderkey"`).
    ///
    /// `source_column` is the leaf column ordinal in the source
    /// file's schema. Must reference an `INT64` column or this call
    /// returns [`CodecError::InvalidInput`].
    pub fn write_sorted_i64<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_column: usize,
    ) -> Result<()> {
        // ---- 1. Validate the source column is INT64 ----------------
        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
        // The schema is depth-first; for flat REQUIRED schemas (the
        // only shape MVP supports), leaf `i` is at schema[i+1]
        // (schema[0] is the root). We don't yet wire general nested
        // walks here — flat is what Π.17b covers and what every TPC-H
        // reference shape is.
        let leaf_schema_idx = source_column
            .checked_add(1)
            .ok_or_else(|| CodecError::InvalidInput("source_column index overflow".into()))?;
        let leaf = md.schema.get(leaf_schema_idx).ok_or_else(|| {
            CodecError::InvalidInput(format!(
                "source_column {source_column} out of range in flat schema"
            ))
        })?;
        let leaf_type = leaf.column_type.ok_or_else(|| {
            CodecError::InvalidInput(format!(
                "schema element at leaf {source_column} has no physical type (group node?)"
            ))
        })?;
        if leaf_type != ParquetType::Int64 {
            return Err(CodecError::InvalidInput(format!(
                "write_sorted_i64: source column {source_column} is {leaf_type:?}, expected INT64"
            )));
        }

        // ---- 2. Walk row groups, bucket values ---------------------
        // Key = (value, source_rg, source_page) so duplicates within
        // the same page coalesce into one bitmap.
        let mut buckets: HashMap<(i64, u32, u32), Vec<u32>> = HashMap::new();
        // Per-(rg, page) num_values — needed to size the rowset bitmap.
        let mut page_sizes: HashMap<(u32, u32), u32> = HashMap::new();

        for rg in 0..md.row_groups.len() {
            // Decode the indexed column for this row group. One alloc
            // per RG; memory peak is one RG's column at a time.
            let values = read_column_i64(self.source, rg, source_column)?;

            // Walk pages to learn page boundaries. No body
            // decompression — header-only.
            let mut absolute_offsets: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_column, |layout| {
                absolute_offsets.push(layout);
                Ok(())
            })?;

            // Cross-check: page boundaries must sum to the row count
            // of the row group. If a writer ever emits a chunk where
            // they don't, the sidecar would be silently miscomputed —
            // fail loud here instead.
            let summed: usize = absolute_offsets.iter().map(|p| p.num_values).sum();
            if summed != values.len() {
                return Err(CodecError::InvalidInput(format!(
                    "page-layout walk for rg={rg} col={source_column} summed to {summed} values, \
                     but column decoded {}",
                    values.len()
                )));
            }

            for page in &absolute_offsets {
                page_sizes.insert((rg as u32, page.page_idx), page.num_values as u32);
                let base = page.first_row;
                for r in 0..page.num_values {
                    let v = values[base + r];
                    buckets
                        .entry((v, rg as u32, page.page_idx))
                        .or_default()
                        .push(r as u32);
                }
            }
        }

        // ---- 3. Sort + materialize rowsets -------------------------
        let mut keys: Vec<(i64, u32, u32)> = buckets.keys().copied().collect();
        keys.sort_unstable();

        let n_rows = keys.len();
        let mut col_value: Vec<i64> = Vec::with_capacity(n_rows);
        let mut col_rg: Vec<i32> = Vec::with_capacity(n_rows);
        let mut col_page: Vec<i32> = Vec::with_capacity(n_rows);
        let mut col_rowset_owned: Vec<Vec<u8>> = Vec::with_capacity(n_rows);

        for (v, rg, page) in keys {
            let num_values = *page_sizes
                .get(&(rg, page))
                .expect("page_sizes entry exists for emitted bucket")
                as usize;
            let positions = buckets
                .remove(&(v, rg, page))
                .expect("bucket exists for emitted key");
            let bitmap_len = num_values.div_ceil(8);
            let mut bitmap = vec![0u8; bitmap_len];
            for r in positions {
                let r = r as usize;
                debug_assert!(r < num_values, "row_within_page out of range");
                bitmap[r / 8] |= 1 << (r % 8);
            }
            col_value.push(v);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(bitmap);
        }

        // ---- 4. Build manifest + emit sidecar parquet --------------
        let fp = compute_source_fingerprint(self.source)?;
        // Store the leaf column's *name* in the manifest. Readers
        // resolve it back to a column ordinal by walking the
        // source's schema. Future nested support adds a
        // `source_column_path: Vec<String>` field; the current
        // single-string form encodes the leaf name only.
        let leaf_name = std::str::from_utf8(leaf.name).map_err(|_| {
            CodecError::InvalidInput("schema element name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::Sorted {
                    source_column: leaf_name.to_owned(),
                    physical_type: PhysicalType::Int64,
                },
                sidecar_row_group: 0,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY, manifest_json.as_str())];

        // ColumnData::ByteArray wants &[&[u8]], so we need a slice of
        // slices alongside the owned `Vec<Vec<u8>>`.
        let rowset_slices: Vec<&[u8]> = col_rowset_owned.iter().map(|v| v.as_slice()).collect();

        let cols: &[(&str, ColumnData<'_>)] = &[
            ("value", ColumnData::I64(&col_value)),
            ("target_rg", ColumnData::I32(&col_rg)),
            ("target_page", ColumnData::I32(&col_page)),
            ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
        ];
        let opts = WriteOptions {
            // Snappy on the value/page columns keeps the lookup
            // decode cheap; the rowset bitmaps are already
            // dense-or-sparse-but-compressible. Per-column codec
            // selection lands later; one codec for the whole file is
            // fine for MVP.
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }
}
