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

use crate::bloom::{optimal_num_blocks, parquet_xxh64, SplitBlockBloomFilterBuilder};
use crate::error::{CodecError, Result};
use crate::index::fingerprint::compute_source_fingerprint;
use crate::index::manifest::{
    IndexEntry, IndexKind, IndexManifest, PhysicalType, Tokenizer, MANIFEST_KEY_V3,
};
use crate::index::page_layout::{walk_data_pages, DataPageLayout};
use crate::read::{read_column_byte_array, read_column_i32, read_column_i64};
use crate::write::{write_table_with_options_to_path, ColumnData, WriteOptions};

/// Builder for the sorted-i64 sidecar. Multi-index sidecars come
/// later; today each call to a `write_*` method emits one sidecar
/// file containing one index.
pub struct IndexBuilder<'a> {
    source: &'a ParquetFile,
}

/// Sorted-index bodies are cut into row groups of this many index
/// rows (v3 sidecars). Each RG carries footer min/max on the sorted
/// `value` column, so a lazy reader can binary-search RG bounds and
/// decode ONLY the ~few-MB group containing a key — instead of the
/// v1/v2 whole-body eager load (~1s and ~0.5-1 GB on a 19M-value
/// lineitem-part index; the "open cost" that made point lookups
/// slower than full scans at SF100).
pub const SIDECAR_RG_ROWS: usize = 256 * 1024;

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
            let rowset = crate::index::rowset::encode_tagged(&positions, num_values);
            col_value.push(v);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(rowset);
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
                sidecar_row_group_count: n_rows.div_ceil(SIDECAR_RG_ROWS).max(1) as u32,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];

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
            // Chunked body (v3): rows are sorted by `value`, so each
            // row group's footer min/max form ordered, prunable ranges —
            // the lazy reader's whole trick.
            row_group_size: SIDECAR_RG_ROWS,
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }

    /// Mirror of [`Self::write_sorted_i64`] over `INT32`. Same
    /// per-page bucket + sort + pack pipeline; the only differences
    /// are the source-column type check and the schema of the
    /// emitted `value` column.
    pub fn write_sorted_i32<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_column: usize,
    ) -> Result<()> {
        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
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
        if leaf_type != ParquetType::Int32 {
            return Err(CodecError::InvalidInput(format!(
                "write_sorted_i32: source column {source_column} is {leaf_type:?}, expected INT32"
            )));
        }

        let mut buckets: HashMap<(i32, u32, u32), Vec<u32>> = HashMap::new();
        let mut page_sizes: HashMap<(u32, u32), u32> = HashMap::new();

        for rg in 0..md.row_groups.len() {
            let values = read_column_i32(self.source, rg, source_column)?;
            let mut absolute_offsets: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_column, |layout| {
                absolute_offsets.push(layout);
                Ok(())
            })?;
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

        let mut keys: Vec<(i32, u32, u32)> = buckets.keys().copied().collect();
        keys.sort_unstable();

        let n_rows = keys.len();
        let mut col_value: Vec<i32> = Vec::with_capacity(n_rows);
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
            let rowset = crate::index::rowset::encode_tagged(&positions, num_values);
            col_value.push(v);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(rowset);
        }

        let fp = compute_source_fingerprint(self.source)?;
        let leaf_name = std::str::from_utf8(leaf.name).map_err(|_| {
            CodecError::InvalidInput("schema element name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::Sorted {
                    source_column: leaf_name.to_owned(),
                    physical_type: PhysicalType::Int32,
                },
                sidecar_row_group: 0,
                sidecar_row_group_count: 1,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];
        let rowset_slices: Vec<&[u8]> = col_rowset_owned.iter().map(|v| v.as_slice()).collect();
        let cols: &[(&str, ColumnData<'_>)] = &[
            ("value", ColumnData::I32(&col_value)),
            ("target_rg", ColumnData::I32(&col_rg)),
            ("target_page", ColumnData::I32(&col_page)),
            ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
        ];
        let opts = WriteOptions {
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }

    /// Build a **per-source-page Bloom filter** index over an `INT64`
    /// column and write it as a sidecar. Each row in the sidecar
    /// holds the SBBF bitset (+ a small header) for one source page;
    /// readers probe the bloom for a query value to decide which
    /// pages are worth fully scanning.
    ///
    /// `target_fpp` is the false-positive probability per page —
    /// `0.01` is a reasonable default. The actual size of each
    /// bloom is `optimal_num_blocks(distinct_per_page, fpp) * 32`
    /// bytes plus the small `BloomFilterHeader` Thrift prefix
    /// (~20 bytes). For a 32 K-value page with ~30 K distinct INT64
    /// values at `fpp=0.01` this is ~36 KB per bloom; for the
    /// low-cardinality Q14-shape `l_shipdate` page (~2500 distinct)
    /// it's ~3 KB.
    ///
    /// Bloom filters are *equality-only* — they have no notion of
    /// ranges. For range queries use [`Self::write_sorted_i64`].
    pub fn write_bloom_page_i64<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_column: usize,
        target_fpp: f64,
    ) -> Result<()> {
        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
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
                "write_bloom_page_i64: source column {source_column} is {leaf_type:?}, expected INT64"
            )));
        }
        if !(target_fpp > 0.0 && target_fpp < 1.0) {
            return Err(CodecError::InvalidInput(format!(
                "write_bloom_page_i64: target_fpp must be in (0, 1), got {target_fpp}"
            )));
        }

        // (source_rg, source_page) → SBBF bytes (header + bitset).
        // Output is sorted lexicographically by (rg, page) so a
        // sequential scan walks pages in their source-file order;
        // a future ColumnIndex on the sidecar's source_rg/source_page
        // columns lets a reader page-skip the sidecar too.
        let mut entries: Vec<(u32, u32, Vec<u8>)> = Vec::new();

        for rg in 0..md.row_groups.len() {
            let values = read_column_i64(self.source, rg, source_column)?;
            let mut absolute_offsets: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_column, |layout| {
                absolute_offsets.push(layout);
                Ok(())
            })?;
            let summed: usize = absolute_offsets.iter().map(|p| p.num_values).sum();
            if summed != values.len() {
                return Err(CodecError::InvalidInput(format!(
                    "page-layout walk for rg={rg} col={source_column} summed to {summed} values, \
                     but column decoded {}",
                    values.len()
                )));
            }

            for page in &absolute_offsets {
                let slice = &values[page.first_row..page.first_row + page.num_values];

                // Distinct count of i64s in the page. A HashSet is the
                // simplest correct way; per-page memory is bounded by
                // the page's row count.
                let mut distinct: std::collections::HashSet<i64> = std::collections::HashSet::new();
                for &v in slice {
                    distinct.insert(v);
                }
                let n_distinct = distinct.len();

                // Empty page → emit an empty filter (single 32-byte
                // block, all zeros). Probing returns false for every
                // value, which is the right answer.
                let num_blocks = optimal_num_blocks(n_distinct, target_fpp);
                let mut bloom = SplitBlockBloomFilterBuilder::new(num_blocks);
                for v in distinct {
                    // Parquet's spec hashes the PLAIN-encoded form;
                    // for INT64 that's the 8-byte little-endian bytes.
                    bloom.insert_hash(parquet_xxh64(&v.to_le_bytes()));
                }
                entries.push((rg as u32, page.page_idx, bloom.into_bytes()));
            }
        }

        // Sort (rg, page) ASC.
        entries.sort_by_key(|(rg, page, _)| (*rg, *page));

        let n = entries.len();
        let mut col_rg: Vec<i32> = Vec::with_capacity(n);
        let mut col_page: Vec<i32> = Vec::with_capacity(n);
        let mut col_bloom_owned: Vec<Vec<u8>> = Vec::with_capacity(n);
        for (rg, page, bytes) in entries {
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_bloom_owned.push(bytes);
        }

        let fp = compute_source_fingerprint(self.source)?;
        let leaf_name = std::str::from_utf8(leaf.name).map_err(|_| {
            CodecError::InvalidInput("schema element name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::BloomPage {
                    source_column: leaf_name.to_owned(),
                    target_fpp,
                },
                sidecar_row_group: 0,
                sidecar_row_group_count: 1,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];

        let bloom_slices: Vec<&[u8]> = col_bloom_owned.iter().map(|v| v.as_slice()).collect();
        let cols: &[(&str, ColumnData<'_>)] = &[
            ("source_rg", ColumnData::I32(&col_rg)),
            ("source_page", ColumnData::I32(&col_page)),
            ("bloom_block", ColumnData::ByteArray(&bloom_slices)),
        ];
        // No compression for the bloom column: the bitset is
        // designed to look uniform-random, so dictionary and Snappy
        // can only hurt. Page offsets (small ints) ride along
        // uncompressed here too for one less codec dispatch.
        let opts = WriteOptions {
            default_codec: CompressionCodec::Uncompressed,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }

    /// Two-column **leading-prefix composite index** over a pair of
    /// `INT64` columns. Schema:
    ///
    /// ```text
    /// value_a:         INT64   (sort key, leading)
    /// value_b:         INT64   (sort key, trailing)
    /// target_rg:       INT32
    /// target_page:     INT32
    /// target_rowset:   BYTE_ARRAY (packed bitmap, page-relative)
    /// ```
    ///
    /// Sorted by `(value_a, value_b)` ASC. Supports:
    /// - **`lookup_composite_eq(name, a, b)`** — exact 2-tuple match.
    /// - **`lookup_composite_prefix(name, a)`** — every hit whose
    ///   `value_a == a` regardless of `value_b`.
    ///
    /// Rowsets are page-relative to `source_col_a` (the leading
    /// column). The convenience reader entries walk `source_col_a`'s
    /// page layout to translate `(rg, page) → first_row` when
    /// assembling the chunk-wide bitmap. Two columns of the same
    /// source file always share row counts per row group, so the
    /// bitmap correctly addresses rows whether the downstream
    /// `read_column_*_masked_into` reads `source_col_a`,
    /// `source_col_b`, or any other column in the same row group.
    ///
    /// 3+ column composites (`(a, b, c, ...)`) come later — they
    /// need a wider schema and a generalized lookup API. For now the
    /// 2-tuple form covers the common case (e.g.
    /// `(l_shipdate, l_partkey)`).
    pub fn write_sorted_composite_prefix_i64_i64<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_col_a: usize,
        source_col_b: usize,
    ) -> Result<()> {
        if source_col_a == source_col_b {
            return Err(CodecError::InvalidInput(format!(
                "write_sorted_composite_prefix_i64_i64: source_col_a and source_col_b must differ (both = {source_col_a})"
            )));
        }

        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
        let leaf_a = leaf_or_err(&md.schema, source_col_a)?;
        let leaf_b = leaf_or_err(&md.schema, source_col_b)?;
        let leaf_a_type = leaf_a.column_type.ok_or_else(|| {
            CodecError::InvalidInput(format!(
                "schema element at leaf {source_col_a} has no physical type (group node?)"
            ))
        })?;
        let leaf_b_type = leaf_b.column_type.ok_or_else(|| {
            CodecError::InvalidInput(format!(
                "schema element at leaf {source_col_b} has no physical type (group node?)"
            ))
        })?;
        if leaf_a_type != ParquetType::Int64 || leaf_b_type != ParquetType::Int64 {
            return Err(CodecError::InvalidInput(format!(
                "write_sorted_composite_prefix_i64_i64: source columns {source_col_a}/{source_col_b} \
                 are {leaf_a_type:?}/{leaf_b_type:?}, expected INT64/INT64"
            )));
        }

        // Bucket key = (value_a, value_b, source_rg, source_page).
        // Page is keyed on source_col_a; rowsets are page-relative to
        // source_col_a's pages.
        let mut buckets: HashMap<(i64, i64, u32, u32), Vec<u32>> = HashMap::new();
        let mut page_sizes: HashMap<(u32, u32), u32> = HashMap::new();

        for rg in 0..md.row_groups.len() {
            let values_a = read_column_i64(self.source, rg, source_col_a)?;
            let values_b = read_column_i64(self.source, rg, source_col_b)?;
            if values_a.len() != values_b.len() {
                return Err(CodecError::InvalidInput(format!(
                    "composite-prefix build: column lengths differ for rg={rg} \
                     (col_a={} rows, col_b={} rows)",
                    values_a.len(),
                    values_b.len()
                )));
            }

            let mut page_layouts: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_col_a, |layout| {
                page_layouts.push(layout);
                Ok(())
            })?;
            let summed: usize = page_layouts.iter().map(|p| p.num_values).sum();
            if summed != values_a.len() {
                return Err(CodecError::InvalidInput(format!(
                    "page-layout walk for rg={rg} col_a={source_col_a} summed to {summed} values, \
                     but column decoded {}",
                    values_a.len()
                )));
            }

            for page in &page_layouts {
                page_sizes.insert((rg as u32, page.page_idx), page.num_values as u32);
                let base = page.first_row;
                for r in 0..page.num_values {
                    let v_a = values_a[base + r];
                    let v_b = values_b[base + r];
                    buckets
                        .entry((v_a, v_b, rg as u32, page.page_idx))
                        .or_default()
                        .push(r as u32);
                }
            }
        }

        // Sort by (value_a, value_b, rg, page).
        let mut keys: Vec<(i64, i64, u32, u32)> = buckets.keys().copied().collect();
        keys.sort_unstable();

        let n = keys.len();
        let mut col_value_a: Vec<i64> = Vec::with_capacity(n);
        let mut col_value_b: Vec<i64> = Vec::with_capacity(n);
        let mut col_rg: Vec<i32> = Vec::with_capacity(n);
        let mut col_page: Vec<i32> = Vec::with_capacity(n);
        let mut col_rowset_owned: Vec<Vec<u8>> = Vec::with_capacity(n);

        for (va, vb, rg, page) in keys {
            let num_values = *page_sizes
                .get(&(rg, page))
                .expect("page_sizes entry exists for emitted bucket")
                as usize;
            let positions = buckets
                .remove(&(va, vb, rg, page))
                .expect("bucket exists for emitted key");
            let rowset = crate::index::rowset::encode_tagged(&positions, num_values);
            col_value_a.push(va);
            col_value_b.push(vb);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(rowset);
        }

        let fp = compute_source_fingerprint(self.source)?;
        let leaf_a_name = std::str::from_utf8(leaf_a.name).map_err(|_| {
            CodecError::InvalidInput("schema element a name is not valid UTF-8".into())
        })?;
        let leaf_b_name = std::str::from_utf8(leaf_b.name).map_err(|_| {
            CodecError::InvalidInput("schema element b name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::CompositePrefix {
                    source_columns: vec![leaf_a_name.to_owned(), leaf_b_name.to_owned()],
                    physical_types: vec![PhysicalType::Int64, PhysicalType::Int64],
                },
                sidecar_row_group: 0,
                sidecar_row_group_count: 1,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];
        let rowset_slices: Vec<&[u8]> = col_rowset_owned.iter().map(|v| v.as_slice()).collect();
        let cols: &[(&str, ColumnData<'_>)] = &[
            ("value_a", ColumnData::I64(&col_value_a)),
            ("value_b", ColumnData::I64(&col_value_b)),
            ("target_rg", ColumnData::I32(&col_rg)),
            ("target_page", ColumnData::I32(&col_page)),
            ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
        ];
        let opts = WriteOptions {
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }

    /// Mirror of [`Self::write_sorted_i64`] over `BYTE_ARRAY`. The
    /// bucket key carries owned `Vec<u8>` values; sort is lex-ASC
    /// (Rust `Vec<u8>` default `Ord` matches Parquet unsigned-lex,
    /// the spec's BYTE_ARRAY sort order).
    pub fn write_sorted_byte_array<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_column: usize,
    ) -> Result<()> {
        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
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
        if leaf_type != ParquetType::ByteArray {
            return Err(CodecError::InvalidInput(format!(
                "write_sorted_byte_array: source column {source_column} is {leaf_type:?}, expected BYTE_ARRAY"
            )));
        }

        let mut buckets: HashMap<(Vec<u8>, u32, u32), Vec<u32>> = HashMap::new();
        let mut page_sizes: HashMap<(u32, u32), u32> = HashMap::new();

        for rg in 0..md.row_groups.len() {
            let values = read_column_byte_array(self.source, rg, source_column)?;
            let mut absolute_offsets: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_column, |layout| {
                absolute_offsets.push(layout);
                Ok(())
            })?;
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
                    let v = values[base + r].clone();
                    buckets
                        .entry((v, rg as u32, page.page_idx))
                        .or_default()
                        .push(r as u32);
                }
            }
        }

        // Sort by (value, rg, page) — lex-ASC on the bytes, matching
        // Parquet's BYTE_ARRAY sort. Rust's default Ord on Vec<u8>
        // is byte-wise unsigned-lex, so this is correct out of the
        // box.
        let mut keys: Vec<(Vec<u8>, u32, u32)> = buckets.keys().cloned().collect();
        keys.sort();

        let n_rows = keys.len();
        let mut col_value_owned: Vec<Vec<u8>> = Vec::with_capacity(n_rows);
        let mut col_rg: Vec<i32> = Vec::with_capacity(n_rows);
        let mut col_page: Vec<i32> = Vec::with_capacity(n_rows);
        let mut col_rowset_owned: Vec<Vec<u8>> = Vec::with_capacity(n_rows);

        for (v, rg, page) in keys {
            let num_values = *page_sizes
                .get(&(rg, page))
                .expect("page_sizes entry exists for emitted bucket")
                as usize;
            let positions = buckets
                .remove(&(v.clone(), rg, page))
                .expect("bucket exists for emitted key");
            let rowset = crate::index::rowset::encode_tagged(&positions, num_values);
            col_value_owned.push(v);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(rowset);
        }

        let fp = compute_source_fingerprint(self.source)?;
        let leaf_name = std::str::from_utf8(leaf.name).map_err(|_| {
            CodecError::InvalidInput("schema element name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::Sorted {
                    source_column: leaf_name.to_owned(),
                    physical_type: PhysicalType::ByteArray,
                },
                sidecar_row_group: 0,
                sidecar_row_group_count: 1,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];

        let value_slices: Vec<&[u8]> = col_value_owned.iter().map(|v| v.as_slice()).collect();
        let rowset_slices: Vec<&[u8]> = col_rowset_owned.iter().map(|v| v.as_slice()).collect();
        let cols: &[(&str, ColumnData<'_>)] = &[
            ("value", ColumnData::ByteArray(&value_slices)),
            ("target_rg", ColumnData::I32(&col_rg)),
            ("target_page", ColumnData::I32(&col_page)),
            ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
        ];
        let opts = WriteOptions {
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }
}

impl<'a> IndexBuilder<'a> {
    /// **Inverted text index** over a `BYTE_ARRAY` column. For every
    /// row, the value is tokenized via `tokenizer`, deduped per row
    /// (so a row containing "foo foo" sets the row's bit in the
    /// `foo` token's bitmap once, not twice), and each `(token, rg,
    /// page)` bucket accumulates the set of row positions in the
    /// page where that token appears.
    ///
    /// Sidecar schema:
    /// ```text
    /// token:           BYTE_ARRAY  (sort key, lex-ASC)
    /// target_rg:       INT32
    /// target_page:     INT32
    /// target_rowset:   BYTE_ARRAY  (packed bitmap, page-relative
    ///                               to the indexed BYTE_ARRAY column)
    /// ```
    ///
    /// The chosen `tokenizer` is recorded in the manifest under
    /// [`IndexKind::Inverted::tokenizer`]; the reader applies the
    /// same one to query terms before lookup.
    ///
    /// MVP v1: single-token-per-query lookup (one row's worth of
    /// matches per `lookup_token` call). Multi-token AND/OR
    /// composition lives in a higher-level engine layer or future
    /// Π.20b.
    pub fn write_inverted_byte_array<P: AsRef<Path>>(
        &self,
        out_path: P,
        index_name: &str,
        source_column: usize,
        tokenizer: Tokenizer,
    ) -> Result<()> {
        let md = self
            .source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read parquet metadata: {e}")))?;
        let leaf = leaf_or_err(&md.schema, source_column)?;
        let leaf_type = leaf.column_type.ok_or_else(|| {
            CodecError::InvalidInput(format!(
                "schema element at leaf {source_column} has no physical type (group node?)"
            ))
        })?;
        if leaf_type != ParquetType::ByteArray {
            return Err(CodecError::InvalidInput(format!(
                "write_inverted_byte_array: source column {source_column} is {leaf_type:?}, expected BYTE_ARRAY"
            )));
        }

        // (token_bytes, source_rg, source_page) → Vec<row_within_page>
        let mut buckets: HashMap<(Vec<u8>, u32, u32), Vec<u32>> = HashMap::new();
        let mut page_sizes: HashMap<(u32, u32), u32> = HashMap::new();

        for rg in 0..md.row_groups.len() {
            let values = read_column_byte_array(self.source, rg, source_column)?;
            let mut page_layouts: Vec<DataPageLayout> = Vec::new();
            walk_data_pages(self.source, rg, source_column, |layout| {
                page_layouts.push(layout);
                Ok(())
            })?;
            let summed: usize = page_layouts.iter().map(|p| p.num_values).sum();
            if summed != values.len() {
                return Err(CodecError::InvalidInput(format!(
                    "page-layout walk for rg={rg} col={source_column} summed to {summed} values, \
                     but column decoded {}",
                    values.len()
                )));
            }

            for page in &page_layouts {
                page_sizes.insert((rg as u32, page.page_idx), page.num_values as u32);
                let base = page.first_row;
                for r in 0..page.num_values {
                    let row_value = &values[base + r];
                    let toks = tokenizer.tokenize(row_value);
                    // Dedupe tokens within this single row; we only
                    // want one bit per (token, row) regardless of how
                    // many times the token appears in the row.
                    let mut seen: std::collections::HashSet<Vec<u8>> =
                        std::collections::HashSet::with_capacity(toks.len());
                    for t in toks {
                        if seen.insert(t.clone()) {
                            buckets
                                .entry((t, rg as u32, page.page_idx))
                                .or_default()
                                .push(r as u32);
                        }
                    }
                }
            }
        }

        // Sort by (token, rg, page) — lex-ASC on the token bytes,
        // matching `lookup_token`'s binary-search expectation.
        let mut keys: Vec<(Vec<u8>, u32, u32)> = buckets.keys().cloned().collect();
        keys.sort();

        let n = keys.len();
        let mut col_token_owned: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut col_rg: Vec<i32> = Vec::with_capacity(n);
        let mut col_page: Vec<i32> = Vec::with_capacity(n);
        let mut col_rowset_owned: Vec<Vec<u8>> = Vec::with_capacity(n);
        for (tok, rg, page) in keys {
            let num_values = *page_sizes
                .get(&(rg, page))
                .expect("page_sizes entry exists for emitted bucket")
                as usize;
            let positions = buckets
                .remove(&(tok.clone(), rg, page))
                .expect("bucket exists for emitted key");
            let rowset = crate::index::rowset::encode_tagged(&positions, num_values);
            col_token_owned.push(tok);
            col_rg.push(rg as i32);
            col_page.push(page as i32);
            col_rowset_owned.push(rowset);
        }

        let fp = compute_source_fingerprint(self.source)?;
        let leaf_name = std::str::from_utf8(leaf.name).map_err(|_| {
            CodecError::InvalidInput("schema element name is not valid UTF-8".into())
        })?;
        let manifest = IndexManifest {
            source_fingerprint: fp,
            indexes: vec![IndexEntry {
                name: index_name.to_owned(),
                kind: IndexKind::Inverted {
                    source_column: leaf_name.to_owned(),
                    tokenizer,
                },
                sidecar_row_group: 0,
                sidecar_row_group_count: 1,
            }],
        };
        let manifest_json = manifest.to_json();
        let kvs = [(MANIFEST_KEY_V3, manifest_json.as_str())];
        let token_slices: Vec<&[u8]> = col_token_owned.iter().map(|v| v.as_slice()).collect();
        let rowset_slices: Vec<&[u8]> = col_rowset_owned.iter().map(|v| v.as_slice()).collect();
        let cols: &[(&str, ColumnData<'_>)] = &[
            ("token", ColumnData::ByteArray(&token_slices)),
            ("target_rg", ColumnData::I32(&col_rg)),
            ("target_page", ColumnData::I32(&col_page)),
            ("target_rowset", ColumnData::ByteArray(&rowset_slices)),
        ];
        let opts = WriteOptions {
            default_codec: CompressionCodec::Snappy,
            kv_metadata: Some(&kvs),
            ..WriteOptions::default()
        };
        write_table_with_options_to_path(out_path, cols, &opts)
    }
}

/// Look up the schema element for leaf-column ordinal `source_column`
/// in a flat REQUIRED schema. `schema[0]` is the root group; leaves
/// follow in depth-first order, so leaf `i` is at `schema[i + 1]`.
fn leaf_or_err<'a>(
    schema: &'a [ematix_parquet_format::metadata::SchemaElement<'a>],
    source_column: usize,
) -> Result<&'a ematix_parquet_format::metadata::SchemaElement<'a>> {
    let leaf_schema_idx = source_column
        .checked_add(1)
        .ok_or_else(|| CodecError::InvalidInput("source_column index overflow".into()))?;
    schema.get(leaf_schema_idx).ok_or_else(|| {
        CodecError::InvalidInput(format!(
            "source_column {source_column} out of range in flat schema"
        ))
    })
}
