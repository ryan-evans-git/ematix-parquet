//! Sidecar-index reader. Π.17b shipped sorted INT64; Π.18 widens to
//! INT32 + BYTE_ARRAY and adds range queries (`lookup_range`,
//! `read_column_*_where_range`).
//!
//! `ParquetIndex::open` parses the sidecar's footer `KeyValueMetadata`,
//! verifies the embedded fingerprint against the source, and **eagerly**
//! decodes the index row groups into memory. Lookups are pure-CPU
//! after open. Eager-load is the simplest correct shape for MVP;
//! sidecars are tens of MB at most, sub-millisecond to load. The
//! Iceberg layer (Π.21+) introduces per-manifest summaries that prune
//! which sidecars even need opening — eager-per-sidecar stays
//! scalable in the dataset case.
//!
//! Equality (`lookup_eq`) is `log(n)` binary search + linear scan
//! over duplicates. Range (`lookup_range`) is the same binary search
//! at `lo` followed by a forward walk until `value > hi`.

use std::ops::RangeInclusive;
use std::path::Path;

use ematix_parquet_io::ParquetFile;

use crate::bloom::{parquet_xxh64, SplitBlockBloomFilter};
use crate::error::{CodecError, Result};
use crate::index::fingerprint::compute_source_fingerprint;
use crate::index::manifest::{
    IndexEntry, IndexKind, IndexManifest, ManifestError, MANIFEST_KEY, MANIFEST_KEY_V2,
};
use crate::index::page_layout::walk_data_pages;
use crate::index::rowset::{to_bitmap, RowsetFormat};
use crate::index::types::{IndexHit, Key};
use crate::index::PhysicalType;
use crate::read::{read_column_byte_array, read_column_i32, read_column_i64};

// ============================================================
// Loaded indexes — one variant per physical type.
// ============================================================
//
// Each variant carries four parallel vectors (values + target_rg +
// target_page + rowset). Per-row alignment is the load-bearing
// invariant; constructors verify the four lengths match before
// returning.

#[derive(Debug)]
struct LoadedTypedI64 {
    values: Vec<i64>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct LoadedTypedI32 {
    values: Vec<i32>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct LoadedTypedBytes {
    values: Vec<Vec<u8>>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

/// One loaded page-Bloom index. Each row of the sidecar parquet
/// becomes one `(source_rg, source_page, bloom_bytes)` triple; the
/// `bloom_bytes` is the SBBF `header + bitset` form
/// (`SplitBlockBloomFilterBuilder::into_bytes()`).
#[derive(Debug)]
struct LoadedBloomPage {
    source_rgs: Vec<i32>,
    source_pages: Vec<i32>,
    blooms: Vec<Vec<u8>>,
}

/// One loaded composite (INT64, INT64) leading-prefix index. Five
/// parallel vectors: `(value_a, value_b)` is the sort key,
/// `(target_rg, target_page, rowset)` carries the page-relative hit
/// (anchored to `source_columns[0]`'s page layout).
#[derive(Debug)]
struct LoadedTypedCompositeI64I64 {
    values_a: Vec<i64>,
    values_b: Vec<i64>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

/// One loaded inverted (text) index. Same shape as a sorted-bytes
/// index but `tokens` are *post-tokenizer* byte sequences rather
/// than original column values; multiple rows of the indexed
/// column can contribute to the same token's bucket.
#[derive(Debug)]
struct LoadedTypedInverted {
    tokens: Vec<Vec<u8>>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

#[derive(Debug)]
enum LoadedIndexData {
    SortedI64(LoadedTypedI64),
    SortedI32(LoadedTypedI32),
    SortedBytes(LoadedTypedBytes),
    BloomPage(LoadedBloomPage),
    CompositePrefixI64I64(LoadedTypedCompositeI64I64),
    Inverted(LoadedTypedInverted),
}

#[derive(Debug)]
struct LoadedIndex {
    entry: IndexEntry,
    data: LoadedIndexData,
}

/// Parsed sidecar parquet. Owns the manifest and the in-memory index
/// data; lookups are pure-CPU after `open` returns.
#[derive(Debug)]
pub struct ParquetIndex {
    manifest: IndexManifest,
    indexes: Vec<LoadedIndex>,
    /// How stored rowset bytes are interpreted — decided by which
    /// manifest KV key the sidecar carried (v1 = raw bitmaps,
    /// v2 = tagged; see `index::rowset`).
    rowset_format: RowsetFormat,
}

impl ParquetIndex {
    /// Open `idx_path`, parse and validate the manifest against
    /// `source`, eagerly load every sorted index row group. Errors
    /// fast on missing/malformed manifest, version mismatch,
    /// fingerprint mismatch, or a sidecar shape that doesn't match
    /// what a sorted-index builder emits.
    pub fn open<P: AsRef<Path>>(idx_path: P, source: &ParquetFile) -> Result<Self> {
        let idx_file = ParquetFile::open(idx_path.as_ref())
            .map_err(|e| CodecError::InvalidInput(format!("open sidecar: {e}")))?;

        // ---- 1. Find + parse the manifest from KV metadata ---------
        let md = idx_file
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("read sidecar metadata: {e}")))?;
        let kvs = md
            .key_value_metadata
            .as_ref()
            .ok_or_else(|| codec_err(ManifestError::Missing))?;
        // v2 first (tagged rowsets), fall back to v1 (raw bitmaps).
        let (manifest_kv, rowset_format) = kvs
            .iter()
            .find(|kv| kv.key == MANIFEST_KEY_V2.as_bytes())
            .map(|kv| (kv, RowsetFormat::V2Tagged))
            .or_else(|| {
                kvs.iter()
                    .find(|kv| kv.key == MANIFEST_KEY.as_bytes())
                    .map(|kv| (kv, RowsetFormat::V1Raw))
            })
            .ok_or_else(|| codec_err(ManifestError::Missing))?;
        let manifest_json_bytes = manifest_kv
            .value
            .ok_or_else(|| codec_err(ManifestError::Missing))?;
        let manifest_json = std::str::from_utf8(manifest_json_bytes).map_err(|_| {
            codec_err(ManifestError::Malformed(
                "manifest value is not valid UTF-8".into(),
            ))
        })?;
        let manifest = IndexManifest::from_json(manifest_json).map_err(codec_err)?;

        // ---- 2. Verify fingerprint --------------------------------
        let actual_fp = compute_source_fingerprint(source)?;
        if actual_fp != manifest.source_fingerprint {
            return Err(codec_err(ManifestError::SourceFingerprintMismatch {
                expected: manifest.source_fingerprint,
                actual: actual_fp,
            }));
        }

        // ---- 3. Eagerly load each index ---------------------------
        let mut indexes: Vec<LoadedIndex> = Vec::with_capacity(manifest.indexes.len());
        for entry in &manifest.indexes {
            match &entry.kind {
                IndexKind::Sorted { physical_type, .. } => {
                    let rg = entry.sidecar_row_group as usize;
                    let data = match physical_type {
                        PhysicalType::Int64 => {
                            let values = read_column_i64(&idx_file, rg, 0)?;
                            let target_rgs = read_column_i32(&idx_file, rg, 1)?;
                            let target_pages = read_column_i32(&idx_file, rg, 2)?;
                            let rowsets = read_column_byte_array(&idx_file, rg, 3)?;
                            check_aligned(
                                &entry.name,
                                rg,
                                values.len(),
                                &target_rgs,
                                &target_pages,
                                &rowsets,
                            )?;
                            LoadedIndexData::SortedI64(LoadedTypedI64 {
                                values,
                                target_rgs,
                                target_pages,
                                rowsets,
                            })
                        }
                        PhysicalType::Int32 => {
                            let values = read_column_i32(&idx_file, rg, 0)?;
                            let target_rgs = read_column_i32(&idx_file, rg, 1)?;
                            let target_pages = read_column_i32(&idx_file, rg, 2)?;
                            let rowsets = read_column_byte_array(&idx_file, rg, 3)?;
                            check_aligned(
                                &entry.name,
                                rg,
                                values.len(),
                                &target_rgs,
                                &target_pages,
                                &rowsets,
                            )?;
                            LoadedIndexData::SortedI32(LoadedTypedI32 {
                                values,
                                target_rgs,
                                target_pages,
                                rowsets,
                            })
                        }
                        PhysicalType::ByteArray => {
                            let values = read_column_byte_array(&idx_file, rg, 0)?;
                            let target_rgs = read_column_i32(&idx_file, rg, 1)?;
                            let target_pages = read_column_i32(&idx_file, rg, 2)?;
                            let rowsets = read_column_byte_array(&idx_file, rg, 3)?;
                            check_aligned(
                                &entry.name,
                                rg,
                                values.len(),
                                &target_rgs,
                                &target_pages,
                                &rowsets,
                            )?;
                            LoadedIndexData::SortedBytes(LoadedTypedBytes {
                                values,
                                target_rgs,
                                target_pages,
                                rowsets,
                            })
                        }
                    };
                    indexes.push(LoadedIndex {
                        entry: entry.clone(),
                        data,
                    });
                }
                IndexKind::CompositePrefix {
                    source_columns,
                    physical_types,
                } => {
                    // MVP: only (INT64, INT64) is supported.
                    if source_columns.len() != 2
                        || physical_types.len() != 2
                        || physical_types[0] != PhysicalType::Int64
                        || physical_types[1] != PhysicalType::Int64
                    {
                        return Err(CodecError::InvalidInput(format!(
                            "composite-prefix index `{}`: MVP supports exactly (INT64, INT64); got types {:?}",
                            entry.name, physical_types
                        )));
                    }
                    let rg = entry.sidecar_row_group as usize;
                    // Schema: value_a INT64, value_b INT64, target_rg INT32, target_page INT32, target_rowset BYTE_ARRAY.
                    let values_a = read_column_i64(&idx_file, rg, 0)?;
                    let values_b = read_column_i64(&idx_file, rg, 1)?;
                    let target_rgs = read_column_i32(&idx_file, rg, 2)?;
                    let target_pages = read_column_i32(&idx_file, rg, 3)?;
                    let rowsets = read_column_byte_array(&idx_file, rg, 4)?;
                    let n = values_a.len();
                    if values_b.len() != n
                        || target_rgs.len() != n
                        || target_pages.len() != n
                        || rowsets.len() != n
                    {
                        return Err(CodecError::InvalidInput(format!(
                            "composite-prefix index `{}` row group {} has mismatched column lengths \
                             (a={}, b={}, rg={}, page={}, rowset={})",
                            entry.name,
                            rg,
                            n,
                            values_b.len(),
                            target_rgs.len(),
                            target_pages.len(),
                            rowsets.len()
                        )));
                    }
                    indexes.push(LoadedIndex {
                        entry: entry.clone(),
                        data: LoadedIndexData::CompositePrefixI64I64(LoadedTypedCompositeI64I64 {
                            values_a,
                            values_b,
                            target_rgs,
                            target_pages,
                            rowsets,
                        }),
                    });
                }
                IndexKind::Inverted { .. } => {
                    let rg = entry.sidecar_row_group as usize;
                    // Schema: token BYTE_ARRAY, target_rg INT32, target_page INT32, target_rowset BYTE_ARRAY.
                    let tokens = read_column_byte_array(&idx_file, rg, 0)?;
                    let target_rgs = read_column_i32(&idx_file, rg, 1)?;
                    let target_pages = read_column_i32(&idx_file, rg, 2)?;
                    let rowsets = read_column_byte_array(&idx_file, rg, 3)?;
                    let n = tokens.len();
                    if target_rgs.len() != n || target_pages.len() != n || rowsets.len() != n {
                        return Err(CodecError::InvalidInput(format!(
                            "inverted index `{}` row group {} has mismatched column lengths \
                             (tokens={}, rg={}, page={}, rowset={})",
                            entry.name,
                            rg,
                            n,
                            target_rgs.len(),
                            target_pages.len(),
                            rowsets.len()
                        )));
                    }
                    indexes.push(LoadedIndex {
                        entry: entry.clone(),
                        data: LoadedIndexData::Inverted(LoadedTypedInverted {
                            tokens,
                            target_rgs,
                            target_pages,
                            rowsets,
                        }),
                    });
                }
                IndexKind::BloomPage { .. } => {
                    let rg = entry.sidecar_row_group as usize;
                    // Schema: source_rg INT32, source_page INT32, bloom_block BYTE_ARRAY.
                    let source_rgs = read_column_i32(&idx_file, rg, 0)?;
                    let source_pages = read_column_i32(&idx_file, rg, 1)?;
                    let blooms = read_column_byte_array(&idx_file, rg, 2)?;
                    if source_rgs.len() != source_pages.len() || source_rgs.len() != blooms.len() {
                        return Err(CodecError::InvalidInput(format!(
                            "page-bloom index `{}` row group {} has mismatched column lengths \
                             (source_rgs={}, source_pages={}, blooms={})",
                            entry.name,
                            rg,
                            source_rgs.len(),
                            source_pages.len(),
                            blooms.len()
                        )));
                    }
                    indexes.push(LoadedIndex {
                        entry: entry.clone(),
                        data: LoadedIndexData::BloomPage(LoadedBloomPage {
                            source_rgs,
                            source_pages,
                            blooms,
                        }),
                    });
                }
            }
        }

        Ok(Self {
            manifest,
            indexes,
            rowset_format,
        })
    }

    /// Borrow the parsed manifest. Useful for tooling.
    pub fn manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    // ============================================================
    // lookup_eq
    // ============================================================

    /// Equality lookup. Returns every `(rg, page, rowset)` triple in
    /// the source file with at least one row matching `key`. Empty
    /// vec = no rows match. The fingerprint check at `open` is the
    /// authoritative guard against source drift; the reader does not
    /// re-verify on each lookup.
    pub fn lookup_eq(&self, index_name: &str, key: &Key<'_>) -> Result<Vec<IndexHit>> {
        let idx = self.find_index(index_name)?;
        let pt = sorted_physical_type(&idx.entry)?;
        if !key.matches_physical_type(pt) {
            return Err(CodecError::InvalidInput(format!(
                "lookup_eq: key variant `{}` does not match index `{}` physical type {:?}",
                key.variant_name(),
                index_name,
                pt,
            )));
        }
        match (&idx.data, key) {
            (LoadedIndexData::SortedI64(d), Key::I64(v)) => eq_hits_i64(d, *v, self.rowset_format),
            (LoadedIndexData::SortedI32(d), Key::I32(v)) => eq_hits_i32(d, *v, self.rowset_format),
            (LoadedIndexData::SortedBytes(d), Key::Bytes(v)) => {
                eq_hits_bytes(d, v, self.rowset_format)
            }
            _ => Err(CodecError::InvalidInput(format!(
                "lookup_eq: key/index type mismatch on `{index_name}`"
            ))),
        }
    }

    // ============================================================
    // lookup_range
    // ============================================================

    /// Inclusive range lookup. Returns every `(rg, page, rowset)`
    /// triple whose `value` is in `[lo, hi]`. Empty vec = no rows in
    /// range. Inverted ranges (`lo > hi`) return empty.
    ///
    /// The `Key` variants of `lo` and `hi` must match each other AND
    /// the index's physical type. The bytes slice in `Key::Bytes` is
    /// compared lex-ASC (matches `Vec<u8>` default `Ord` and the
    /// Parquet BYTE_ARRAY sort).
    pub fn lookup_range(
        &self,
        index_name: &str,
        lo: &Key<'_>,
        hi: &Key<'_>,
    ) -> Result<Vec<IndexHit>> {
        let idx = self.find_index(index_name)?;
        let pt = sorted_physical_type(&idx.entry)?;
        if !lo.matches_physical_type(pt) || !hi.matches_physical_type(pt) {
            return Err(CodecError::InvalidInput(format!(
                "lookup_range: key variants ({}, {}) do not match index `{}` physical type {:?}",
                lo.variant_name(),
                hi.variant_name(),
                index_name,
                pt,
            )));
        }
        match (&idx.data, lo, hi) {
            (LoadedIndexData::SortedI64(d), Key::I64(a), Key::I64(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                range_hits_i64(d, *a..=*b, self.rowset_format)
            }
            (LoadedIndexData::SortedI32(d), Key::I32(a), Key::I32(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                range_hits_i32(d, *a..=*b, self.rowset_format)
            }
            (LoadedIndexData::SortedBytes(d), Key::Bytes(a), Key::Bytes(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                range_hits_bytes(d, a, b, self.rowset_format)
            }
            _ => Err(CodecError::InvalidInput(format!(
                "lookup_range: key/index type mismatch on `{index_name}`"
            ))),
        }
    }

    // ============================================================
    // Composite leading-prefix lookups
    // ============================================================

    /// Exact 2-tuple equality on a composite `(INT64, INT64)` index.
    /// Returns every `(rg, page, rowset)` triple containing rows
    /// where `(value_a, value_b) == (key_a, key_b)`.
    ///
    /// Anchored to `source_columns[0]`'s page layout — see
    /// [`crate::index::IndexBuilder::write_sorted_composite_prefix_i64_i64`].
    pub fn lookup_composite_eq(
        &self,
        index_name: &str,
        key_a: &Key<'_>,
        key_b: &Key<'_>,
    ) -> Result<Vec<IndexHit>> {
        let idx = self.find_index(index_name)?;
        let d = match &idx.data {
            LoadedIndexData::CompositePrefixI64I64(d) => d,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "lookup_composite_eq: index `{index_name}` is not a (INT64, INT64) composite-prefix index"
                )))
            }
        };
        let (a, b) = match (key_a, key_b) {
            (Key::I64(a), Key::I64(b)) => (*a, *b),
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "lookup_composite_eq: key variants `{}`+`{}` do not match (INT64, INT64)",
                    key_a.variant_name(),
                    key_b.variant_name()
                )))
            }
        };
        composite_eq_hits(d, a, b, self.rowset_format)
    }

    /// Leading-prefix equality on a composite `(INT64, INT64)`
    /// index: returns every hit whose `value_a == key_a`, regardless
    /// of `value_b`. The trailing dimension is left free — useful
    /// for "WHERE col_a = X AND col_b IN [...]"-shaped predicates
    /// where the IN-list is too big to lookup point-wise.
    pub fn lookup_composite_prefix(
        &self,
        index_name: &str,
        key_a: &Key<'_>,
    ) -> Result<Vec<IndexHit>> {
        let idx = self.find_index(index_name)?;
        let d = match &idx.data {
            LoadedIndexData::CompositePrefixI64I64(d) => d,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "lookup_composite_prefix: index `{index_name}` is not a (INT64, INT64) composite-prefix index"
                )))
            }
        };
        let a = match key_a {
            Key::I64(a) => *a,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "lookup_composite_prefix: key variant `{}` does not match INT64",
                    key_a.variant_name()
                )))
            }
        };
        composite_prefix_hits(d, a, self.rowset_format)
    }

    /// `INT64` exact 2-tuple composite + masked decode. Equivalent
    /// to: `lookup_composite_eq → assemble per-rg bitmap → read_column_i64_masked_into`.
    pub fn read_column_i64_where_composite_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key_a: i64,
        key_b: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        let hits = self.lookup_composite_eq(index_name, &Key::I64(key_a), &Key::I64(key_b))?;
        self.materialize_i64_composite(source, index_name, &hits, target_column)
    }

    /// `INT64` leading-prefix composite + masked decode.
    pub fn read_column_i64_where_composite_prefix(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key_a: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        let hits = self.lookup_composite_prefix(index_name, &Key::I64(key_a))?;
        self.materialize_i64_composite(source, index_name, &hits, target_column)
    }

    // ============================================================
    // Inverted (text) lookups
    // ============================================================

    /// Look up rows containing `normalized_token` in an inverted
    /// index. **Caller must pre-normalize** the token to match the
    /// builder's tokenizer — pass the tokenizer's output as-is, not
    /// raw user input. For the higher-level path that applies the
    /// manifest's tokenizer to a query string, use
    /// [`Self::read_column_byte_array_where_token`].
    ///
    /// Returns `IndexHit`s pointing at pages/rowsets of the indexed
    /// `BYTE_ARRAY` source column. Empty vec = the token is not in
    /// the index (i.e. it appeared in no row, or never made it past
    /// the build-time tokenizer).
    pub fn lookup_token(&self, index_name: &str, normalized_token: &[u8]) -> Result<Vec<IndexHit>> {
        let idx = self.find_index(index_name)?;
        let d = match &idx.data {
            LoadedIndexData::Inverted(d) => d,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "lookup_token: index `{index_name}` is not an inverted index"
                )))
            }
        };
        inverted_eq_hits(d, normalized_token, self.rowset_format)
    }

    /// Read the indexed `BYTE_ARRAY` target column for every row
    /// containing `query` (after applying the manifest's tokenizer
    /// to normalize the query).
    ///
    /// The tokenizer is expected to produce **exactly one token**
    /// from `query` for an unambiguous lookup. Passing a multi-word
    /// query like `b"the quick fox"` errors with
    /// `query produced N tokens (expected 1)` — multi-token AND/OR
    /// composition is a higher-level engine concern.
    pub fn read_column_byte_array_where_token(
        &self,
        source: &ParquetFile,
        index_name: &str,
        query: &[u8],
        target_column: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let idx = self.find_index(index_name)?;
        let tokenizer = match &idx.entry.kind {
            IndexKind::Inverted { tokenizer, .. } => *tokenizer,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                "read_column_byte_array_where_token: index `{index_name}` is not an inverted index"
            )))
            }
        };
        let toks = tokenizer.tokenize(query);
        if toks.len() != 1 {
            return Err(CodecError::InvalidInput(format!(
                "read_column_byte_array_where_token: query produced {} tokens (expected 1; multi-token \
                 composition is a higher-level concern)",
                toks.len()
            )));
        }
        let hits = self.lookup_token(index_name, &toks[0])?;
        self.materialize_byte_array(source, index_name, &hits, target_column)
    }

    // ============================================================
    // bloom_probe + read_column_*_via_bloom_eq
    // ============================================================

    /// Probe a page-Bloom index with an equality query. Returns the
    /// `(rg, page)` pairs whose Bloom filter says the value *might*
    /// be present. False positives are possible (bounded by the
    /// `target_fpp` chosen at build time); false negatives are not.
    ///
    /// The returned list is ordered by `(rg, page)` ASC (the
    /// sidecar's build-time sort). Callers that want to materialize
    /// rows should use [`Self::read_column_i64_via_bloom_eq`] —
    /// `bloom_probe` is the lower-level primitive useful when
    /// combining the result with another index or with manual scan
    /// logic.
    pub fn bloom_probe(&self, index_name: &str, key: &Key<'_>) -> Result<Vec<(u32, u32)>> {
        let idx = self.find_index(index_name)?;
        let bloom = match &idx.data {
            LoadedIndexData::BloomPage(b) => b,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "bloom_probe: index `{index_name}` is not a page-Bloom index"
                )))
            }
        };
        // Hash the key per Parquet's spec (PLAIN-encoded bytes).
        let hash = match key {
            Key::I64(v) => parquet_xxh64(&v.to_le_bytes()),
            Key::I32(v) => parquet_xxh64(&v.to_le_bytes()),
            Key::Bytes(v) => parquet_xxh64(v),
        };
        let mut hits: Vec<(u32, u32)> = Vec::new();
        for i in 0..bloom.blooms.len() {
            let f = SplitBlockBloomFilter::from_bytes(&bloom.blooms[i])?;
            if f.contains_hash(hash) {
                hits.push((bloom.source_rgs[i] as u32, bloom.source_pages[i] as u32));
            }
        }
        Ok(hits)
    }

    /// `INT64` equality lookup via a page-Bloom index.
    ///
    /// `bloom_probe` returns *candidate* `(rg, page)` pairs — some
    /// of which may be false positives. This method, for each
    /// candidate page, masks all rows in the page as "decode me",
    /// calls `read_column_i64_masked_into` (which gets the v0.14.0
    /// skip-decompress-when-zero-popcount lever for *non*-matched
    /// pages for free), and then **filters the result** in memory
    /// for exact equality. The output is the same as what a sorted
    /// equality index would have returned, modulo CPU cost on
    /// false-positive pages.
    ///
    /// Use a Bloom-only index when the source has too many distinct
    /// values for a sorted index to be cost-effective, or when you
    /// only need pruning (no rowset granularity).
    pub fn read_column_i64_via_bloom_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        let candidate_pages = self.bloom_probe(index_name, &Key::I64(key))?;
        if candidate_pages.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve the indexed column's leaf ordinal so we can walk its
        // page layout and convert (rg, page) → first_row range.
        let idx = self.find_index(index_name)?;
        let source_col_name = match &idx.entry.kind {
            IndexKind::BloomPage { source_column, .. } => source_column.as_str(),
            _ => unreachable!("bloom_probe already validated the kind"),
        };
        let source_col_idx = resolve_leaf_by_name(source, source_col_name)?;

        // Group candidates by row group; build a chunk-wide bitmap
        // per row group with full rows set in the candidate pages.
        let md = source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("source metadata: {e}")))?;
        let mut by_rg: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (rg, page) in candidate_pages {
            by_rg.entry(rg).or_default().push(page);
        }

        let mut decoded: Vec<i64> = Vec::new();
        for (rg, pages) in by_rg {
            let rg_meta = md.row_groups.get(rg as usize).ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "bloom hit references row_group {rg} out of source range"
                ))
            })?;
            let n_rows = rg_meta.num_rows as usize;
            let mut bitmap = vec![0u8; n_rows.div_ceil(8)];

            // Walk indexed column to get page boundaries.
            let mut page_layouts: Vec<(usize, usize)> = Vec::new(); // (first_row, num_values)
            walk_data_pages(source, rg as usize, source_col_idx, |layout| {
                page_layouts.push((layout.first_row, layout.num_values));
                Ok(())
            })?;

            for page in pages {
                let (first_row, num_values) =
                    *page_layouts.get(page as usize).ok_or_else(|| {
                        CodecError::InvalidInput(format!(
                            "bloom hit references page {page} in rg {rg} but indexed column has only {} pages",
                            page_layouts.len()
                        ))
                    })?;
                let end = first_row + num_values;
                // Set bits [first_row, end). Byte-granular fast path
                // for the dense interior, bit-granular at the edges.
                set_range_bits(&mut bitmap, first_row, end);
            }

            crate::read::read_column_i64_masked_into(
                source,
                rg as usize,
                target_column,
                &bitmap,
                &mut decoded,
            )?;
        }

        // Filter for exact equality: Bloom has false positives, so
        // the decoded set may include extra rows from candidate pages
        // that didn't actually match.
        decoded.retain(|&v| v == key);
        Ok(decoded)
    }

    // ============================================================
    // Convenience: read_column_*_where_*
    // ============================================================

    /// Indexed equality + masked decode for an `INT64` target
    /// column. See type-level [`Self`] docs for the algorithm; the
    /// short version is: `lookup_eq` → group hits by row group →
    /// assemble chunk-wide bitmap → `read_column_i64_masked_into`.
    pub fn read_column_i64_where_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        let hits = self.lookup_eq(index_name, &Key::I64(key))?;
        self.materialize_i64(source, index_name, &hits, target_column)
    }

    /// Indexed range + masked decode for an `INT64` target column.
    pub fn read_column_i64_where_range(
        &self,
        source: &ParquetFile,
        index_name: &str,
        lo: i64,
        hi: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        let hits = self.lookup_range(index_name, &Key::I64(lo), &Key::I64(hi))?;
        self.materialize_i64(source, index_name, &hits, target_column)
    }

    /// Indexed equality + masked decode for an `INT32` target column.
    pub fn read_column_i32_where_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key: i32,
        target_column: usize,
    ) -> Result<Vec<i32>> {
        let hits = self.lookup_eq(index_name, &Key::I32(key))?;
        self.materialize_i32(source, index_name, &hits, target_column)
    }

    /// Indexed range + masked decode for an `INT32` target column.
    pub fn read_column_i32_where_range(
        &self,
        source: &ParquetFile,
        index_name: &str,
        lo: i32,
        hi: i32,
        target_column: usize,
    ) -> Result<Vec<i32>> {
        let hits = self.lookup_range(index_name, &Key::I32(lo), &Key::I32(hi))?;
        self.materialize_i32(source, index_name, &hits, target_column)
    }

    /// Indexed equality + masked decode for a `BYTE_ARRAY` target
    /// column. The lookup key is compared byte-wise (lex).
    pub fn read_column_byte_array_where_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key: &[u8],
        target_column: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let hits = self.lookup_eq(index_name, &Key::Bytes(key))?;
        self.materialize_byte_array(source, index_name, &hits, target_column)
    }

    /// Indexed range + masked decode for a `BYTE_ARRAY` target column.
    pub fn read_column_byte_array_where_range(
        &self,
        source: &ParquetFile,
        index_name: &str,
        lo: &[u8],
        hi: &[u8],
        target_column: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let hits = self.lookup_range(index_name, &Key::Bytes(lo), &Key::Bytes(hi))?;
        self.materialize_byte_array(source, index_name, &hits, target_column)
    }

    // ============================================================
    // Internal helpers
    // ============================================================

    fn find_index(&self, name: &str) -> Result<&LoadedIndex> {
        self.indexes
            .iter()
            .find(|i| i.entry.name == name)
            .ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "sidecar has no index named `{name}` (have: {:?})",
                    self.indexes
                        .iter()
                        .map(|i| &i.entry.name)
                        .collect::<Vec<_>>()
                ))
            })
    }

    fn materialize_i64(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
        target_column: usize,
    ) -> Result<Vec<i64>> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let bitmaps = self.assemble_bitmaps(source, index_name, hits)?;
        let mut out: Vec<i64> = Vec::new();
        for (rg, bitmap) in bitmaps {
            crate::read::read_column_i64_masked_into(
                source,
                rg as usize,
                target_column,
                &bitmap,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Composite-index sibling of [`Self::materialize_i64`]. Same
    /// body — `assemble_bitmaps` already dispatches on Sorted vs
    /// CompositePrefix to find the indexed column — but exposed as a
    /// separately-named helper so future composite-specific
    /// optimizations don't have to retrofit the sorted-only path.
    fn materialize_i64_composite(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
        target_column: usize,
    ) -> Result<Vec<i64>> {
        self.materialize_i64(source, index_name, hits, target_column)
    }

    fn materialize_i32(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
        target_column: usize,
    ) -> Result<Vec<i32>> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let bitmaps = self.assemble_bitmaps(source, index_name, hits)?;
        let mut out: Vec<i32> = Vec::new();
        for (rg, bitmap) in bitmaps {
            crate::read::read_column_i32_masked_into(
                source,
                rg as usize,
                target_column,
                &bitmap,
                &mut out,
            )?;
        }
        Ok(out)
    }

    fn materialize_byte_array(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
        target_column: usize,
    ) -> Result<Vec<Vec<u8>>> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let bitmaps = self.assemble_bitmaps(source, index_name, hits)?;
        let mut out: Vec<Vec<u8>> = Vec::new();
        for (rg, bitmap) in bitmaps {
            crate::read::read_column_byte_array_masked_into(
                source,
                rg as usize,
                target_column,
                &bitmap,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Given a list of `IndexHit`s, group them by source row group
    /// and assemble a chunk-wide row bitmap per group. The bitmap is
    /// sized to the row group's `num_rows`; bits are placed using
    /// the *indexed* column's page boundaries (since rowsets are
    /// page-relative to the indexed column).
    ///
    /// Works for both Sorted and CompositePrefix index kinds — the
    /// "indexed column" for a composite is `source_columns[0]` (the
    /// leading sort key, which is the column whose page layout
    /// rowsets are anchored to by the builder).
    fn assemble_bitmaps(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
    ) -> Result<Vec<(u32, Vec<u8>)>> {
        let idx = self.find_index(index_name)?;
        let source_col_name = match &idx.entry.kind {
            IndexKind::Sorted { source_column, .. } => source_column.as_str(),
            IndexKind::CompositePrefix { source_columns, .. } => {
                source_columns.first().map(String::as_str).ok_or_else(|| {
                    CodecError::InvalidInput(format!(
                        "composite index `{index_name}` has empty source_columns"
                    ))
                })?
            }
            IndexKind::Inverted { source_column, .. } => source_column.as_str(),
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "index `{index_name}` does not produce IndexHits"
                )))
            }
        };
        let source_col_idx = resolve_leaf_by_name(source, source_col_name)?;

        let md = source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("source metadata: {e}")))?;
        let mut by_rg: std::collections::BTreeMap<u32, Vec<&IndexHit>> =
            std::collections::BTreeMap::new();
        for h in hits {
            by_rg.entry(h.row_group).or_default().push(h);
        }

        let mut out: Vec<(u32, Vec<u8>)> = Vec::with_capacity(by_rg.len());
        for (rg, rg_hits) in by_rg {
            let rg_meta = md.row_groups.get(rg as usize).ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "index hit references row_group {rg} out of source range"
                ))
            })?;
            let n_rows = rg_meta.num_rows as usize;
            let mut bitmap = vec![0u8; n_rows.div_ceil(8)];

            let mut first_row_by_page: Vec<usize> = Vec::new();
            walk_data_pages(source, rg as usize, source_col_idx, |layout| {
                first_row_by_page.push(layout.first_row);
                Ok(())
            })?;

            for hit in rg_hits {
                let first_row = *first_row_by_page.get(hit.page as usize).ok_or_else(|| {
                    CodecError::InvalidInput(format!(
                        "index hit references page {} in rg {rg} but indexed column has only {} pages",
                        hit.page,
                        first_row_by_page.len()
                    ))
                })?;
                for (byte_idx, &b) in hit.rowset.iter().enumerate() {
                    for bit in 0..8 {
                        if (b >> bit) & 1 == 1 {
                            let row = first_row + byte_idx * 8 + bit;
                            if row < n_rows {
                                bitmap[row / 8] |= 1 << (row % 8);
                            }
                        }
                    }
                }
            }
            out.push((rg, bitmap));
        }
        Ok(out)
    }
}

// ============================================================
// Per-type lookup helpers — binary-search + duplicate-walk.
// ============================================================

fn eq_hits_i64(d: &LoadedTypedI64, target: i64, format: RowsetFormat) -> Result<Vec<IndexHit>> {
    let pos = match d.values.binary_search(&target) {
        Ok(i) => i,
        Err(_) => return Ok(Vec::new()),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1] == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end] == target {
        end += 1;
    }
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn eq_hits_i32(d: &LoadedTypedI32, target: i32, format: RowsetFormat) -> Result<Vec<IndexHit>> {
    let pos = match d.values.binary_search(&target) {
        Ok(i) => i,
        Err(_) => return Ok(Vec::new()),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1] == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end] == target {
        end += 1;
    }
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn eq_hits_bytes(
    d: &LoadedTypedBytes,
    target: &[u8],
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let pos = match d.values.binary_search_by(|v| v.as_slice().cmp(target)) {
        Ok(i) => i,
        Err(_) => return Ok(Vec::new()),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1].as_slice() == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end].as_slice() == target {
        end += 1;
    }
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn range_hits_i64(
    d: &LoadedTypedI64,
    range: RangeInclusive<i64>,
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let (lo, hi) = (*range.start(), *range.end());
    // First index with value >= lo.
    let start = d.values.partition_point(|v| *v < lo);
    // First index with value > hi.
    let end = d.values.partition_point(|v| *v <= hi);
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn range_hits_i32(
    d: &LoadedTypedI32,
    range: RangeInclusive<i32>,
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let (lo, hi) = (*range.start(), *range.end());
    let start = d.values.partition_point(|v| *v < lo);
    let end = d.values.partition_point(|v| *v <= hi);
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn range_hits_bytes(
    d: &LoadedTypedBytes,
    lo: &[u8],
    hi: &[u8],
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let start = d.values.partition_point(|v| v.as_slice() < lo);
    let end = d.values.partition_point(|v| v.as_slice() <= hi);
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn composite_eq_hits(
    d: &LoadedTypedCompositeI64I64,
    a: i64,
    b: i64,
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    // Two-step binary search:
    //   1. Find the run where values_a == a (partition by `< a` and `<= a`).
    //   2. Within that run, find the row where values_b == b.
    // Both runs are tiny once values_a is pinned, so a linear scan
    // for the second step is fine and avoids a second sorted-by-b
    // contract requirement.
    let a_start = d.values_a.partition_point(|v| *v < a);
    let a_end = d.values_a.partition_point(|v| *v <= a);
    if a_start == a_end {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    // Within the a-run, the rows are sorted by (a, b, rg, page).
    // Since values_a is constant, sort within the run is by (b, rg, page).
    // Use binary search on b for efficiency.
    let b_slice = &d.values_b[a_start..a_end];
    let b_start_local = b_slice.partition_point(|v| *v < b);
    let b_end_local = b_slice.partition_point(|v| *v <= b);
    if b_start_local == b_end_local {
        return Ok(out);
    }
    out.extend(collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        a_start + b_start_local,
        a_start + b_end_local,
        format,
    )?);
    Ok(out)
}

fn inverted_eq_hits(
    d: &LoadedTypedInverted,
    token: &[u8],
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    // `tokens` is sorted lex-ASC; binary search finds any row with
    // the target token, then linear scan covers duplicates of the
    // same token across pages.
    let pos = match d.tokens.binary_search_by(|t| t.as_slice().cmp(token)) {
        Ok(i) => i,
        Err(_) => return Ok(Vec::new()),
    };
    let mut start = pos;
    while start > 0 && d.tokens[start - 1].as_slice() == token {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.tokens.len() && d.tokens[end].as_slice() == token {
        end += 1;
    }
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn composite_prefix_hits(
    d: &LoadedTypedCompositeI64I64,
    a: i64,
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let start = d.values_a.partition_point(|v| *v < a);
    let end = d.values_a.partition_point(|v| *v <= a);
    collect_hits(
        &d.target_rgs,
        &d.target_pages,
        &d.rowsets,
        start,
        end,
        format,
    )
}

fn collect_hits(
    target_rgs: &[i32],
    target_pages: &[i32],
    rowsets: &[Vec<u8>],
    start: usize,
    end: usize,
    format: RowsetFormat,
) -> Result<Vec<IndexHit>> {
    let mut out = Vec::with_capacity(end.saturating_sub(start));
    for i in start..end {
        out.push(IndexHit {
            row_group: target_rgs[i] as u32,
            page: target_pages[i] as u32,
            // Normalize to the packed-bitmap contract at hit time —
            // hits exist only for the looked-up key, so this touches
            // a handful of rowsets however large the index is.
            rowset: to_bitmap(&rowsets[i], format)?,
        });
    }
    Ok(out)
}

// ============================================================
// Sanity helpers.
// ============================================================

fn check_aligned(
    name: &str,
    rg: usize,
    n_values: usize,
    target_rgs: &[i32],
    target_pages: &[i32],
    rowsets: &[Vec<u8>],
) -> Result<()> {
    if target_rgs.len() != n_values || target_pages.len() != n_values || rowsets.len() != n_values {
        return Err(CodecError::InvalidInput(format!(
            "index `{name}` row group {rg} has mismatched column lengths \
             (value={n_values}, rg={}, page={}, rowset={})",
            target_rgs.len(),
            target_pages.len(),
            rowsets.len(),
        )));
    }
    Ok(())
}

fn sorted_physical_type(entry: &IndexEntry) -> Result<PhysicalType> {
    match &entry.kind {
        IndexKind::Sorted { physical_type, .. } => Ok(*physical_type),
        _ => Err(CodecError::InvalidInput(format!(
            "index `{}` is not a sorted index",
            entry.name
        ))),
    }
}

fn codec_err(e: ManifestError) -> CodecError {
    CodecError::InvalidInput(format!("{e}"))
}

/// Set bits `[start_bit, end_bit)` in a packed bitmap. Used by the
/// page-Bloom convenience entry to mark all rows in candidate
/// pages "decode me".
fn set_range_bits(bitmap: &mut [u8], start_bit: usize, end_bit: usize) {
    if start_bit >= end_bit {
        return;
    }
    let n_bits = bitmap.len() * 8;
    let end_bit = end_bit.min(n_bits);
    let start_bit = start_bit.min(end_bit);

    // Head: bit-by-bit until aligned to a byte boundary.
    let mut bit = start_bit;
    while bit < end_bit && bit % 8 != 0 {
        bitmap[bit / 8] |= 1 << (bit % 8);
        bit += 1;
    }
    // Body: whole bytes (= 0xFF).
    while bit + 8 <= end_bit {
        bitmap[bit / 8] = 0xFF;
        bit += 8;
    }
    // Tail: bit-by-bit.
    while bit < end_bit {
        bitmap[bit / 8] |= 1 << (bit % 8);
        bit += 1;
    }
}

fn resolve_leaf_by_name(source: &ParquetFile, name: &str) -> Result<usize> {
    let md = source
        .metadata()
        .map_err(|e| CodecError::InvalidInput(format!("source metadata: {e}")))?;
    for (i, se) in md.schema.iter().enumerate().skip(1) {
        if se.name == name.as_bytes() {
            return Ok(i - 1);
        }
    }
    Err(CodecError::InvalidInput(format!(
        "indexed column `{name}` not found in source file's schema"
    )))
}
