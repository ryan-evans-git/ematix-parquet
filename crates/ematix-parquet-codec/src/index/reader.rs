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

use crate::error::{CodecError, Result};
use crate::index::fingerprint::compute_source_fingerprint;
use crate::index::manifest::{IndexEntry, IndexKind, IndexManifest, ManifestError, MANIFEST_KEY};
use crate::index::page_layout::walk_data_pages;
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

#[derive(Debug)]
enum LoadedSorted {
    I64(LoadedTypedI64),
    I32(LoadedTypedI32),
    Bytes(LoadedTypedBytes),
}

#[derive(Debug)]
struct LoadedIndex {
    entry: IndexEntry,
    data: LoadedSorted,
}

/// Parsed sidecar parquet. Owns the manifest and the in-memory index
/// data; lookups are pure-CPU after `open` returns.
#[derive(Debug)]
pub struct ParquetIndex {
    manifest: IndexManifest,
    indexes: Vec<LoadedIndex>,
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
        let manifest_kv = kvs
            .iter()
            .find(|kv| kv.key == MANIFEST_KEY.as_bytes())
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
                            LoadedSorted::I64(LoadedTypedI64 {
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
                            LoadedSorted::I32(LoadedTypedI32 {
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
                            LoadedSorted::Bytes(LoadedTypedBytes {
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
                // Other kinds reserved for Π.19+ (BloomPage,
                // CompositePrefix, Inverted). Skipping is forward-compat
                // friendly — a request for one of them through
                // `lookup_eq` will fail loud at lookup time.
                _ => continue,
            }
        }

        Ok(Self { manifest, indexes })
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
            (LoadedSorted::I64(d), Key::I64(v)) => Ok(eq_hits_i64(d, *v)),
            (LoadedSorted::I32(d), Key::I32(v)) => Ok(eq_hits_i32(d, *v)),
            (LoadedSorted::Bytes(d), Key::Bytes(v)) => Ok(eq_hits_bytes(d, v)),
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
            (LoadedSorted::I64(d), Key::I64(a), Key::I64(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                Ok(range_hits_i64(d, *a..=*b))
            }
            (LoadedSorted::I32(d), Key::I32(a), Key::I32(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                Ok(range_hits_i32(d, *a..=*b))
            }
            (LoadedSorted::Bytes(d), Key::Bytes(a), Key::Bytes(b)) => {
                if a > b {
                    return Ok(Vec::new());
                }
                Ok(range_hits_bytes(d, a, b))
            }
            _ => Err(CodecError::InvalidInput(format!(
                "lookup_range: key/index type mismatch on `{index_name}`"
            ))),
        }
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
    fn assemble_bitmaps(
        &self,
        source: &ParquetFile,
        index_name: &str,
        hits: &[IndexHit],
    ) -> Result<Vec<(u32, Vec<u8>)>> {
        let idx = self.find_index(index_name)?;
        let source_col_name = match &idx.entry.kind {
            IndexKind::Sorted { source_column, .. } => source_column.as_str(),
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "index `{index_name}` is not a sorted index"
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

fn eq_hits_i64(d: &LoadedTypedI64, target: i64) -> Vec<IndexHit> {
    let pos = match d.values.binary_search(&target) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1] == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end] == target {
        end += 1;
    }
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn eq_hits_i32(d: &LoadedTypedI32, target: i32) -> Vec<IndexHit> {
    let pos = match d.values.binary_search(&target) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1] == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end] == target {
        end += 1;
    }
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn eq_hits_bytes(d: &LoadedTypedBytes, target: &[u8]) -> Vec<IndexHit> {
    let pos = match d.values.binary_search_by(|v| v.as_slice().cmp(target)) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let mut start = pos;
    while start > 0 && d.values[start - 1].as_slice() == target {
        start -= 1;
    }
    let mut end = pos + 1;
    while end < d.values.len() && d.values[end].as_slice() == target {
        end += 1;
    }
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn range_hits_i64(d: &LoadedTypedI64, range: RangeInclusive<i64>) -> Vec<IndexHit> {
    let (lo, hi) = (*range.start(), *range.end());
    // First index with value >= lo.
    let start = d.values.partition_point(|v| *v < lo);
    // First index with value > hi.
    let end = d.values.partition_point(|v| *v <= hi);
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn range_hits_i32(d: &LoadedTypedI32, range: RangeInclusive<i32>) -> Vec<IndexHit> {
    let (lo, hi) = (*range.start(), *range.end());
    let start = d.values.partition_point(|v| *v < lo);
    let end = d.values.partition_point(|v| *v <= hi);
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn range_hits_bytes(d: &LoadedTypedBytes, lo: &[u8], hi: &[u8]) -> Vec<IndexHit> {
    let start = d.values.partition_point(|v| v.as_slice() < lo);
    let end = d.values.partition_point(|v| v.as_slice() <= hi);
    collect_hits(&d.target_rgs, &d.target_pages, &d.rowsets, start, end)
}

fn collect_hits(
    target_rgs: &[i32],
    target_pages: &[i32],
    rowsets: &[Vec<u8>],
    start: usize,
    end: usize,
) -> Vec<IndexHit> {
    let mut out = Vec::with_capacity(end.saturating_sub(start));
    for i in start..end {
        out.push(IndexHit {
            row_group: target_rgs[i] as u32,
            page: target_pages[i] as u32,
            rowset: rowsets[i].clone(),
        });
    }
    out
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
