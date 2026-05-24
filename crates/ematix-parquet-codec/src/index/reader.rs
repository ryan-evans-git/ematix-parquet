//! Sidecar-index reader. Π.17b ships **sorted INT64** lookups; Π.18+
//! widens to the other physical types and to range queries on top of
//! the same data structures.
//!
//! `ParquetIndex::open` parses the sidecar parquet's footer
//! `KeyValueMetadata`, verifies the embedded fingerprint against the
//! source, and **eagerly** decodes all index entries into memory. This
//! is the simplest correct implementation for MVP: sidecars built from
//! TPC-H-shaped sources are tens-of-MB at most, so the eager-load cost
//! is well under a millisecond. The Iceberg-layer work (Π.21+) will
//! introduce per-manifest summaries that prune which sidecars even
//! need opening, so eager-load-per-sidecar stays scalable in the
//! dataset case too.
//!
//! `lookup_eq` is `log(n)` via binary search on the sorted value
//! column, plus linear scan over duplicates of the same key (a
//! low-cardinality column with many rows-per-key has long duplicate
//! runs, but they're tight — same value, three small ints per row).
//!
//! `read_column_i64_where_eq` is the convenience that ties the lookup
//! into the existing `read_column_i64_masked_into` path: it groups
//! hits by row group, assembles a chunk-wide bitmap from per-page
//! rowsets, and lets the codec's existing zero-popcount-skip
//! machinery drop pages whose mask is empty.

use std::path::Path;

use ematix_parquet_io::ParquetFile;

use crate::error::{CodecError, Result};
use crate::index::fingerprint::compute_source_fingerprint;
use crate::index::manifest::{IndexEntry, IndexKind, IndexManifest, ManifestError, MANIFEST_KEY};
use crate::index::page_layout::walk_data_pages;
use crate::index::types::{IndexHit, Key};
use crate::index::PhysicalType;
use crate::read::{read_column_byte_array, read_column_i32, read_column_i64};

/// One loaded index, kept fully in memory. The four parallel vectors
/// are aligned by row: row `i` of the underlying sidecar parquet
/// becomes `(values[i], target_rgs[i], target_pages[i], rowsets[i])`.
#[derive(Debug)]
struct LoadedSortedI64 {
    entry: IndexEntry,
    values: Vec<i64>,
    target_rgs: Vec<i32>,
    target_pages: Vec<i32>,
    rowsets: Vec<Vec<u8>>,
}

/// Parsed sidecar parquet. Owns the manifest and the in-memory index
/// data; lookups are pure-CPU after `open` returns.
#[derive(Debug)]
pub struct ParquetIndex {
    manifest: IndexManifest,
    indexes: Vec<LoadedSortedI64>,
}

impl ParquetIndex {
    /// Open `idx_path`, parse and validate the manifest against
    /// `source`, eagerly load every index row group. Errors fast on:
    /// - missing or malformed manifest (`ManifestError`)
    /// - manifest version other than `v1`
    /// - source-fingerprint mismatch (sidecar built against a
    ///   different state of the source file)
    /// - schema shape on the sidecar that doesn't match what a
    ///   sorted-i64 index emits
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
        let mut indexes: Vec<LoadedSortedI64> = Vec::with_capacity(manifest.indexes.len());
        for entry in &manifest.indexes {
            match &entry.kind {
                IndexKind::Sorted {
                    physical_type: PhysicalType::Int64,
                    ..
                } => {
                    let rg = entry.sidecar_row_group as usize;
                    // Schema check: column 0 = value (INT64), 1 = target_rg (INT32),
                    // 2 = target_page (INT32), 3 = target_rowset (BYTE_ARRAY).
                    // If the column count or types disagree the typed-read
                    // calls below will error.
                    let values = read_column_i64(&idx_file, rg, 0)?;
                    let target_rgs = read_column_i32(&idx_file, rg, 1)?;
                    let target_pages = read_column_i32(&idx_file, rg, 2)?;
                    let target_rowsets_borrowed = read_column_byte_array(&idx_file, rg, 3)?;
                    let target_rowsets: Vec<Vec<u8>> =
                        target_rowsets_borrowed.into_iter().collect();

                    // Sanity: all four columns share length.
                    let n = values.len();
                    if target_rgs.len() != n || target_pages.len() != n || target_rowsets.len() != n
                    {
                        return Err(CodecError::InvalidInput(format!(
                            "index `{}` row group {} has mismatched column lengths \
                             (value={}, rg={}, page={}, rowset={})",
                            entry.name,
                            rg,
                            n,
                            target_rgs.len(),
                            target_pages.len(),
                            target_rowsets.len()
                        )));
                    }
                    indexes.push(LoadedSortedI64 {
                        entry: entry.clone(),
                        values,
                        target_rgs,
                        target_pages,
                        rowsets: target_rowsets,
                    });
                }
                // Other kinds are reserved for Π.18+ and not yet
                // populated by any builder. Defensive: skip rather
                // than error so a future sidecar with mixed kinds
                // doesn't break older readers (the lookup-by-name
                // path will then fail loud on requests for that
                // index).
                _ => continue,
            }
        }

        Ok(Self { manifest, indexes })
    }

    /// Borrow the parsed manifest. Useful for tooling that wants to
    /// inspect the sidecar (which indexes exist, what columns, etc.).
    pub fn manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    /// Equality lookup. Returns every (rg, page, rowset-within-page)
    /// triple in the source file that has at least one row matching
    /// `key`.
    ///
    /// Empty result = no rows match. The reader does NOT verify that
    /// the hit pages still contain those rows — the fingerprint
    /// check at `open` time is the authoritative guard against source
    /// drift.
    pub fn lookup_eq(&self, index_name: &str, key: &Key<'_>) -> Result<Vec<IndexHit>> {
        let idx = self
            .indexes
            .iter()
            .find(|i| i.entry.name == index_name)
            .ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "sidecar has no index named `{index_name}` (have: {:?})",
                    self.indexes
                        .iter()
                        .map(|i| &i.entry.name)
                        .collect::<Vec<_>>()
                ))
            })?;
        let pt = match &idx.entry.kind {
            IndexKind::Sorted { physical_type, .. } => *physical_type,
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "index `{index_name}` is not a sorted index"
                )))
            }
        };
        if !key.matches_physical_type(pt) {
            return Err(CodecError::InvalidInput(format!(
                "lookup_eq: key variant `{}` does not match index `{}` physical type {:?}",
                key.variant_name(),
                index_name,
                pt,
            )));
        }
        let target = match key {
            Key::I64(v) => *v,
            other => {
                return Err(CodecError::InvalidInput(format!(
                    "Π.17b only supports Key::I64; got {}",
                    other.variant_name()
                )))
            }
        };

        // Sorted values → binary search for ANY index whose value
        // == target, then linear-scan both directions to cover
        // duplicates.
        let idx_pos = match idx.values.binary_search(&target) {
            Ok(i) => i,
            Err(_) => return Ok(Vec::new()),
        };
        // Walk backwards to the first duplicate.
        let mut start = idx_pos;
        while start > 0 && idx.values[start - 1] == target {
            start -= 1;
        }
        // Walk forwards to the last duplicate.
        let mut end = idx_pos + 1;
        while end < idx.values.len() && idx.values[end] == target {
            end += 1;
        }

        let mut hits = Vec::with_capacity(end - start);
        for i in start..end {
            hits.push(IndexHit {
                row_group: idx.target_rgs[i] as u32,
                page: idx.target_pages[i] as u32,
                rowset: idx.rowsets[i].clone(),
            });
        }
        Ok(hits)
    }

    /// Convenience: index the named INT64 column for `key`, then
    /// decode `target_column` (any physical type the codec can read
    /// via `read_column_i64_masked_into` — this entry point is
    /// specialized to INT64 target columns in Π.17b; sibling entries
    /// for other target types arrive in Π.18+).
    ///
    /// Internally:
    /// 1. `lookup_eq` → `Vec<IndexHit>`.
    /// 2. Group hits by `row_group`.
    /// 3. For each row group, derive page boundaries on the *indexed*
    ///    column (via [`walk_data_pages`]) and OR each hit's
    ///    page-relative rowset into a chunk-wide bitmap at the page's
    ///    `first_row` offset.
    /// 4. Call `read_column_i64_masked_into(source, rg, target_column, &bitmap, &mut out)`.
    ///    Pages whose bitmap range has zero popcount get skipped
    ///    before decompression — the v0.14.0 cross-column page-skip
    ///    lever fires automatically.
    pub fn read_column_i64_where_eq(
        &self,
        source: &ParquetFile,
        index_name: &str,
        key: i64,
        target_column: usize,
    ) -> Result<Vec<i64>> {
        // Resolve the indexed column ordinal by name. (Builder writes
        // the leaf name; resolver walks the source's schema.)
        let idx = self
            .indexes
            .iter()
            .find(|i| i.entry.name == index_name)
            .ok_or_else(|| {
                CodecError::InvalidInput(format!("sidecar has no index named `{index_name}`"))
            })?;
        let source_col_name = match &idx.entry.kind {
            IndexKind::Sorted { source_column, .. } => source_column.as_str(),
            _ => {
                return Err(CodecError::InvalidInput(format!(
                    "index `{index_name}` is not a sorted index"
                )))
            }
        };
        let source_col_idx = resolve_leaf_by_name(source, source_col_name)?;

        // Run the lookup.
        let hits = self.lookup_eq(index_name, &Key::I64(key))?;
        if hits.is_empty() {
            return Ok(Vec::new());
        }

        // Group hits by row group.
        let md = source
            .metadata()
            .map_err(|e| CodecError::InvalidInput(format!("source metadata: {e}")))?;
        let mut by_rg: std::collections::BTreeMap<u32, Vec<&IndexHit>> =
            std::collections::BTreeMap::new();
        for h in &hits {
            by_rg.entry(h.row_group).or_default().push(h);
        }

        let mut out: Vec<i64> = Vec::new();
        for (rg, rg_hits) in by_rg {
            let rg_meta = md.row_groups.get(rg as usize).ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "index hit references row_group {rg} out of source range"
                ))
            })?;
            let n_rows = rg_meta.num_rows as usize;
            let mut bitmap = vec![0u8; n_rows.div_ceil(8)];

            // Walk the indexed column's pages for this RG to learn
            // first_row per page.
            let mut first_row_by_page: Vec<usize> = Vec::new();
            walk_data_pages(source, rg as usize, source_col_idx, |layout| {
                first_row_by_page.push(layout.first_row);
                Ok(())
            })?;

            // OR each per-page rowset into the chunk-wide bitmap.
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

            // Pull matching rows via the existing masked-decode path.
            // Zero-popcount pages are skipped before decompression by
            // `decode_chunk_row_masked_into`.
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
}

/// Map a `ManifestError` into a `CodecError` for uniform error
/// surfacing at the public API.
fn codec_err(e: ManifestError) -> CodecError {
    CodecError::InvalidInput(format!("{e}"))
}

/// Walk the source file's depth-first schema list and return the
/// leaf-column ordinal whose name matches `name`. Only flat REQUIRED
/// schemas are supported in Π.17b (every TPC-H reference shape);
/// nested-column path lookup arrives with Π.20+.
fn resolve_leaf_by_name(source: &ParquetFile, name: &str) -> Result<usize> {
    let md = source
        .metadata()
        .map_err(|e| CodecError::InvalidInput(format!("source metadata: {e}")))?;
    // schema[0] is the root group; subsequent entries are the leaves
    // in depth-first order for a flat schema. Match by name.
    for (i, se) in md.schema.iter().enumerate().skip(1) {
        if se.name == name.as_bytes() {
            return Ok(i - 1);
        }
    }
    Err(CodecError::InvalidInput(format!(
        "indexed column `{name}` not found in source file's schema"
    )))
}
