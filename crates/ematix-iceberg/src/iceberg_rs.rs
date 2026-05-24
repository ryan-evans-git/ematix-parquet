//! `iceberg-rust` integration — encode/decode the ematix extension on
//! real [`iceberg::spec::DataFile`] entries, and prune-by-summary
//! helpers over a collection of files.
//!
//! Gated behind the `iceberg` Cargo feature so the default build of
//! `ematix-iceberg` stays dep-light (the iceberg-rust dep tree adds
//! tokio + opendal + arrow + ~100 transitives).
//!
//! ## Where the extension lives on a `DataFile`
//!
//! Iceberg's `DataFile` struct exposes a single per-file binary
//! channel that's writeable from outside iceberg-rust core:
//! [`DataFile::key_metadata`]. The spec describes it as
//! "Implementation-specific key metadata for encryption" — i.e. the
//! content is opaque to the table-format layer. We co-opt it,
//! distinguishing our payloads from any legitimate encryption
//! metadata via a 4-byte magic prefix:
//!
//! ```text
//! key_metadata bytes := b"EMTX" || <UTF-8 JSON for EmatixDataFileExtension>
//! ```
//!
//! [`decode_key_metadata`] returns `Ok(None)` when the magic isn't
//! present — that's our signal to defer to the original "encryption
//! metadata" interpretation, never a parse error. Only a magic-
//! prefixed payload with malformed JSON returns `Err`.
//!
//! Encryption-aware Iceberg deployments that already use
//! `key_metadata` for PME-style key wrapping must either:
//!
//! 1. Migrate to a separate channel (Iceberg table properties keyed
//!    by data-file path — a Π.21c follow-up), or
//! 2. Stop using `key_metadata` for encryption (e.g. by moving keys
//!    into a sibling KMS).
//!
//! Most workloads don't use PME and have `key_metadata = None` on
//! every file; for those, this works today with no migration.

use iceberg::spec::{DataContentType, DataFile, DataFileBuilder};
use iceberg::table::Table;

use crate::error::{IcebergIndexError, Result};
use crate::extension::EmatixDataFileExtension;
use crate::summary::IndexSummary;
use ematix_parquet_codec::index::Key;

/// 4-byte sentinel that prefixes our `key_metadata` payload. Lets
/// readers distinguish an ematix extension from any legitimate
/// encryption metadata that some other producer might have put in
/// the same field. Bumping the magic is a v2 escape hatch — for v1
/// it stays exactly `b"EMTX"`.
pub const KEY_METADATA_MAGIC: &[u8; 4] = b"EMTX";

/// Encode an extension into the bytes that go into
/// [`DataFile::key_metadata`]. The result is `MAGIC || utf-8(JSON)`.
pub fn encode_key_metadata(ext: &EmatixDataFileExtension) -> Vec<u8> {
    let json = ext.to_json();
    let mut out = Vec::with_capacity(KEY_METADATA_MAGIC.len() + json.len());
    out.extend_from_slice(KEY_METADATA_MAGIC);
    out.extend_from_slice(json.as_bytes());
    out
}

/// Inverse of [`encode_key_metadata`]. Returns:
///
/// - `Ok(None)` if `bytes` doesn't start with [`KEY_METADATA_MAGIC`].
///   This is the *expected* result for files whose `key_metadata`
///   holds genuine encryption metadata, *not* an error.
/// - `Ok(Some(ext))` if the magic matches and the JSON parses.
/// - `Err(_)` only when the magic matches but the JSON is malformed
///   or carries the wrong version.
pub fn decode_key_metadata(bytes: &[u8]) -> Result<Option<EmatixDataFileExtension>> {
    if bytes.len() < KEY_METADATA_MAGIC.len()
        || &bytes[..KEY_METADATA_MAGIC.len()] != KEY_METADATA_MAGIC
    {
        return Ok(None);
    }
    let json_bytes = &bytes[KEY_METADATA_MAGIC.len()..];
    let json = std::str::from_utf8(json_bytes).map_err(|e| {
        IcebergIndexError::Malformed(format!("key_metadata after magic is not UTF-8: {e}"))
    })?;
    EmatixDataFileExtension::from_json(json).map(Some)
}

/// Pull the extension off a [`DataFile`], if present. Returns
/// `Ok(None)` if the file has no `key_metadata` at all or if the
/// bytes don't carry our magic prefix (i.e. someone else's payload).
///
/// This is the read-side counterpart to [`attach_extension`].
pub fn extract_extension(df: &DataFile) -> Result<Option<EmatixDataFileExtension>> {
    match df.key_metadata() {
        Some(bytes) => decode_key_metadata(bytes),
        None => Ok(None),
    }
}

/// Embed the extension in a `DataFileBuilder` via its `key_metadata`
/// setter. The returned builder is identical to the input except
/// `key_metadata` is set to `Some(MAGIC || JSON)`.
///
/// Consumers compose this into their normal `DataFileBuilder` flow
/// alongside `content`, `file_path`, etc. See the unit tests for a
/// minimal end-to-end example.
pub fn attach_extension(
    mut builder: DataFileBuilder,
    ext: &EmatixDataFileExtension,
) -> DataFileBuilder {
    builder.key_metadata(Some(encode_key_metadata(ext)));
    builder
}

// ============================================================
// File-level pruning helpers
// ============================================================

/// File-level equality prune. Iterates the input data files,
/// extracts each one's [`EmatixDataFileExtension`] (if any), and
/// returns the subset whose summary for `index_name` "could contain"
/// the query key.
///
/// A file is **kept** in any of these cases:
/// 1. Its extension's summary for `index_name` returns `true` from
///    [`IndexSummary::could_contain_eq`].
/// 2. Its extension is present but lacks a summary for `index_name`
///    (no info → conservative keep).
/// 3. It has no ematix extension at all (no info → conservative keep).
///
/// A file is **dropped** only when its summary explicitly says the
/// key falls outside `[min_key, max_key]`. False negatives are
/// impossible by construction.
///
/// Errors propagate if any file's extension exists but is malformed
/// (magic-prefixed payload with bad JSON), or if the summary's
/// physical type doesn't match the query key (planner error).
pub fn prune_data_files_eq<'a>(
    files: &'a [DataFile],
    index_name: &str,
    key: &Key<'_>,
) -> Result<Vec<&'a DataFile>> {
    let mut kept = Vec::with_capacity(files.len());
    for df in files {
        let ext = match extract_extension(df)? {
            Some(e) => e,
            None => {
                // No extension at all → conservative keep.
                kept.push(df);
                continue;
            }
        };
        let summary = match ext.summary(index_name) {
            Some(s) => s,
            None => {
                // Extension present, no summary for this index → keep.
                kept.push(df);
                continue;
            }
        };
        if summary.could_contain_eq(key)? {
            kept.push(df);
        }
    }
    Ok(kept)
}

/// File-level range prune. Same shape as [`prune_data_files_eq`] but
/// against an inclusive `[low, high]` window with either end
/// optionally open.
pub fn prune_data_files_range<'a>(
    files: &'a [DataFile],
    index_name: &str,
    low: Option<&Key<'_>>,
    high: Option<&Key<'_>>,
) -> Result<Vec<&'a DataFile>> {
    let mut kept = Vec::with_capacity(files.len());
    for df in files {
        let ext = match extract_extension(df)? {
            Some(e) => e,
            None => {
                kept.push(df);
                continue;
            }
        };
        let summary = match ext.summary(index_name) {
            Some(s) => s,
            None => {
                kept.push(df);
                continue;
            }
        };
        if summary.could_contain_range(low, high)? {
            kept.push(df);
        }
    }
    Ok(kept)
}

/// Convenience: extract the summary for `index_name` from a single
/// [`DataFile`], returning `Ok(None)` if either the extension or the
/// per-index summary is absent. Useful when callers want to inspect
/// the summary themselves (e.g. to log pruning decisions) rather
/// than just take the boolean outcome.
pub fn summary_for(
    df: &DataFile,
    index_name: &str,
) -> Result<Option<(IndexSummary, EmatixDataFileExtension)>> {
    match extract_extension(df)? {
        Some(ext) => {
            let s = ext.summary(index_name).cloned();
            Ok(s.map(|s| (s, ext)))
        }
        None => Ok(None),
    }
}

// ============================================================
// Sidecar URI resolution + pruned-file candidates (Π.22a)
// ============================================================

/// Join a sidecar path against the data file's directory, producing
/// the URI a sidecar reader would open.
///
/// Rules (in order):
/// 1. If `relative` already names an absolute URI (contains `"://"`)
///    or starts with `/`, it is returned unchanged. Producers that
///    write absolute sidecar paths get back exactly what they wrote.
/// 2. Otherwise the prefix of `data_file_uri` up to (and excluding)
///    the final `/` is joined with `relative`. Works uniformly for
///    `s3://bucket/dir/file.parquet`, `file:///abs/file.parquet`,
///    and `/abs/file.parquet`.
/// 3. If `data_file_uri` has no `/` at all, `relative` is returned
///    unchanged (no directory to anchor to).
///
/// Resolution is purely textual — no canonicalization, no `..`
/// support, no scheme-aware normalization. The producer side is
/// responsible for writing sane relative paths; the consumer side
/// gets a 1-line transformation.
pub fn resolve_sidecar_uri(data_file_uri: &str, relative: &str) -> String {
    if relative.starts_with('/') || relative.contains("://") {
        return relative.to_string();
    }
    match data_file_uri.rfind('/') {
        Some(idx) => {
            let mut out = String::with_capacity(idx + 1 + relative.len());
            out.push_str(&data_file_uri[..idx]);
            out.push('/');
            out.push_str(relative);
            out
        }
        None => relative.to_string(),
    }
}

/// A [`DataFile`] that survived pruning, bundled with its ematix
/// extension and the resolved URI of the per-file sidecar.
///
/// This is the unit of work a query executor iterates over: each
/// candidate is one file to open with [`ematix_parquet_codec`]'s
/// `ParquetFile::open(<file_path>)` followed by
/// `ParquetIndex::open(<sidecar_uri>, &source)`, then a sidecar
/// lookup against the chosen index.
///
/// Owned (not borrowed) because the producer-side pipeline often
/// consumes its input `Vec<DataFile>` to produce candidates and
/// then hands them off to async I/O — borrows would tangle
/// lifetimes with the iceberg manifest walker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunedDataFile {
    /// Original Iceberg manifest entry; carries `file_path` (URI),
    /// `record_count`, partition tuple, etc.
    pub data_file: DataFile,
    /// Decoded ematix extension. Always present — files without
    /// our extension don't become `PrunedDataFile`s (see
    /// [`pair_with_extensions`]).
    pub extension: EmatixDataFileExtension,
    /// Fully resolved sidecar URI (output of [`resolve_sidecar_uri`]).
    /// Pre-computed so the executor doesn't repeat the resolution per
    /// query against the same candidate set.
    pub sidecar_uri: String,
}

/// Pair each [`DataFile`] with its ematix extension and the resolved
/// sidecar URI, **dropping files that lack our extension**. The
/// "drop on missing extension" choice is the only place in this
/// crate that's not conservative — the contract is "candidates are
/// files we can plan against", and we can't plan against a file
/// with no sidecar pointer.
///
/// Callers that want to keep "no extension" files (e.g. for a
/// fallback full-scan path) should use [`prune_data_files_eq`] /
/// [`prune_data_files_range`] directly and pair manually.
///
/// Errors only when an extension *is* present but malformed —
/// `Ok(None)` from [`extract_extension`] is normal and silently
/// drops the file.
pub fn pair_with_extensions(files: Vec<DataFile>) -> Result<Vec<PrunedDataFile>> {
    let mut out = Vec::with_capacity(files.len());
    for df in files {
        let Some(ext) = extract_extension(&df)? else {
            continue;
        };
        let sidecar_uri = resolve_sidecar_uri(df.file_path(), &ext.sidecar_relative_path);
        out.push(PrunedDataFile {
            data_file: df,
            extension: ext,
            sidecar_uri,
        });
    }
    Ok(out)
}

// ============================================================
// Async manifest walker (Π.22b)
// ============================================================

/// Walk all data files in the table's **current snapshot**,
/// returning them as a flat `Vec<DataFile>`.
///
/// This is the natural input to [`prune_data_files_eq`] /
/// [`prune_data_files_range`] / [`pair_with_extensions`]: most
/// query planners want one flat candidate set per query, not a
/// nested manifest-by-manifest traversal.
///
/// **Snapshot scope.** Only the current snapshot is walked. Iceberg
/// time-travel queries that target a non-current snapshot need to
/// re-clone the [`Table`] with `with_metadata` pointing at that
/// snapshot, then call this function — keeps the snapshot-selection
/// concern outside the walker.
///
/// **Liveness.** Only entries with [`ManifestStatus::Added`] or
/// [`Existing`] are kept (via [`ManifestEntry::is_alive`]); deletes
/// are skipped. Only files with [`DataContentType::Data`] are
/// returned — equality and position delete files are filtered out
/// at this layer (handling them is the executor's job, not the
/// planner's prune step).
///
/// **Empty tables.** Returns an empty `Vec` (not an error) when the
/// table has no current snapshot — e.g. a newly created table
/// before its first commit.
///
/// **I/O.** Each manifest file is loaded via the table's
/// [`Table::file_io`] handle, which can point at the local FS, S3,
/// GCS, or in-memory storage depending on how the catalog created
/// it. Concurrent fetch is *not* done here; loads are serial. For
/// large fan-outs, callers can wrap this in `try_join_all` over
/// per-manifest tasks themselves.
///
/// [`ManifestStatus::Added`]: iceberg::spec::ManifestStatus::Added
/// [`Existing`]: iceberg::spec::ManifestStatus::Existing
/// [`ManifestEntry::is_alive`]: iceberg::spec::ManifestEntry::is_alive
pub async fn collect_data_files(table: &Table) -> Result<Vec<DataFile>> {
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(Vec::new());
    };
    let file_io = table.file_io();
    let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;

    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(file_io).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != DataContentType::Data {
                continue;
            }
            files.push(df.clone());
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summary::SummaryKey;
    use iceberg::spec::{DataContentType, DataFileFormat, Struct};

    /// Build a minimal `DataFile` for tests. Required fields filled
    /// with sentinel values; `key_metadata` left to the caller.
    fn make_data_file(file_path: &str, key_metadata: Option<Vec<u8>>) -> DataFile {
        let mut b = DataFileBuilder::default();
        b.content(DataContentType::Data)
            .file_path(file_path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1_000)
            .record_count(10)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .key_metadata(key_metadata);
        b.build().expect("data file builder")
    }

    fn ext_with_i64_range(index_name: &str, lo: i64, hi: i64) -> EmatixDataFileExtension {
        EmatixDataFileExtension {
            sidecar_relative_path: format!("{index_name}.idx"),
            summaries: vec![
                IndexSummary::new(index_name).with_range(SummaryKey::I64(lo), SummaryKey::I64(hi))
            ],
        }
    }

    #[test]
    fn round_trip_through_key_metadata() {
        let ext = ext_with_i64_range("idx_orderkey", 1, 6_000_000);
        let bytes = encode_key_metadata(&ext);
        assert_eq!(&bytes[..4], KEY_METADATA_MAGIC);
        let decoded = decode_key_metadata(&bytes).unwrap().unwrap();
        assert_eq!(decoded, ext);
    }

    #[test]
    fn decode_bytes_without_magic_returns_none() {
        // Genuine encryption metadata from a different producer —
        // never starts with EMTX. Decoder must return None, not Err.
        let other_payload = b"someone-elses-bytes";
        assert!(decode_key_metadata(other_payload).unwrap().is_none());
        // Empty bytes likewise.
        assert!(decode_key_metadata(&[]).unwrap().is_none());
        // Bytes shorter than magic length.
        assert!(decode_key_metadata(b"EM").unwrap().is_none());
        // Bytes that share a prefix with magic but aren't the magic.
        assert!(decode_key_metadata(b"EMTYsomething").unwrap().is_none());
    }

    #[test]
    fn decode_malformed_after_magic_errors() {
        let mut bytes = KEY_METADATA_MAGIC.to_vec();
        bytes.extend_from_slice(b"not even close to JSON");
        let err = decode_key_metadata(&bytes).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn decode_wrong_version_after_magic_errors() {
        let mut bytes = KEY_METADATA_MAGIC.to_vec();
        bytes.extend_from_slice(
            br#"{"version":"v99","sidecar_relative_path":"x.idx","summaries":[]}"#,
        );
        let err = decode_key_metadata(&bytes).unwrap_err();
        assert!(matches!(err, IcebergIndexError::UnsupportedVersion(_)));
    }

    #[test]
    fn decode_invalid_utf8_after_magic_errors() {
        let mut bytes = KEY_METADATA_MAGIC.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFE, 0xFD]);
        let err = decode_key_metadata(&bytes).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn extract_on_real_data_file() {
        let ext = ext_with_i64_range("idx_orderkey", 100, 200);
        let bytes = encode_key_metadata(&ext);
        let df = make_data_file("s3://bucket/path/0.parquet", Some(bytes));
        let extracted = extract_extension(&df).unwrap().unwrap();
        assert_eq!(extracted, ext);
    }

    #[test]
    fn extract_on_data_file_with_no_key_metadata() {
        let df = make_data_file("s3://bucket/path/0.parquet", None);
        assert!(extract_extension(&df).unwrap().is_none());
    }

    #[test]
    fn extract_on_data_file_with_foreign_key_metadata() {
        let df = make_data_file(
            "s3://bucket/path/0.parquet",
            Some(b"encryption_metadata_blob".to_vec()),
        );
        assert!(extract_extension(&df).unwrap().is_none());
    }

    #[test]
    fn attach_extension_round_trip_via_builder() {
        let ext = ext_with_i64_range("idx_orderkey", 1, 100);
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path("s3://bucket/path/0.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1_000)
            .record_count(10)
            .partition(Struct::empty())
            .partition_spec_id(0);
        let builder = attach_extension(builder, &ext);
        let df = builder.build().unwrap();
        let extracted = extract_extension(&df).unwrap().unwrap();
        assert_eq!(extracted, ext);
    }

    #[test]
    fn prune_eq_keeps_only_in_range_files() {
        // Three files with non-overlapping [min, max] for the same
        // index name.
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 0, 99))),
        );
        let f2 = make_data_file(
            "s3://b/f2.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 100, 199))),
        );
        let f3 = make_data_file(
            "s3://b/f3.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 200, 299))),
        );
        let files = vec![f1, f2, f3];

        // Query for 150 — only f2 survives.
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(150)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f2.parquet");

        // Query for 50 — only f1.
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(50)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f1.parquet");

        // Query for boundary 99 — only f1 (inclusive on high).
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(99)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f1.parquet");

        // Query for 100 — only f2 (inclusive on low).
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(100)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f2.parquet");

        // Query for 1000 — nothing.
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(1000)).unwrap();
        assert!(kept.is_empty());
    }

    #[test]
    fn prune_eq_keeps_files_with_no_ematix_extension() {
        // Mixed input: one file with our extension, one without. The
        // one without is always kept (we have no information).
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 0, 50))),
        );
        let f2 = make_data_file("s3://b/f2.parquet", None);
        let f3 = make_data_file("s3://b/f3.parquet", Some(b"someone_elses_bytes".to_vec()));
        let files = vec![f1, f2, f3];

        // 1000 is outside f1's [0, 50] → f1 drops; f2 and f3 keep.
        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(1000)).unwrap();
        let paths: Vec<&str> = kept.iter().map(|d| d.file_path()).collect();
        assert_eq!(paths, vec!["s3://b/f2.parquet", "s3://b/f3.parquet"]);
    }

    #[test]
    fn prune_eq_keeps_files_with_no_summary_for_this_index() {
        // File has our extension but lists summaries for a different
        // index name — we have no info on the queried one, keep.
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_y", 0, 50))),
        );
        let files = vec![f1];

        let kept = prune_data_files_eq(&files, "idx_x", &Key::I64(1000)).unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn prune_eq_errors_on_malformed_extension() {
        let mut bad = KEY_METADATA_MAGIC.to_vec();
        bad.extend_from_slice(b"not JSON");
        let f1 = make_data_file("s3://b/f1.parquet", Some(bad));
        let files = vec![f1];
        let err = prune_data_files_eq(&files, "idx_x", &Key::I64(0)).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn prune_range_keeps_overlapping_files() {
        // f1: [0, 99], f2: [100, 199], f3: [200, 299].
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 0, 99))),
        );
        let f2 = make_data_file(
            "s3://b/f2.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 100, 199))),
        );
        let f3 = make_data_file(
            "s3://b/f3.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 200, 299))),
        );
        let files = vec![f1, f2, f3];

        // Query [50, 150] → f1 + f2.
        let kept =
            prune_data_files_range(&files, "idx_x", Some(&Key::I64(50)), Some(&Key::I64(150)))
                .unwrap();
        let paths: Vec<&str> = kept.iter().map(|d| d.file_path()).collect();
        assert_eq!(paths, vec!["s3://b/f1.parquet", "s3://b/f2.parquet"]);

        // Query [-∞, 50] → only f1.
        let kept = prune_data_files_range(&files, "idx_x", None, Some(&Key::I64(50))).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f1.parquet");

        // Query [250, +∞) → only f3.
        let kept = prune_data_files_range(&files, "idx_x", Some(&Key::I64(250)), None).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].file_path(), "s3://b/f3.parquet");

        // Open-ended both sides → everything kept.
        let kept = prune_data_files_range(&files, "idx_x", None, None).unwrap();
        assert_eq!(kept.len(), 3);

        // No overlap on either side → empty.
        let kept = prune_data_files_range(
            &files,
            "idx_x",
            Some(&Key::I64(1_000)),
            Some(&Key::I64(2_000)),
        )
        .unwrap();
        assert!(kept.is_empty());
    }

    #[test]
    fn summary_for_returns_when_present() {
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 1, 99))),
        );
        let (summary, ext) = summary_for(&f1, "idx_x").unwrap().unwrap();
        assert_eq!(summary.name, "idx_x");
        assert_eq!(summary.min_key, Some(SummaryKey::I64(1)));
        assert_eq!(ext.sidecar_relative_path, "idx_x.idx");
    }

    #[test]
    fn summary_for_returns_none_when_no_extension() {
        let f1 = make_data_file("s3://b/f1.parquet", None);
        assert!(summary_for(&f1, "idx_x").unwrap().is_none());
    }

    #[test]
    fn summary_for_returns_none_when_no_summary_for_name() {
        let f1 = make_data_file(
            "s3://b/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_y", 1, 99))),
        );
        assert!(summary_for(&f1, "idx_x").unwrap().is_none());
    }

    // ============================================================
    // Π.22a: resolve_sidecar_uri + pair_with_extensions
    // ============================================================

    #[test]
    fn resolve_sidecar_s3_uri() {
        assert_eq!(
            resolve_sidecar_uri("s3://bucket/dir/file.parquet", "file.parquet.idx"),
            "s3://bucket/dir/file.parquet.idx"
        );
        assert_eq!(
            resolve_sidecar_uri("s3://bucket/dir/file.parquet", "sub/file.idx"),
            "s3://bucket/dir/sub/file.idx"
        );
    }

    #[test]
    fn resolve_sidecar_file_scheme_uri() {
        assert_eq!(
            resolve_sidecar_uri("file:///abs/dir/file.parquet", "file.parquet.idx"),
            "file:///abs/dir/file.parquet.idx"
        );
    }

    #[test]
    fn resolve_sidecar_bare_absolute_path() {
        assert_eq!(
            resolve_sidecar_uri("/abs/dir/file.parquet", "file.parquet.idx"),
            "/abs/dir/file.parquet.idx"
        );
    }

    #[test]
    fn resolve_sidecar_no_directory_in_data_path() {
        // Bare filename input: nothing to anchor against, the relative
        // path is returned as-is.
        assert_eq!(
            resolve_sidecar_uri("file.parquet", "file.parquet.idx"),
            "file.parquet.idx"
        );
    }

    #[test]
    fn resolve_sidecar_absolute_relative_overrides_directory() {
        // If the producer wrote an absolute path (leading slash) into
        // the extension, honor it — don't anchor against the data
        // file's directory.
        assert_eq!(
            resolve_sidecar_uri("s3://bucket/dir/file.parquet", "/other/path.idx"),
            "/other/path.idx"
        );
    }

    #[test]
    fn resolve_sidecar_full_uri_relative_overrides_directory() {
        // A relative path that itself contains a scheme is treated as
        // absolute — useful for cross-bucket setups.
        assert_eq!(
            resolve_sidecar_uri("s3://bucket/dir/file.parquet", "s3://other-bucket/path.idx"),
            "s3://other-bucket/path.idx"
        );
    }

    #[test]
    fn resolve_sidecar_root_directory() {
        // Data file directly under root, sidecar relative.
        assert_eq!(
            resolve_sidecar_uri("/file.parquet", "sidecar.idx"),
            "/sidecar.idx"
        );
    }

    #[test]
    fn pair_with_extensions_skips_files_without_extension() {
        // Mixed input: 2 files with our extension, 1 without, 1 with
        // foreign key_metadata. Only the 2 with extensions survive.
        let f1 = make_data_file(
            "s3://b/dir/f1.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 0, 99))),
        );
        let f2 = make_data_file("s3://b/dir/f2.parquet", None);
        let f3 = make_data_file(
            "s3://b/dir/f3.parquet",
            Some(encode_key_metadata(&ext_with_i64_range("idx_x", 100, 199))),
        );
        let f4 = make_data_file(
            "s3://b/dir/f4.parquet",
            Some(b"foreign_bytes_no_magic".to_vec()),
        );

        let candidates = pair_with_extensions(vec![f1, f2, f3, f4]).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].data_file.file_path(), "s3://b/dir/f1.parquet");
        assert_eq!(candidates[0].sidecar_uri, "s3://b/dir/idx_x.idx");
        assert_eq!(candidates[1].data_file.file_path(), "s3://b/dir/f3.parquet");
        assert_eq!(candidates[1].sidecar_uri, "s3://b/dir/idx_x.idx");
    }

    #[test]
    fn pair_with_extensions_errors_on_malformed() {
        // A magic-prefixed but malformed payload errors loudly rather
        // than silently dropping the file (which would mask a bug).
        let mut bad = KEY_METADATA_MAGIC.to_vec();
        bad.extend_from_slice(b"not even close to JSON");
        let f1 = make_data_file("s3://b/dir/f1.parquet", Some(bad));
        let err = pair_with_extensions(vec![f1]).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn pair_with_extensions_empty_input() {
        let candidates = pair_with_extensions(vec![]).unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn pair_with_extensions_resolves_sidecar_at_root() {
        // Data file at top of bucket — sidecar resolves to top of
        // bucket too.
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "sidecar.idx".into(),
            summaries: vec![],
        };
        let f1 = make_data_file("s3://bucket/file.parquet", Some(encode_key_metadata(&ext)));
        let candidates = pair_with_extensions(vec![f1]).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].sidecar_uri, "s3://bucket/sidecar.idx");
    }

    #[test]
    fn pair_with_extensions_respects_absolute_sidecar_path() {
        // Producer wrote an absolute sidecar URI in the extension —
        // pair honors it instead of anchoring to data_file's dir.
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "s3://other-bucket/sidecar.idx".into(),
            summaries: vec![],
        };
        let f1 = make_data_file(
            "s3://bucket/dir/file.parquet",
            Some(encode_key_metadata(&ext)),
        );
        let candidates = pair_with_extensions(vec![f1]).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].sidecar_uri, "s3://other-bucket/sidecar.idx");
    }

    // ============================================================
    // Π.22b: collect_data_files
    // ============================================================
    //
    // The populated-table case wants a full Iceberg fixture
    // (TableMetadata + written manifest_list + written manifest
    // files) — that lives in the Π.22c oracle, which also exercises
    // the end-to-end sidecar lookup. Here we only confirm the empty-
    // snapshot branch (a freshly created table with no commits).

    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::io::FileIOBuilder;
    use iceberg::spec::{
        FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
        TableMetadataBuilder, Type,
    };
    use iceberg::table::Table;
    use iceberg::TableIdent;

    /// Build a brand-new Iceberg `Table` over an in-memory FileIO,
    /// with no current snapshot. Used to drive the empty-snapshot
    /// branch of `collect_data_files`.
    fn build_empty_in_memory_table() -> Table {
        let file_io = FileIOBuilder::new("memory").build().unwrap();
        let schema = Schema::builder()
            .with_fields(vec![Arc::new(NestedField::required(
                1,
                "v",
                Type::Primitive(PrimitiveType::Long),
            ))])
            .build()
            .unwrap();
        let metadata_built = TableMetadataBuilder::new(
            schema,
            PartitionSpec::unpartition_spec(),
            SortOrder::unsorted_order(),
            "memory:///table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .unwrap()
        .build()
        .unwrap();
        Table::builder()
            .metadata(metadata_built.metadata)
            .identifier(TableIdent::from_strs(["db", "t"]).unwrap())
            .file_io(file_io)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn collect_returns_empty_for_table_without_snapshot() {
        let table = build_empty_in_memory_table();
        assert!(table.metadata().current_snapshot().is_none());
        let files = collect_data_files(&table).await.unwrap();
        assert!(files.is_empty());
    }
}
