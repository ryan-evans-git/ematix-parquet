//! Π.20 oracle: inverted (text) index over BYTE_ARRAY.
//!
//! Correctness invariant: **`read_column_byte_array_where_token`
//! returns exactly the rows whose tokenized form contains the
//! tokenized query token**, modulo source-row order. Anything that
//! breaks this property breaks the text-search use case.

use ematix_parquet_codec::index::{IndexBuilder, ParquetIndex, Tokenizer};
use ematix_parquet_codec::read::read_column_byte_array;
use ematix_parquet_codec::write::write_byte_array_column_to_path;
use ematix_parquet_io::ParquetFile;

/// Full-scan baseline: rows whose tokenized form contains
/// `normalized_token`. Uses the same tokenizer as the index.
fn baseline_contains(
    source: &ParquetFile,
    col: usize,
    tokenizer: Tokenizer,
    normalized_token: &[u8],
) -> Vec<Vec<u8>> {
    let values = read_column_byte_array(source, 0, col).unwrap();
    values
        .into_iter()
        .filter(|v| tokenizer.tokenize(v).iter().any(|t| t == normalized_token))
        .collect()
}

fn build_source_and_inverted(
    dir: &std::path::Path,
    name_prefix: &str,
    values: &[&[u8]],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = dir.join(format!("{name_prefix}.parquet"));
    let idx = dir.join(format!("{name_prefix}.parquet.idx"));
    write_byte_array_column_to_path(&src, "v", values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_inverted_byte_array(&idx, "idx_v", 0, Tokenizer::WhitespaceLowercaseV1)
        .expect("build inverted sidecar");
    (src, idx)
}

#[test]
fn single_token_query_matches_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![
        b"the quick brown fox",
        b"jumps over the lazy dog",
        b"the rain in Spain",
        b"never bring a fox to a dog fight",
        b"all hands on deck",
        b"hello world",
    ];
    let (src, idx) = build_source_and_inverted(dir.path(), "single", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // "fox" — appears in rows 0 and 3.
    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"fox", 0)
        .unwrap();
    let baseline = baseline_contains(&source, 0, Tokenizer::WhitespaceLowercaseV1, b"fox");
    assert_eq!(hits, baseline);
    assert_eq!(hits.len(), 2);

    // "the" — appears in rows 0, 1, 2.
    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"the", 0)
        .unwrap();
    let baseline = baseline_contains(&source, 0, Tokenizer::WhitespaceLowercaseV1, b"the");
    assert_eq!(hits, baseline);
    assert_eq!(hits.len(), 3);

    // "hello" — only row 5.
    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"hello", 0)
        .unwrap();
    assert_eq!(hits.len(), 1);

    // "missing" — empty.
    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"missing", 0)
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn query_is_normalized_via_tokenizer() {
    // The convenience entry applies the tokenizer to the query, so
    // user input with mixed case + leading/trailing whitespace
    // still matches.
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"Hello World", b"hello there", b"goodbye"];
    let (src, idx) = build_source_and_inverted(dir.path(), "norm", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // "  HELLO  " query → tokenizer produces ["hello"]; matches rows 0 + 1.
    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"  HELLO  ", 0)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0], b"Hello World".to_vec());
    assert_eq!(hits[1], b"hello there".to_vec());
}

#[test]
fn duplicate_token_in_same_row_counted_once() {
    // "fox fox fox" should set the row's bit in the `fox` bucket
    // exactly once — not three times — so the row still appears
    // exactly once in `read_column_byte_array_where_token`'s result.
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"fox fox fox", b"plain fox", b"no animal"];
    let (src, idx) = build_source_and_inverted(dir.path(), "dup", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let hits = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"fox", 0)
        .unwrap();
    // Both rows containing "fox" appear once each; row 0 is NOT
    // duplicated despite three occurrences.
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0], b"fox fox fox".to_vec());
    assert_eq!(hits[1], b"plain fox".to_vec());
}

#[test]
fn empty_query_errors() {
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"hello world"];
    let (src, idx) = build_source_and_inverted(dir.path(), "empty_q", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Empty + whitespace-only queries produce zero tokens.
    for q in [b"" as &[u8], b"   ", b"\t\n"] {
        let err = reader
            .read_column_byte_array_where_token(&source, "idx_v", q, 0)
            .unwrap_err();
        assert!(format!("{err}").contains("expected 1"), "got: {err}");
    }
}

#[test]
fn multi_token_query_errors() {
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"hello world"];
    let (src, idx) = build_source_and_inverted(dir.path(), "multi_q", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // "hello world" → 2 tokens; convenience entry rejects.
    let err = reader
        .read_column_byte_array_where_token(&source, "idx_v", b"hello world", 0)
        .unwrap_err();
    assert!(format!("{err}").contains("expected 1"));
}

#[test]
fn low_level_lookup_token_requires_pre_normalized_token() {
    // `lookup_token` does NOT apply the tokenizer. Pass a token
    // that's already in the tokenizer's normalized form for hits;
    // pass un-normalized and you get nothing.
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"Hello World"];
    let (src, idx) = build_source_and_inverted(dir.path(), "raw_lookup", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Pre-normalized "hello" → hits.
    let hits = reader.lookup_token("idx_v", b"hello").unwrap();
    assert_eq!(hits.len(), 1);

    // Un-normalized "Hello" → empty (the index stores lowercased forms).
    let hits = reader.lookup_token("idx_v", b"Hello").unwrap();
    assert!(hits.is_empty());
}

#[test]
fn builder_rejects_non_byte_array_column() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("wrong_type.parquet");
    let idx = dir.path().join("wrong_type.parquet.idx");
    let values: Vec<i64> = (0..10i64).collect();
    ematix_parquet_codec::write::write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    let err = IndexBuilder::new(&source)
        .write_inverted_byte_array(&idx, "idx_v", 0, Tokenizer::WhitespaceLowercaseV1)
        .unwrap_err();
    assert!(format!("{err}").contains("BYTE_ARRAY"));
}

#[test]
fn rejects_cross_kind_calls() {
    // lookup_token on a sorted index → reject.
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("sorted_for_token.parquet");
    let idx = dir.path().join("sorted_for_token.parquet.idx");
    let values: Vec<i64> = (0..10i64).collect();
    ematix_parquet_codec::write::write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let err = reader.lookup_token("idx_v", b"hello").unwrap_err();
    assert!(format!("{err}").contains("inverted index"));
}

#[test]
fn manifest_records_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let raw: Vec<&[u8]> = vec![b"a b c"];
    let (src, idx) = build_source_and_inverted(dir.path(), "manifest", &raw);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let m = reader.manifest();
    assert_eq!(m.indexes.len(), 1);
    match &m.indexes[0].kind {
        ematix_parquet_codec::index::IndexKind::Inverted { tokenizer, .. } => {
            assert_eq!(*tokenizer, Tokenizer::WhitespaceLowercaseV1);
        }
        other => panic!("expected Inverted, got {other:?}"),
    }
}
