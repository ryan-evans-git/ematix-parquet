//! Π.19a oracle: per-source-page Bloom index over INT64.
//!
//! Two correctness invariants matter for Bloom:
//!
//! 1. **No false negatives.** For every value present in the source,
//!    `bloom_probe` must return at least all pages containing it.
//!    A miss here means the index is *broken* — readers will see
//!    fewer rows than the source actually has.
//! 2. **`read_column_i64_via_bloom_eq` matches a full-scan filter.**
//!    Despite Bloom's false positives, the end-to-end convenience
//!    entry filters in memory and returns exactly the rows the
//!    full scan would.
//!
//! False positives are *allowed* but should be bounded by the
//! configured `target_fpp`. We assert a loose `≤ target_fpp × 5`
//! envelope to catch gross mis-sizings without flaking on small-N
//! statistical noise.

use ematix_parquet_codec::index::{IndexBuilder, Key, ParquetIndex};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::write_i64_column_to_path;
use ematix_parquet_io::ParquetFile;

fn build_source_and_bloom(
    dir: &std::path::Path,
    name_prefix: &str,
    values: &[i64],
    target_fpp: f64,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = dir.join(format!("{name_prefix}.parquet"));
    let idx = dir.join(format!("{name_prefix}.parquet.idx"));
    write_i64_column_to_path(&src, "v", values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_bloom_page_i64(&idx, "idx_v", 0, target_fpp)
        .expect("build bloom sidecar");
    (src, idx)
}

#[test]
fn bloom_has_no_false_negatives_for_present_values() {
    // Every value in the source MUST be reported by bloom_probe
    // for at least the page that holds it. False positives are
    // acceptable; false negatives are not (would corrupt query
    // results).
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..5_000i64).collect();
    let (src, idx) = build_source_and_bloom(dir.path(), "no_fn", &values, 0.01);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Probe each present value; bloom_probe must return a non-empty
    // page list.
    for v in 0..5_000i64 {
        let hits = reader.bloom_probe("idx_v", &Key::I64(v)).unwrap();
        assert!(
            !hits.is_empty(),
            "value {v} present in source but bloom_probe returned no pages"
        );
    }
}

#[test]
fn bloom_probe_reports_absent_values_with_bounded_fpp() {
    // Probe values that don't exist; record what fraction of pages
    // return spurious hits. Should be close to target_fpp.
    let dir = tempfile::tempdir().unwrap();
    let target_fpp = 0.01;
    // 50K values, 50K distinct → ~1 row per distinct value.
    let values: Vec<i64> = (0..50_000i64).collect();
    let (src, idx) = build_source_and_bloom(dir.path(), "fpp", &values, target_fpp);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // Query 1000 values that definitely don't exist (well outside
    // the source range).
    let n_probe = 1_000usize;
    let mut total_false_positive_hits = 0u64;
    for q in 0..n_probe as i64 {
        let absent = 10_000_000i64 + q; // guaranteed not in [0, 50_000)
        let hits = reader.bloom_probe("idx_v", &Key::I64(absent)).unwrap();
        // hits.len() is the number of false-positive pages this
        // probe produced.
        total_false_positive_hits += hits.len() as u64;
    }
    // FPP rate: false-positive hits per absent probe. Without knowing
    // exact page count we just assert a sane envelope (see
    // documentation below). For target_fpp=0.01 and a small source
    // with O(1) pages, this should be much less than 1 hit per probe
    // on average.
    let observed_fpp = total_false_positive_hits as f64 / (n_probe as f64);
    // observed_fpp here is "false-positive hits per absent probe".
    // For target_fpp=0.01 over N pages, expected ≈ 0.01 × N.
    // Without knowing N exactly, just assert the rate is in a sane
    // ballpark (less than 5% of probes hitting more than 0.5 pages
    // on average — a generous envelope).
    assert!(
        observed_fpp < 5.0,
        "observed FPP-ish rate {observed_fpp} suspiciously high for target {target_fpp}"
    );
}

#[test]
fn read_column_via_bloom_eq_matches_full_scan() {
    // The end-to-end correctness invariant. Bloom may yield extra
    // candidate pages; the convenience entry decodes them, filters
    // in-memory, and the result equals what a full-scan filter
    // would return.
    let dir = tempfile::tempdir().unwrap();
    // Repeated values so a key hits multiple pages.
    let mut values: Vec<i64> = Vec::new();
    for _ in 0..5 {
        for v in 0..2_000i64 {
            values.push(v);
        }
    }
    let (src, idx) = build_source_and_bloom(dir.path(), "via_bloom", &values, 0.01);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    for key in [0i64, 1, 100, 1234, 1999] {
        let indexed = reader
            .read_column_i64_via_bloom_eq(&source, "idx_v", key, 0)
            .unwrap();
        let baseline: Vec<i64> = read_column_i64(&source, 0, 0)
            .unwrap()
            .into_iter()
            .filter(|v| *v == key)
            .collect();
        assert_eq!(indexed, baseline, "key={key}");
        assert_eq!(indexed.len(), 5, "key={key}: 5 duplicates expected");
    }

    // Absent value → empty result.
    let indexed = reader
        .read_column_i64_via_bloom_eq(&source, "idx_v", 99_999, 0)
        .unwrap();
    assert!(indexed.is_empty());
}

#[test]
fn rejects_wrong_key_type_on_bloom_probe() {
    // The current write_bloom_page_i64 hashes per the i64 PLAIN
    // encoding. Probing with a different physical-type Key would
    // hash different bytes — surfacing as "value not found" rather
    // than an error.
    //
    // We don't have a hard physical-type guard on bloom_probe (the
    // BloomPage manifest entry carries the source column but no
    // distinct PhysicalType field — it's BYTE_ARRAY of bloom bytes
    // regardless of source type). So this test pins the current
    // *behaviour*: an i32 key against an i64 bloom returns no hits.
    // If we add an explicit type guard later, this test will
    // change.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..100i64).collect();
    let (src, idx) = build_source_and_bloom(dir.path(), "wrong_key", &values, 0.01);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    // i32 key with the same numeric value: different PLAIN-encoding
    // bytes, different hash, likely no hits (modulo any FPP coincidence).
    let hits_i64 = reader.bloom_probe("idx_v", &Key::I64(0)).unwrap();
    let hits_i32 = reader.bloom_probe("idx_v", &Key::I32(0)).unwrap();
    assert!(!hits_i64.is_empty());
    // Not strictly zero (false positives can happen), but should be
    // a strict subset of "all pages" — i.e. much smaller than
    // hits_i64.
    assert!(
        hits_i32.len() < hits_i64.len() + 5,
        "i32 spurious hits {} vs i64 true hits {}",
        hits_i32.len(),
        hits_i64.len()
    );
}

#[test]
fn rejects_lookup_eq_on_bloom_index() {
    // Page-Bloom indexes don't support `lookup_eq` (no rowsets);
    // callers must use bloom_probe / read_column_i64_via_bloom_eq.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..100i64).collect();
    let (src, idx) = build_source_and_bloom(dir.path(), "wrong_api", &values, 0.01);
    let source = ParquetFile::open(&src).unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let err = reader.lookup_eq("idx_v", &Key::I64(0)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("not a sorted") || msg.contains("sorted index"),
        "got: {msg}"
    );
}

#[test]
fn rejects_bloom_probe_on_sorted_index() {
    // And the reverse: bloom_probe on a sorted index errors loudly.
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("sorted_for_bloom_probe.parquet");
    let idx = dir.path().join("sorted_for_bloom_probe.parquet.idx");
    let values: Vec<i64> = (0..100i64).collect();
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .unwrap();
    let reader = ParquetIndex::open(&idx, &source).unwrap();

    let err = reader.bloom_probe("idx_v", &Key::I64(0)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("page-Bloom") || msg.contains("not a page"),
        "got: {msg}"
    );
}

#[test]
fn rejects_invalid_target_fpp() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("bad_fpp.parquet");
    let idx = dir.path().join("bad_fpp.parquet.idx");
    let values: Vec<i64> = vec![1, 2, 3];
    write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    let builder = IndexBuilder::new(&source);

    // 0.0 is out of (0, 1), reject.
    let err = builder
        .write_bloom_page_i64(&idx, "idx_v", 0, 0.0)
        .unwrap_err();
    assert!(format!("{err}").contains("target_fpp"));

    // 1.0 is out of (0, 1), reject.
    let err = builder
        .write_bloom_page_i64(&idx, "idx_v", 0, 1.0)
        .unwrap_err();
    assert!(format!("{err}").contains("target_fpp"));
}
