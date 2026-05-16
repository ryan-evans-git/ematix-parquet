//! Π.14e — adaptive façade integration tests.
//!
//! Round-trip via the codec's own dict writer + adaptive reader:
//!
//! 1. Low-selectivity predicate (~1%) → façade returns `Fused`
//!    bitmap output; popcount matches the static fused reference.
//! 2. High-selectivity predicate (~70%) → façade returns
//!    `Materialized` values output; values match the eager
//!    read + filter reference.
//! 3. Telemetry callback fires once per chunk with the right
//!    `SelectivityProbe`.
//! 4. PLAIN-encoded chunk (no dict) → façade returns
//!    `InvalidInput`.

use ematix_parquet_codec::adaptive::{
    AdaptiveDispatchOptions, AdaptiveOutputKind, Dispatch, SelectivityProbe,
};
use ematix_parquet_codec::read::{read_column_i32, read_column_i32_predicate_adaptive};
use ematix_parquet_codec::write::{write_i32_column_dict_to_path, write_i32_column_to_path};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use tempfile::NamedTempFile;

/// 64 K rows, dict_size 256, indices round-robin so every dict
/// entry is hit ~256 times. Predicate selectivity equals the
/// fraction of dict slots marked.
fn write_dict_fixture(path: &std::path::Path) -> Vec<i32> {
    let dict_size: i32 = 256;
    let rows: i32 = 65_536;
    let values: Vec<i32> = (0..rows).map(|i| i % dict_size).collect();
    write_i32_column_dict_to_path(path, "id", &values, CompressionCodec::Snappy).unwrap();
    values
}

#[test]
fn low_selectivity_returns_bitmap_fused() {
    let tmp = NamedTempFile::new().unwrap();
    let values = write_dict_fixture(tmp.path());

    let file = ParquetFile::open(tmp.path()).unwrap();
    let opts = AdaptiveDispatchOptions::default();
    // Predicate: v < 4  → ~1.5% of dict slots (4/256). Each slot
    // hit ~256 times → ~1.5% of rows.
    let out = read_column_i32_predicate_adaptive(&file, 0, 0, |v| *v < 4, opts, None).unwrap();

    assert_eq!(out.dispatch, Dispatch::Fused);
    assert_eq!(out.total_rows, values.len());
    let expected_passes = values.iter().filter(|v| **v < 4).count();
    match out.kind {
        AdaptiveOutputKind::Bitmap {
            bitmap: _,
            set_bits,
        } => {
            assert_eq!(set_bits, expected_passes);
        }
        AdaptiveOutputKind::Values(_) => panic!("expected Bitmap, got Values"),
    }
}

#[test]
fn high_selectivity_returns_values_materialized() {
    let tmp = NamedTempFile::new().unwrap();
    let values = write_dict_fixture(tmp.path());

    let file = ParquetFile::open(tmp.path()).unwrap();
    let opts = AdaptiveDispatchOptions::default();
    // Predicate: v < 192 → 75% of dict slots → 75% of rows.
    let out = read_column_i32_predicate_adaptive(&file, 0, 0, |v| *v < 192, opts, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Materialized);

    let expected: Vec<i32> = values.into_iter().filter(|v| *v < 192).collect();
    match out.kind {
        AdaptiveOutputKind::Values(got) => {
            assert_eq!(got.len(), expected.len());
            assert_eq!(got, expected);
        }
        AdaptiveOutputKind::Bitmap { .. } => panic!("expected Values, got Bitmap"),
    }
}

#[test]
fn telemetry_callback_fires_through_facade() {
    let tmp = NamedTempFile::new().unwrap();
    write_dict_fixture(tmp.path());

    let file = ParquetFile::open(tmp.path()).unwrap();
    let opts = AdaptiveDispatchOptions::default();

    let mut probes: Vec<SelectivityProbe> = Vec::new();
    {
        let mut cb = |p: SelectivityProbe| probes.push(p);
        let _out = read_column_i32_predicate_adaptive(&file, 0, 0, |v| *v < 4, opts, Some(&mut cb))
            .unwrap();
    }
    assert_eq!(probes.len(), 1, "façade fires telemetry exactly once");
    let p = probes[0];
    assert_eq!(p.dispatch, Dispatch::Fused);
    assert!(
        p.selectivity < 0.05,
        "low-selectivity predicate should show <5% (got {})",
        p.selectivity
    );
}

#[test]
fn plain_only_chunk_rejected() {
    // `write_i32_column_to_path` emits a PLAIN-encoded data page
    // (no DictionaryPage). The adaptive façade must error.
    let tmp = NamedTempFile::new().unwrap();
    let values: Vec<i32> = (0..100).collect();
    write_i32_column_to_path(tmp.path(), "id", &values).unwrap();

    // Sanity: the file *does* round-trip via the non-adaptive reader.
    let file = ParquetFile::open(tmp.path()).unwrap();
    let direct = read_column_i32(&file, 0, 0).unwrap();
    assert_eq!(direct, values);

    let opts = AdaptiveDispatchOptions::default();
    let r = read_column_i32_predicate_adaptive(&file, 0, 0, |v| *v < 50, opts, None);
    assert!(r.is_err(), "PLAIN-only chunk must reject adaptive façade");
}

#[test]
fn custom_options_threshold_overrides_default() {
    let tmp = NamedTempFile::new().unwrap();
    let _values = write_dict_fixture(tmp.path());
    let file = ParquetFile::open(tmp.path()).unwrap();

    // 30% selectivity. Default threshold (0.10) → Materialized.
    // Override threshold = 0.5 → Fused.
    let pred = |v: &i32| *v < 77; // 77/256 ≈ 30%
    let out_default = read_column_i32_predicate_adaptive(
        &file,
        0,
        0,
        pred,
        AdaptiveDispatchOptions::default(),
        None,
    )
    .unwrap();
    assert_eq!(out_default.dispatch, Dispatch::Materialized);

    let opts_high = AdaptiveDispatchOptions {
        threshold: 0.5,
        ..AdaptiveDispatchOptions::default()
    };
    let out_high = read_column_i32_predicate_adaptive(&file, 0, 0, pred, opts_high, None).unwrap();
    assert_eq!(out_high.dispatch, Dispatch::Fused);
}
