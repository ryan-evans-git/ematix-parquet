//! BYTE_ARRAY adaptive façade oracle.
//!
//! Mirrors `adaptive_facade_oracle.rs` (i32 path) for the
//! byte_array adaptive entry point added in v0.9.1:
//!
//! 1. Low-selectivity predicate → Fused bitmap, popcount matches
//!    the materialise + filter reference.
//! 2. High-selectivity predicate → Materialized values, the
//!    reconstructed `(bytes, offsets)` matches the eager read +
//!    filter reference.
//! 3. PLAIN-only chunk → InvalidInput.
//! 4. Telemetry callback fires once through the façade.

use ematix_parquet_codec::adaptive::{AdaptiveDispatchOptions, Dispatch, SelectivityProbe};
use ematix_parquet_codec::read::{
    read_column_byte_array, read_column_byte_array_predicate_adaptive, AdaptiveByteArrayOutputKind,
};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_table_to_path, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use tempfile::NamedTempFile;

/// Build a low-cardinality byte_array fixture: 8 distinct dict
/// values, 4096 rows round-robin (so each dict entry hits ~512
/// times). Returns (file, owned rows).
fn write_dict_fixture(path: &std::path::Path) -> Vec<Vec<u8>> {
    let dict_values: [&[u8]; 8] = [
        b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf", b"hotel",
    ];
    let rows: Vec<Vec<u8>> = (0..4096).map(|i| dict_values[i % 8].to_vec()).collect();
    let row_refs: Vec<&[u8]> = rows.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_dict_to_path(path, "tag", &row_refs, CompressionCodec::Snappy).unwrap();
    rows
}

#[test]
fn low_selectivity_returns_bitmap_fused() {
    let tmp = NamedTempFile::new().unwrap();
    let rows = write_dict_fixture(tmp.path());
    let file = ParquetFile::open(tmp.path()).unwrap();

    // 1 of 8 dict entries passes → ~12.5%... that's above default
    // threshold 0.10. To stay on Fused we need < 10%. Predicate
    // matches only one specific value out of 8 = 12.5%. Use a
    // narrower predicate: empty match (0%).
    let opts = AdaptiveDispatchOptions::default();
    let out = read_column_byte_array_predicate_adaptive(
        &file,
        0,
        0,
        |v| v == b"NEVER_MATCHES",
        opts,
        None,
    )
    .unwrap();

    assert_eq!(out.dispatch, Dispatch::Fused);
    assert_eq!(out.total_rows, rows.len());
    match out.kind {
        AdaptiveByteArrayOutputKind::Bitmap {
            bitmap: _,
            set_bits,
        } => {
            assert_eq!(set_bits, 0);
        }
        AdaptiveByteArrayOutputKind::Values { .. } => panic!("expected Bitmap"),
    }
}

#[test]
fn high_selectivity_returns_values_materialized() {
    let tmp = NamedTempFile::new().unwrap();
    let rows = write_dict_fixture(tmp.path());
    let file = ParquetFile::open(tmp.path()).unwrap();

    // 5 of 8 entries pass → ~62.5%. Well above the 0.10 threshold.
    let pass = |v: &[u8]| matches!(v, b"alpha" | b"bravo" | b"charlie" | b"delta" | b"echo");
    let opts = AdaptiveDispatchOptions::default();
    let out = read_column_byte_array_predicate_adaptive(&file, 0, 0, pass, opts, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Materialized);

    let expected: Vec<&[u8]> = rows
        .iter()
        .map(|v| v.as_slice())
        .filter(|v| pass(v))
        .collect();

    match out.kind {
        AdaptiveByteArrayOutputKind::Values { bytes, offsets } => {
            assert_eq!(
                offsets.len(),
                expected.len() + 1,
                "offsets must be set_bits + 1"
            );
            for (i, exp) in expected.iter().enumerate() {
                let lo = offsets[i] as usize;
                let hi = offsets[i + 1] as usize;
                assert_eq!(&bytes[lo..hi], *exp, "row {i}");
            }
        }
        AdaptiveByteArrayOutputKind::Bitmap { .. } => panic!("expected Values"),
    }
}

#[test]
fn telemetry_fires_once_through_facade() {
    let tmp = NamedTempFile::new().unwrap();
    write_dict_fixture(tmp.path());
    let file = ParquetFile::open(tmp.path()).unwrap();

    let opts = AdaptiveDispatchOptions::default();
    let mut probes: Vec<SelectivityProbe> = Vec::new();
    {
        let mut cb = |p: SelectivityProbe| probes.push(p);
        let _ = read_column_byte_array_predicate_adaptive(
            &file,
            0,
            0,
            |v| v == b"alpha",
            opts,
            Some(&mut cb),
        )
        .unwrap();
    }
    assert_eq!(probes.len(), 1);
    let p = probes[0];
    // 1/8 dict entries → 12.5%; default threshold 0.10 → Materialized.
    assert_eq!(p.dispatch, Dispatch::Materialized);
    assert!(
        (p.selectivity - 0.125).abs() < 0.02,
        "expected ~12.5% selectivity, got {}",
        p.selectivity
    );
}

#[test]
fn plain_only_chunk_rejected() {
    let rows: Vec<&[u8]> = (0..32).map(|_| b"x".as_slice()).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_table_to_path(
        tmp.path(),
        &[("v", ColumnData::ByteArray(&rows))],
        CompressionCodec::Uncompressed,
    )
    .unwrap();
    let file = ParquetFile::open(tmp.path()).unwrap();

    // Sanity: PLAIN read works.
    let plain = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(plain.len(), 32);

    let opts = AdaptiveDispatchOptions::default();
    let r = read_column_byte_array_predicate_adaptive(&file, 0, 0, |_| true, opts, None);
    assert!(
        r.is_err(),
        "PLAIN-only chunk must reject byte_array adaptive façade"
    );
}

#[test]
fn custom_threshold_keeps_low_sel_on_fused() {
    // 1/8 dict entries pass → 12.5%. Default threshold (0.10) → Materialized.
    // Override threshold 0.5 → Fused.
    let tmp = NamedTempFile::new().unwrap();
    write_dict_fixture(tmp.path());
    let file = ParquetFile::open(tmp.path()).unwrap();

    let pred = |v: &[u8]| v == b"alpha";
    let out_default = read_column_byte_array_predicate_adaptive(
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
    let out_high =
        read_column_byte_array_predicate_adaptive(&file, 0, 0, pred, opts_high, None).unwrap();
    assert_eq!(out_high.dispatch, Dispatch::Fused);
}
