//! Π.14d — extended acceptance tests for the adaptive chunk runner.
//!
//! Π.14b's oracle covers the happy paths (low-sel → Fused, high-sel
//! → Materialized, telemetry, empty). This file adds the scenarios
//! that prove the API holds up across configs:
//!
//! 1. Width coverage — bw ∈ {14, 16, 18} (the NEON-fused widths
//!    beyond Q14's bw=12) round-trip on both Fused and Materialized.
//! 2. `probe_pages` edge cases — probe = 1 page, probe > total
//!    pages.
//! 3. Custom threshold override — caller-set `threshold = 0.5`
//!    delays the switch to Materialized.
//! 4. Mid-chunk selectivity shift — probe sees low-sel pages, the
//!    runner stays Fused based on the probe even though later
//!    pages are dense. (Verifies the runner doesn't peek past the
//!    probe budget when deciding.)
//! 5. Bit-for-bit oracle: adaptive Fused output matches static
//!    page-by-page fused across all widths covered.

use ematix_parquet_codec::adaptive::{
    run_adaptive_dict_chunk, AdaptiveDictPredicate, AdaptiveOutputKind, AdaptivePageInput, Dispatch,
};
use ematix_parquet_codec::dict::{
    build_dict_predicate_mask, decode_rle_dictionary_into, decode_rle_dictionary_predicate_bitmap,
};
use ematix_parquet_codec::rle::encode_rle_bit_packed_single_run;

fn build_body(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut body = vec![bit_width];
    body.extend(encode_rle_bit_packed_single_run(indices, bit_width));
    body
}

/// Build pages with each page using `indices_per_page` indices
/// drawn round-robin from `0..dict_size`. Returns owned bodies +
/// row counts.
fn synth_pages(
    pages_n: usize,
    rows_per_page: usize,
    dict_size: usize,
    bit_width: u8,
) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut bodies = Vec::with_capacity(pages_n);
    let mut rows = Vec::with_capacity(pages_n);
    for page_idx in 0..pages_n {
        let indices: Vec<u32> = (0..rows_per_page)
            .map(|i| ((page_idx * rows_per_page + i) % dict_size) as u32)
            .collect();
        bodies.push(build_body(&indices, bit_width));
        rows.push(rows_per_page);
    }
    (bodies, rows)
}

fn pages_input<'a>(bodies: &'a [Vec<u8>], rows: &[usize]) -> Vec<AdaptivePageInput<'a>> {
    bodies
        .iter()
        .zip(rows.iter())
        .map(|(b, n)| AdaptivePageInput {
            body: b.as_slice(),
            num_values: *n,
        })
        .collect()
}

fn reference_bitmap(bodies: &[Vec<u8>], rows: &[usize], dict_mask: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (body, n) in bodies.iter().zip(rows.iter()) {
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut out).unwrap();
    }
    out
}

#[test]
fn width_14_low_selectivity_matches_static_fused() {
    let dict_size = 4096;
    let (bodies, rows) = synth_pages(4, 8192, dict_size, 14);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    // Pass ~1% of dict entries.
    let dict_mask =
        build_dict_predicate_mask(&dict, 14, |v| *v < (dict_size as i32 / 100)).unwrap();
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Fused);
    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, .. } => {
            assert_eq!(bitmap, reference_bitmap(&bodies, &rows, &dict_mask));
        }
        _ => panic!("expected Bitmap"),
    }
}

#[test]
fn width_16_high_selectivity_returns_values_matching_static() {
    let dict_size = 16384;
    let (bodies, rows) = synth_pages(3, 8192, dict_size, 16);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    // Pass ~70% of dict entries.
    let pass = (dict_size as f32 * 0.70) as i32;
    let dict_mask = build_dict_predicate_mask(&dict, 16, |v| *v < pass).unwrap();
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Materialized);

    // Static-materialised reference.
    let mut want: Vec<i32> = Vec::new();
    for (body, n) in bodies.iter().zip(rows.iter()) {
        let mut tmp = Vec::new();
        decode_rle_dictionary_into(body, &dict, *n, &mut tmp).unwrap();
        let mut bm = Vec::new();
        decode_rle_dictionary_predicate_bitmap(body, *n, &dict_mask, &mut bm).unwrap();
        for row in 0..*n {
            if (bm[row / 8] >> (row % 8)) & 1 != 0 {
                want.push(tmp[row]);
            }
        }
    }
    match out.kind {
        AdaptiveOutputKind::Values(got) => assert_eq!(got, want),
        _ => panic!("expected Values"),
    }
}

#[test]
fn width_18_round_trip_low_selectivity() {
    // bw=18 → dict_mask of 2^18 = 256K entries. We use a striped
    // predicate (every 100th dict entry passes) so every page sees
    // uniform ~1% selectivity, regardless of how the round-robin
    // indices fall.
    let dict_size = 200_000;
    let (bodies, rows) = synth_pages(2, 8192, dict_size, 18);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    let dict_mask = build_dict_predicate_mask(&dict, 18, |v| *v % 100 == 0).unwrap();
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Fused);
    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, .. } => {
            assert_eq!(bitmap, reference_bitmap(&bodies, &rows, &dict_mask));
        }
        _ => panic!("expected Bitmap"),
    }
}

#[test]
fn probe_pages_one_still_dispatches() {
    // probe_pages = 1: dispatch decision driven by a single page.
    let dict_size = 1024;
    let (bodies, rows) = synth_pages(4, 8192, dict_size, 12);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    let dict_mask = build_dict_predicate_mask(&dict, 12, |v| *v < 8).unwrap();
    let mut cfg = AdaptiveDictPredicate::new(dict_mask);
    cfg.probe_pages = 1;

    let pages = pages_input(&bodies, &rows);
    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    // ~0.8% selectivity → Fused.
    assert_eq!(out.dispatch, Dispatch::Fused);
}

#[test]
fn probe_pages_exceeds_chunk_size() {
    // probe_pages > total pages: probe consumes the whole chunk,
    // dispatch decision is on the aggregated whole-chunk selectivity.
    let dict_size = 1024;
    let (bodies, rows) = synth_pages(2, 8192, dict_size, 12);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    let dict_mask = build_dict_predicate_mask(&dict, 12, |v| *v < 8).unwrap();
    let mut cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    cfg.probe_pages = 10; // chunk only has 2 pages.

    let pages = pages_input(&bodies, &rows);
    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    // Should still complete with a valid output.
    assert_eq!(out.dispatch, Dispatch::Fused);
    assert_eq!(out.total_rows, 2 * 8192);
    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, .. } => {
            assert_eq!(bitmap, reference_bitmap(&bodies, &rows, &dict_mask));
        }
        _ => panic!("expected Bitmap"),
    }
}

#[test]
fn custom_threshold_override_delays_materialized_switch() {
    // At ~30% selectivity with default threshold (0.10) → Materialized.
    // With caller-set threshold 0.5 → still Fused.
    let dict_size = 1024;
    let (bodies, rows) = synth_pages(4, 8192, dict_size, 12);
    let dict: Vec<i32> = (0..dict_size as i32).collect();
    let dict_mask =
        build_dict_predicate_mask(&dict, 12, |v| *v < (dict_size as i32 * 30 / 100)).unwrap();

    // Default cfg: should pick Materialized.
    let cfg_default = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);
    let out_default = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg_default, None).unwrap();
    assert_eq!(out_default.dispatch, Dispatch::Materialized);

    // High threshold: should pick Fused.
    let mut cfg_high = AdaptiveDictPredicate::new(dict_mask);
    cfg_high.threshold = 0.5;
    let out_high = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg_high, None).unwrap();
    assert_eq!(out_high.dispatch, Dispatch::Fused);
}

#[test]
fn dispatch_is_decided_by_probe_only_not_later_pages() {
    // Probe pages see low selectivity (page 0–2 are all out-of-range
    // for the predicate). Later pages are dense matches. Runner must
    // still commit to Fused based on probe — and emit a bitmap whose
    // popcount reflects every page, not just the probed ones.
    let dict_size: i32 = 1024;
    let dict: Vec<i32> = (0..dict_size).collect();
    let bit_width = 12;

    // Pages 0–2: all indices ≥ 100 (predicate filters < 100, so 0% pass).
    // Pages 3–4: all indices ∈ [0, 50] (~all pass).
    let mut bodies: Vec<Vec<u8>> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    for _ in 0..3 {
        let indices: Vec<u32> = (0..8192).map(|i| 100 + (i as u32 % 800)).collect();
        bodies.push(build_body(&indices, bit_width));
        rows.push(8192);
    }
    for _ in 0..2 {
        let indices: Vec<u32> = (0..8192).map(|i| i as u32 % 50).collect();
        bodies.push(build_body(&indices, bit_width));
        rows.push(8192);
    }

    let dict_mask = build_dict_predicate_mask(&dict, bit_width, |v| *v < 100).unwrap();
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone()); // probe_pages = 3
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(
        out.dispatch,
        Dispatch::Fused,
        "probe saw 0% selectivity over pages 0–2 → must commit Fused even though later pages are dense"
    );

    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, set_bits } => {
            // Pages 0–2: 0 matches. Pages 3–4: every row matches → 8192 × 2.
            assert_eq!(
                set_bits,
                2 * 8192,
                "later dense pages must still contribute to the bitmap"
            );
            assert_eq!(bitmap, reference_bitmap(&bodies, &rows, &dict_mask));
        }
        _ => panic!("expected Bitmap (dispatch was Fused)"),
    }
}
