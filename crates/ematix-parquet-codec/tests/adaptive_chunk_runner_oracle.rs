//! Π.14b oracle: the adaptive chunk runner picks the right path
//! and emits output that matches the static-path reference.
//!
//! Three scenarios exercised, each by synthesising a multi-page
//! chunk against a `dict_mask` tuned to a target selectivity:
//!
//! 1. **Low selectivity (~1%)** — Q14-shape. Runner must commit
//!    `Fused` and emit a bitmap that's byte-identical to the
//!    output of running `decode_rle_dictionary_predicate_bitmap`
//!    page-by-page.
//! 2. **High selectivity (~70%)** — runner must commit
//!    `Materialized` and emit values that match
//!    `decode_rle_dictionary_into` + filter through the same
//!    `dict_mask`.
//! 3. **Telemetry callback** fires once per chunk with the right
//!    counts + dispatch decision.

use ematix_parquet_codec::adaptive::{
    run_adaptive_dict_chunk, AdaptiveDictPredicate, AdaptiveOutputKind, AdaptivePageInput,
    Dispatch, SelectivityProbe,
};
use ematix_parquet_codec::dict::{
    build_dict_predicate_mask, decode_rle_dictionary_into, decode_rle_dictionary_predicate_bitmap,
};
use ematix_parquet_codec::rle::encode_rle_bit_packed_single_run;

/// Build one data-page body: `bit_width` byte + RLE/bit-packed indices.
fn build_body(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut body = vec![bit_width];
    body.extend(encode_rle_bit_packed_single_run(indices, bit_width));
    body
}

/// Build a (dict, dict_mask, pages) fixture for a target selectivity.
///
/// - dict is `dict_size` i32s 0..dict_size.
/// - `pages_n` pages of `rows_per_page` rows each.
/// - dict_mask is built so `target_select * dict_size` entries pass.
fn fixture(
    pages_n: usize,
    rows_per_page: usize,
    dict_size: usize,
    bit_width: u8,
    target_select: f32,
) -> (Vec<i32>, Vec<u8>, Vec<Vec<u8>>, Vec<usize>) {
    // dict = 0..dict_size.
    let dict: Vec<i32> = (0..dict_size as i32).collect();

    // Predicate: keep dict entries < threshold. Adjust threshold so
    // approx target_select * dict_size dict entries pass.
    let pass_count = ((dict_size as f32) * target_select).round() as i32;
    let pred = |v: &i32| *v < pass_count;
    let dict_mask = build_dict_predicate_mask(&dict, bit_width, pred).unwrap();

    // Round-robin indices give uniform selectivity across pages.
    let mut bodies = Vec::with_capacity(pages_n);
    let mut rows = Vec::with_capacity(pages_n);
    for page_idx in 0..pages_n {
        let indices: Vec<u32> = (0..rows_per_page)
            .map(|i| ((page_idx * rows_per_page + i) % dict_size) as u32)
            .collect();
        bodies.push(build_body(&indices, bit_width));
        rows.push(rows_per_page);
    }
    (dict, dict_mask, bodies, rows)
}

/// Reference bitmap: run the static fused kernel page-by-page and
/// concat the per-page bitmaps. Because all our pages are
/// rows_per_page-aligned (and rows_per_page divides 8), the
/// per-page bitmaps simply append.
fn reference_bitmap_concat(bodies: &[Vec<u8>], rows: &[usize], dict_mask: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (body, n) in bodies.iter().zip(rows.iter()) {
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut out).unwrap();
    }
    out
}

/// Reference values: decode each page's indices → values, then
/// emit each row where `dict_mask[idx]` says it passes.
fn reference_values(
    bodies: &[Vec<u8>],
    rows: &[usize],
    dict: &[i32],
    dict_mask: &[u8],
) -> Vec<i32> {
    let mut out = Vec::new();
    for (body, n) in bodies.iter().zip(rows.iter()) {
        let mut page_values: Vec<i32> = Vec::new();
        decode_rle_dictionary_into(body, dict, *n, &mut page_values).unwrap();
        let bw = body[0];
        let mut bm: Vec<u8> = Vec::new();
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut bm).unwrap();
        for row in 0..*n {
            let bit = (bm[row / 8] >> (row % 8)) & 1;
            if bit != 0 {
                out.push(page_values[row]);
            }
        }
        // bw read only to silence the warning in case body is empty.
        let _ = bw;
    }
    out
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

#[test]
fn low_selectivity_commits_fused_bitmap_matches_static() {
    // 5 pages × 8192 rows, bw=12, dict_size=2048, ~1% selectivity.
    let (dict, dict_mask, bodies, rows) = fixture(5, 8192, 2048, 12, 0.01);
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Fused);
    assert_eq!(out.total_rows, 5 * 8192);

    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, set_bits } => {
            let want = reference_bitmap_concat(&bodies, &rows, &dict_mask);
            assert_eq!(bitmap, want, "fused bitmap must match static reference");
            // Selectivity ≈ 1%; set bits should land near 5*8192*0.01 ≈ 410.
            // Allow ±5% slack for round-robin index quirks.
            let expected = (5.0 * 8192.0 * 0.01) as usize;
            assert!(
                (set_bits as isize - expected as isize).abs() < (expected as isize) / 5,
                "set_bits={set_bits} far from expected≈{expected}"
            );
        }
        AdaptiveOutputKind::Values(_) => panic!("expected Bitmap, got Values"),
    }
}

#[test]
fn high_selectivity_commits_materialized_values_match_static() {
    // 4 pages × 8192 rows, bw=12, dict_size=1024, ~70% selectivity.
    let (dict, dict_mask, bodies, rows) = fixture(4, 8192, 1024, 12, 0.70);
    let cfg = AdaptiveDictPredicate::new(dict_mask.clone());
    let pages = pages_input(&bodies, &rows);

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.dispatch, Dispatch::Materialized);
    assert_eq!(out.total_rows, 4 * 8192);

    match out.kind {
        AdaptiveOutputKind::Values(values) => {
            let want = reference_values(&bodies, &rows, &dict, &dict_mask);
            assert_eq!(
                values.len(),
                want.len(),
                "materialised values count must match static reference"
            );
            assert_eq!(
                values, want,
                "materialised values must match static reference"
            );
        }
        AdaptiveOutputKind::Bitmap { .. } => panic!("expected Values, got Bitmap"),
    }
}

#[test]
fn telemetry_callback_fires_with_correct_counts() {
    let (dict, dict_mask, bodies, rows) = fixture(5, 8192, 2048, 12, 0.01);
    let cfg = AdaptiveDictPredicate::new(dict_mask);
    let pages = pages_input(&bodies, &rows);

    let mut probes: Vec<SelectivityProbe> = Vec::new();
    {
        let mut cb = |p: SelectivityProbe| probes.push(p);
        let _out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, Some(&mut cb)).unwrap();
    }
    assert_eq!(
        probes.len(),
        1,
        "telemetry must fire exactly once per chunk"
    );
    let p = probes[0];
    assert_eq!(p.pages_probed, cfg.probe_pages);
    assert_eq!(p.rows_in, cfg.probe_pages * 8192);
    assert_eq!(p.dispatch, Dispatch::Fused);
    assert!(
        p.selectivity < 0.05,
        "probed selectivity should reflect the ~1% fixture (got {})",
        p.selectivity
    );
}

#[test]
fn empty_chunk_returns_empty_bitmap() {
    let dict: Vec<i32> = (0..16).collect();
    let dict_mask = build_dict_predicate_mask(&dict, 4, |v| *v < 8).unwrap();
    let cfg = AdaptiveDictPredicate::new(dict_mask);
    let pages: Vec<AdaptivePageInput<'_>> = Vec::new();

    let out = run_adaptive_dict_chunk::<i32>(&pages, &dict, &cfg, None).unwrap();
    assert_eq!(out.total_rows, 0);
    match out.kind {
        AdaptiveOutputKind::Bitmap { bitmap, set_bits } => {
            assert!(bitmap.is_empty());
            assert_eq!(set_bits, 0);
        }
        _ => panic!("empty chunk should emit empty bitmap"),
    }
}
