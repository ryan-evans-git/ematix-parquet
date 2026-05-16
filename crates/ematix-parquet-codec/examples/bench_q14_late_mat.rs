//! Π.10c — Q14 façade-level late-materialization bench.
//!
//! Compares two end-to-end strategies for the Q14-shape filter
//! (`l_shipdate ∈ [1995-09-01, 1995-10-01)` → project partkey,
//! extendedprice, discount):
//!
//! 1. **Baseline (full decode → filter)**: decode all 4 columns in
//!    full via `read_column_*_into`, then apply the predicate and
//!    materialise the matching subset.
//! 2. **Late-mat (façade)**: decode shipdate, build a packed mask
//!    via `build_packed_mask`, then decode the 3 other columns via
//!    `read_column_*_masked_into` — only the ~1.4% of matching rows
//!    are materialised.
//!
//! The 2 ms gap to Polars from the upstream Q14 analysis is the
//! target. Acceptance: late-mat total ≤ baseline total / 2 (rough
//! ~50% speedup on the multi-column decode, since 3 of 4 columns
//! are skipping ~98.6% of values).
//!
//! Usage:
//!   cargo run --release --example bench_q14_late_mat
//!   TPCH_DATA_DIR=/path/to/sf1 cargo run --release --example bench_q14_late_mat
//!
//! Without TPC-H data, the bench prints a setup hint and exits 0.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ematix_parquet_codec::read::{
    build_packed_mask, read_column_f64_into, read_column_f64_masked_into, read_column_i32_into,
    read_column_i64_into, read_column_i64_masked_into,
};
use ematix_parquet_io::ParquetFile;

// Q14 shipdate window — days since 1970-01-01 epoch.
//   1995-09-01 = 9374, 1995-10-01 = 9404. 30 days × 7-year range
//   → ~1.4% selectivity on uniformly-distributed shipdate.
const LO: i32 = 9374;
const HI: i32 = 9404;

// TPC-H lineitem column indices.
const COL_PARTKEY: usize = 1; // INT64
const COL_EXTENDEDPRICE: usize = 5; // DOUBLE
const COL_DISCOUNT: usize = 6; // DOUBLE
const COL_SHIPDATE: usize = 10; // INT32

const WARMUPS: usize = 3;
const ITERS: usize = 12;

fn data_dir() -> Option<PathBuf> {
    if let Ok(s) = std::env::var("TPCH_DATA_DIR") {
        let p = PathBuf::from(s);
        if p.exists() {
            return Some(p);
        }
    }
    // Convention: peer ematix-flow checkout has TPC-H SF=1 staged.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .parent()?
        .parent()?
        .join("ematix-flow/examples/tpch/data/sf1");
    p.exists().then_some(p)
}

/// Run `f` `WARMUPS` times to warm caches, then `ITERS` times
/// measured. Returns (median, min, max).
fn bench<R>(mut f: impl FnMut() -> R) -> (Duration, Duration, Duration) {
    for _ in 0..WARMUPS {
        let _ = black_box(f());
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ = black_box(f());
        samples.push(t0.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    (median, min, max)
}

fn fmt(d: Duration) -> String {
    format!("{:>7.3} ms", d.as_secs_f64() * 1e3)
}

// ============================================================
// Strategy 1: baseline — full decode then filter
// ============================================================

/// Baseline: decode all 4 columns in full into reusable Vec buffers,
/// then walk rows, evaluate predicate, and materialise the matching
/// (partkey, extprice, discount) triple per row. The Vecs are
/// caller-owned so allocation cost amortises across iterations.
#[allow(clippy::too_many_arguments)]
fn baseline_full_decode(
    file: &ParquetFile,
    rg: usize,
    shipdate_buf: &mut Vec<i32>,
    partkey_buf: &mut Vec<i64>,
    extprice_buf: &mut Vec<f64>,
    discount_buf: &mut Vec<f64>,
    out_partkey: &mut Vec<i64>,
    out_extprice: &mut Vec<f64>,
    out_discount: &mut Vec<f64>,
) {
    read_column_i32_into(file, rg, COL_SHIPDATE, shipdate_buf).unwrap();
    read_column_i64_into(file, rg, COL_PARTKEY, partkey_buf).unwrap();
    read_column_f64_into(file, rg, COL_EXTENDEDPRICE, extprice_buf).unwrap();
    read_column_f64_into(file, rg, COL_DISCOUNT, discount_buf).unwrap();

    out_partkey.clear();
    out_extprice.clear();
    out_discount.clear();
    for (i, &d) in shipdate_buf.iter().enumerate() {
        if (LO..HI).contains(&d) {
            out_partkey.push(partkey_buf[i]);
            out_extprice.push(extprice_buf[i]);
            out_discount.push(discount_buf[i]);
        }
    }
}

// ============================================================
// Strategy 2: façade-level late-mat
// ============================================================

/// Late-mat via the new Π.10a façade:
///   1. Decode shipdate.
///   2. Build a packed mask via `build_packed_mask`.
///   3. For each other column, call `read_column_*_masked_into`
///      with the mask. Only matching rows are materialised; dict
///      pages use `gather_dict_at_bitmap_into` per page; PLAIN pages
///      use `plain_sparse_decode_*_into`.
fn late_mat_facade(
    file: &ParquetFile,
    rg: usize,
    shipdate_buf: &mut Vec<i32>,
    out_partkey: &mut Vec<i64>,
    out_extprice: &mut Vec<f64>,
    out_discount: &mut Vec<f64>,
) {
    read_column_i32_into(file, rg, COL_SHIPDATE, shipdate_buf).unwrap();

    let mask = build_packed_mask(shipdate_buf.len(), |i| {
        let d = shipdate_buf[i];
        (LO..HI).contains(&d)
    });

    out_partkey.clear();
    out_extprice.clear();
    out_discount.clear();

    read_column_i64_masked_into(file, rg, COL_PARTKEY, &mask, out_partkey).unwrap();
    read_column_f64_masked_into(file, rg, COL_EXTENDEDPRICE, &mask, out_extprice).unwrap();
    read_column_f64_masked_into(file, rg, COL_DISCOUNT, &mask, out_discount).unwrap();
}

// ============================================================
// Driver
// ============================================================

fn main() {
    let dir = match data_dir() {
        Some(d) => d,
        None => {
            println!(
                "TPC-H SF=1 lineitem not found.\n\
                 Set TPCH_DATA_DIR=/path/to/sf1 (must contain lineitem.parquet)\n\
                 or check out ematix-flow alongside this repo with its sf1 fixture."
            );
            return;
        }
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        println!("lineitem.parquet not at {}", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open lineitem");
    let md = file.metadata().expect("metadata");
    let total_rg0 = md.row_groups[0].columns[COL_SHIPDATE]
        .meta_data
        .as_ref()
        .expect("inline col meta")
        .num_values as usize;

    println!("== Q14 façade-level late-materialization ==");
    println!("data: {}", path.display());
    println!("row_group 0: {total_rg0} rows");
    println!("filter: l_shipdate ∈ [{LO}, {HI})  // 1995-09-01 .. 1995-10-01");
    println!("project: l_partkey (INT64), l_extendedprice (DOUBLE), l_discount (DOUBLE)");
    println!("warmups: {WARMUPS}, iters: {ITERS}");
    println!();

    // Caller-owned buffers reused across iterations.
    let mut shipdate_buf: Vec<i32> = Vec::with_capacity(total_rg0);
    let mut partkey_buf: Vec<i64> = Vec::with_capacity(total_rg0);
    let mut extprice_buf: Vec<f64> = Vec::with_capacity(total_rg0);
    let mut discount_buf: Vec<f64> = Vec::with_capacity(total_rg0);
    let mut out_pk: Vec<i64> = Vec::with_capacity(total_rg0 / 50);
    let mut out_ep: Vec<f64> = Vec::with_capacity(total_rg0 / 50);
    let mut out_dc: Vec<f64> = Vec::with_capacity(total_rg0 / 50);

    // --- Strategy 1: baseline ---
    let (baseline_med, baseline_min, baseline_max) = bench(|| {
        baseline_full_decode(
            &file,
            0,
            &mut shipdate_buf,
            &mut partkey_buf,
            &mut extprice_buf,
            &mut discount_buf,
            &mut out_pk,
            &mut out_ep,
            &mut out_dc,
        );
    });
    let baseline_matches = out_pk.len();

    // --- Strategy 2: façade late-mat ---
    let (latemat_med, latemat_min, latemat_max) = bench(|| {
        late_mat_facade(
            &file,
            0,
            &mut shipdate_buf,
            &mut out_pk,
            &mut out_ep,
            &mut out_dc,
        );
    });
    let latemat_matches = out_pk.len();

    // Sanity: both strategies must agree on row count.
    assert_eq!(
        baseline_matches, latemat_matches,
        "match-count mismatch: baseline={baseline_matches}, late-mat={latemat_matches}"
    );

    // --- Output ---
    let sel = baseline_matches as f64 / total_rg0 as f64 * 100.0;
    println!("matches: {baseline_matches} ({sel:.2}% selectivity)");
    println!();
    println!(
        "  baseline (4× full decode + filter): median {}  min {}  max {}",
        fmt(baseline_med),
        fmt(baseline_min),
        fmt(baseline_max)
    );
    println!(
        "  late-mat (façade _masked_into)    : median {}  min {}  max {}",
        fmt(latemat_med),
        fmt(latemat_min),
        fmt(latemat_max)
    );

    let speedup = baseline_med.as_secs_f64() / latemat_med.as_secs_f64();
    let pct_faster = (1.0 - latemat_med.as_secs_f64() / baseline_med.as_secs_f64()) * 100.0;
    let arrow = if speedup >= 1.0 { "✓" } else { "✗" };
    println!();
    println!(
        "  {arrow} speedup: {:.2}× ({:.1}% faster)",
        speedup, pct_faster
    );

    println!();
    println!("Per the Π.10 design doc, target end-to-end Q14 in ematix-flow is");
    println!("≤ 13.0 ms (beats Polars 12.53 with breathing room). This bench");
    println!("measures the codec-layer contribution; engine-layer overhead is");
    println!("in ematix-flow's own benchmark.");
}
