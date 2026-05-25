//! Π.17–Π.20 sidecar-index perf bench.
//!
//! Backs the ≥ 10× "sidecar wins on selective predicates" claim in
//! the plan doc with actual numbers. Writes a synthetic
//! sorted-INT64 column to a temp parquet, builds a sidecar index
//! against it, and times two strategies for a typical "where col =
//! v" / "where col BETWEEN lo AND hi" query:
//!
//! 1. **Baseline (full scan + filter)**: `read_column_i64` (decodes
//!    the entire chunk), then a single-pass scalar filter.
//! 2. **Indexed (sidecar lookup + masked decode)**:
//!    `ParquetIndex::read_column_i64_where_eq` /
//!    `read_column_i64_where_range`. The sidecar locates the
//!    matching pages, and the codec's `read_column_*_masked_into`
//!    only decodes (and only decompresses) the rows that actually
//!    match.
//!
//! Selectivity is swept across 1%, 5%, 10%, 50%, and 100% so the
//! crossover where indexing stops paying for itself is visible.
//!
//! Usage:
//!   cargo run --release --example bench_indexed_lookup
//!
//! Notes:
//! - Synthetic data only — no TPC-H fixture required. Lives in a
//!   per-process tempdir.
//! - The column has `N_UNIQUE` distinct values arranged in sorted,
//!   contiguous runs (each value spans `N_ROWS / N_UNIQUE` rows).
//!   This is the natural shape for an indexed column — categorical
//!   IDs, dates, low-cardinality keys. Truly unique-per-row data
//!   isn't a realistic sorted-index target (the in-memory bitmap
//!   set during builder would be `O(distinct_values × page_bitmap_len)`
//!   bytes, which doesn't pay back; for cardinality≈rows, use a
//!   page-Bloom or composite-prefix index instead).
//! - The bench reports median wall-time over `ITERS` measured runs
//!   after `WARMUPS` warm-ups (same shape as `bench_q14_late_mat`).

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use ematix_parquet_codec::index::{IndexBuilder, ParquetIndex};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::{write_table_to_path_with_row_group_size, ColumnData};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

const N_ROWS: usize = 1_000_000;
/// Distinct values in the indexed column. With `N_ROWS = 1M` and
/// `N_UNIQUE = 100`, each value appears in `10_000` rows.
const N_UNIQUE: i64 = 100;
/// Row-group size in the source parquet. Keeping this small (10K
/// rows = 1% of N_ROWS) makes the source split into ~100 row
/// groups — every unique value lives in exactly one row group, so
/// the sidecar lookup can skip ~99% of decompress work, which is
/// where the perf win lives. Mirrors the column-chunk granularity
/// real-world Iceberg writers produce on sorted-ingest tables.
const ROW_GROUP_SIZE: usize = 10_000;
const WARMUPS: usize = 2;
const ITERS: usize = 8;

/// Run `f` `WARMUPS` times to warm caches, then `ITERS` times
/// measured. Returns the median.
fn bench<R>(mut f: impl FnMut() -> R) -> Duration {
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
    samples[samples.len() / 2]
}

fn fmt_ms(d: Duration) -> String {
    format!("{:>9.3} ms", d.as_secs_f64() * 1e3)
}

fn n_row_groups(file: &ParquetFile) -> usize {
    file.metadata().unwrap().row_groups.len()
}

/// Baseline strategy: decode every row group of the INT64 column in
/// turn, scalar-filter the values in `[lo, hi)` and accumulate.
/// Represents the "no sidecar" path — every row group is fully
/// decompressed + decoded.
fn baseline_full_scan(file: &ParquetFile, lo: i64, hi: i64) -> Vec<i64> {
    let mut out = Vec::new();
    for rg in 0..n_row_groups(file) {
        let all = read_column_i64(file, rg, 0).unwrap();
        for v in all {
            if v >= lo && v < hi {
                out.push(v);
            }
        }
    }
    out
}

/// Equality baseline — same shape, scalar `==` filter.
fn baseline_full_scan_eq(file: &ParquetFile, key: i64) -> Vec<i64> {
    let mut out = Vec::new();
    for rg in 0..n_row_groups(file) {
        let all = read_column_i64(file, rg, 0).unwrap();
        for v in all {
            if v == key {
                out.push(v);
            }
        }
    }
    out
}

/// Build the source parquet + sidecar in `dir`. Returns the source
/// path and the sidecar path.
fn build_fixture(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = dir.join("data.parquet");
    let idx = dir.join("data.parquet.idx");
    // N_UNIQUE distinct values, each in a contiguous sorted run of
    // N_ROWS/N_UNIQUE rows. Realistic for indexed categorical /
    // low-cardinality columns (date, region, customer tier, …).
    let stride = (N_ROWS as i64) / N_UNIQUE;
    let values: Vec<i64> = (0..N_ROWS as i64).map(|i| i / stride).collect();
    write_table_to_path_with_row_group_size(
        &src,
        &[("v", ColumnData::I64(&values))],
        CompressionCodec::Uncompressed,
        ROW_GROUP_SIZE,
    )
    .expect("write source");
    let source = ParquetFile::open(&src).expect("open source");
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .expect("build sidecar");
    (src, idx)
}

fn main() {
    println!("--- bench_indexed_lookup ---");
    println!("rows: {N_ROWS}, warmups: {WARMUPS}, iters: {ITERS}");
    println!();

    let tmp = tempfile::tempdir().expect("tempdir");
    let (src, idx) = build_fixture(tmp.path());
    let source = ParquetFile::open(&src).expect("reopen source");
    let reader = ParquetIndex::open(&idx, &source).expect("open sidecar");
    println!(
        "source row groups: {}  (target {} rows/RG)",
        n_row_groups(&source),
        ROW_GROUP_SIZE
    );
    println!();

    // Print a header so the wins/losses line up in a single column.
    println!(
        "{:<26} {:>13} {:>13} {:>10}",
        "predicate (selectivity)", "baseline", "indexed", "speedup"
    );
    println!("{:-<26} {:->13} {:->13} {:->10}", "", "", "", "");

    // ============================================================
    // Equality probes — the sweet spot for sorted-index lookup.
    // Selectivity is fixed at `1 / N_UNIQUE` per key.
    // ============================================================
    let eq_keys: [(i64, &str); 3] = [
        (0, "eq @ min"),
        (N_UNIQUE / 2, "eq @ mid"),
        (N_UNIQUE - 1, "eq @ max"),
    ];
    for (key, label) in eq_keys {
        let baseline = bench(|| baseline_full_scan_eq(&source, key));
        let indexed = bench(|| {
            reader
                .read_column_i64_where_eq(&source, "idx_v", key, 0)
                .unwrap()
        });
        let speedup = baseline.as_secs_f64() / indexed.as_secs_f64();
        println!(
            "{:<26} {} {} {:>9.2}×",
            label,
            fmt_ms(baseline),
            fmt_ms(indexed),
            speedup
        );
    }

    // ============================================================
    // Range probes — show how the curve flattens as selectivity grows.
    // `hi` is exclusive in the baseline (`< hi`) and inclusive in
    // the indexed call (`<= hi - 1`), so the two match exactly.
    // ============================================================
    let range_cases: [(i64, i64, &str); 5] = [
        (0, N_UNIQUE / 100, "range 1%"),
        (0, N_UNIQUE / 20, "range 5%"),
        (0, N_UNIQUE / 10, "range 10%"),
        (0, N_UNIQUE / 2, "range 50%"),
        (0, N_UNIQUE, "range 100%"),
    ];
    for (lo, hi, label) in range_cases {
        let baseline = bench(|| baseline_full_scan(&source, lo, hi));
        let indexed = bench(|| {
            reader
                .read_column_i64_where_range(&source, "idx_v", lo, hi - 1, 0)
                .unwrap()
        });
        let speedup = baseline.as_secs_f64() / indexed.as_secs_f64();
        println!(
            "{:<26} {} {} {:>9.2}×",
            label,
            fmt_ms(baseline),
            fmt_ms(indexed),
            speedup
        );
    }

    // ============================================================
    // Correctness sanity — bench wins are worthless if the indexed
    // path silently drops rows. Spot-check mid-eq and a 1% range.
    // ============================================================
    let mid = N_UNIQUE / 2;
    let baseline = baseline_full_scan_eq(&source, mid);
    let indexed = reader
        .read_column_i64_where_eq(&source, "idx_v", mid, 0)
        .unwrap();
    assert_eq!(
        baseline, indexed,
        "indexed eq result diverged from baseline"
    );

    let baseline = baseline_full_scan(&source, 0, N_UNIQUE / 100);
    let indexed = reader
        .read_column_i64_where_range(&source, "idx_v", 0, N_UNIQUE / 100 - 1, 0)
        .unwrap();
    assert_eq!(
        baseline, indexed,
        "indexed range result diverged from baseline"
    );

    println!();
    println!("correctness: indexed result matches baseline on spot checks");
}
