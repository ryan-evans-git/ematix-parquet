//! Π.15e — parallel multi-RG decode scaling bench.
//!
//! Builds a temp multi-RG parquet file (50 row groups × 65 536 rows
//! of i64, Snappy-compressed with a redundant payload so decompress
//! is the dominant cost — not memcpy), then times sequential vs
//! `read_columns_parallel` at thread counts 1, 2, 4, 8, … capped at
//! the host's CPU count. Reports per-config median + speedup +
//! efficiency.
//!
//! ## Known bottleneck
//! As of v0.10 `ParquetFile.read_range` uses lock-free positional
//! I/O (`pread(2)` on Unix; `seek_read` on Windows), so the file
//! mutex is gone. Sequential decode also got faster as a result
//! (no mutex acquisition per read).
//!
//! On this synthetic single-column 50-RG fixture the speedup
//! plateaus around 1.3–2.0× (host-dependent) — the remaining
//! bottleneck is likely DRAM bandwidth, since the whole working
//! set is small (~26 MB at 3.3M i64) and 14 cores quickly
//! saturate per-socket memory throughput on the decompress pass.
//! Real ematix-flow workloads (multi-column, complex predicates,
//! larger per-RG compute) push per-worker compute much higher
//! relative to memory traffic and should see more linear scaling.
//!
//! On Linux also runs a NUMA-pinned variant via `build_numa_pinned_pool`
//! (Π.15c+d). On macOS / Windows the NUMA path is cfg'd out so that
//! row of the report is skipped.
//!
//! Acceptance check (Π.15 plan criterion #1) requires running this on
//! a 2-socket Linux box to see the linear-to-socket-count, sub-linear-
//! beyond pattern. On a single-NUMA-node host we see plain rayon
//! scaling and the NUMA-pinned variant should match the unpinned
//! within bench noise.
//!
//! Run:
//!   cargo run --release --example bench_parallel_scaling \
//!       -p ematix-parquet-codec --features parallel

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ematix_parquet_codec::parallel::{read_columns_parallel, ParallelDecodeOptions};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::{write_table_to_path_with_row_group_size, ColumnData};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

const NUM_RGS: usize = 50;
const ROWS_PER_RG: usize = 65_536;
const WARMUPS: usize = 3;
const ITERS: usize = 11;

fn build_fixture() -> tempfile::NamedTempFile {
    // Snappy + a payload with enough redundancy to actually compress,
    // so each RG has real decompress work — otherwise the cheap
    // memcpy dominates and the parallel speedup is invisible behind
    // I/O + the ParquetFile mutex.
    let total_rows = NUM_RGS * ROWS_PER_RG;
    let values: Vec<i64> = (0..total_rows).map(|i| (i % 256) as i64).collect();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    write_table_to_path_with_row_group_size(
        tmp.path(),
        &[("v", ColumnData::I64(&values))],
        CompressionCodec::Snappy,
        ROWS_PER_RG,
    )
    .unwrap();
    tmp
}

fn sequential_decode(file: &ParquetFile) -> usize {
    let mut total = 0usize;
    for rg in 0..NUM_RGS {
        let v = read_column_i64(file, rg, 0).unwrap();
        total += v.len();
    }
    total
}

fn parallel_decode(file: &ParquetFile, opts: ParallelDecodeOptions) -> usize {
    let targets: Vec<(usize, usize)> = (0..NUM_RGS).map(|rg| (rg, 0)).collect();
    let out = read_columns_parallel(file, &targets, opts, read_column_i64).unwrap();
    out.iter().map(|v| v.len()).sum()
}

fn bench<R>(label: &str, mut f: impl FnMut() -> R) -> Duration {
    for _ in 0..WARMUPS {
        black_box(f());
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        black_box(f());
        times.push(start.elapsed());
    }
    times.sort();
    let med = times[ITERS / 2];
    println!(
        "  {label:<36} median {:>7.2} ms  min {:>7.2} ms",
        med.as_secs_f64() * 1000.0,
        times[0].as_secs_f64() * 1000.0,
    );
    med
}

#[cfg(target_os = "linux")]
fn print_speedup(label: &str, baseline: Duration, current: Duration) {
    let speedup = baseline.as_secs_f64() / current.as_secs_f64();
    println!("    → {label}: speedup {speedup:.2}× vs sequential",);
}

fn main() {
    println!(
        "== Π.15e parallel multi-RG decode scaling ({WARMUPS} warmups + {ITERS} iters per cell) =="
    );
    println!(
        "fixture: {NUM_RGS} row groups × {ROWS_PER_RG} rows = {} rows total",
        NUM_RGS * ROWS_PER_RG
    );
    println!("column: i64 Snappy-compressed (redundant payload for real decompress work)\n");

    let tmp = build_fixture();
    let file = ParquetFile::open(tmp.path()).unwrap();

    // Sanity: every path produces the same total row count.
    let seq_rows = sequential_decode(&file);
    let par_rows = parallel_decode(&file, ParallelDecodeOptions::default());
    assert_eq!(seq_rows, par_rows);
    assert_eq!(seq_rows, NUM_RGS * ROWS_PER_RG);

    let host_cpus = num_cpus_best_effort();
    println!("host CPUs: {host_cpus}\n");

    println!("Sequential baseline:");
    let t_seq = bench("sequential", || sequential_decode(&file));
    println!();

    println!("Plain rayon pool (caller-owned), varying num_threads:");
    let thread_counts: Vec<usize> = thread_count_sweep(host_cpus);
    let mut per_count = Vec::new();
    for &n in &thread_counts {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap(),
        );
        let label = format!("parallel(N={n})");
        let t = bench(&label, || {
            parallel_decode(
                &file,
                ParallelDecodeOptions {
                    pool: Some(pool.clone()),
                    cancel: None,
                },
            )
        });
        per_count.push((n, t));
    }
    println!();

    println!("Scaling summary (vs sequential):");
    for (n, t) in &per_count {
        let speedup = t_seq.as_secs_f64() / t.as_secs_f64();
        let efficiency = 100.0 * speedup / (*n as f64);
        println!("  N={n:>3}  speedup {speedup:>5.2}×   efficiency {efficiency:>5.1}%");
    }
    println!();

    // NUMA-pinned variant — Linux-only.
    numa_section(&file, t_seq, host_cpus);
}

#[cfg(target_os = "linux")]
fn numa_section(file: &ParquetFile, t_seq: Duration, host_cpus: usize) {
    use ematix_parquet_codec::parallel::numa::{build_numa_pinned_pool, NumaTopology};
    let topology = match NumaTopology::detect() {
        Ok(t) => t,
        Err(e) => {
            println!("NUMA-pinned: skipped (topology detect failed: {e})");
            return;
        }
    };
    println!(
        "NUMA-pinned pool (Π.15c+d): {} NUMA node{}, {} total CPUs",
        topology.num_nodes(),
        if topology.num_nodes() == 1 { "" } else { "s" },
        topology.all_cpus().len()
    );
    if topology.num_nodes() == 1 {
        println!("  (single-node host — pinned/unpinned should match within noise)");
    }
    let pool = Arc::new(build_numa_pinned_pool(host_cpus).expect("pool"));
    let t_numa = bench("parallel(NUMA-pinned)", || {
        parallel_decode(
            file,
            ParallelDecodeOptions {
                pool: Some(pool.clone()),
                cancel: None,
            },
        )
    });
    print_speedup("NUMA-pinned", t_seq, t_numa);
}

#[cfg(not(target_os = "linux"))]
fn numa_section(_file: &ParquetFile, _t_seq: Duration, _host_cpus: usize) {
    println!("NUMA-pinned: skipped (not Linux — module is cfg'd out)");
}

fn thread_count_sweep(host_cpus: usize) -> Vec<usize> {
    let mut out: Vec<usize> = vec![1, 2];
    let mut n = 4;
    while n <= host_cpus {
        out.push(n);
        n *= 2;
    }
    // Always include host_cpus itself if it isn't already a power of 2.
    if *out.last().unwrap() != host_cpus && host_cpus > 1 {
        out.push(host_cpus);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn num_cpus_best_effort() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
