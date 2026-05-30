//! REV.12 decode-side microbench: `read_column_i64_downcast` vs
//! `read_column_i64` on a real INT64 column.
//!
//! Confirms (1) the narrowed footprint actually shrinks and (2) the
//! per-value narrowing cast does not regress decode throughput vs the
//! memcpy i64 path — the de-risk before investing in flow-side
//! consumption (where the agg/cache win is realized).
//!
//! Usage:
//!   PARQUET=/abs/path/lineitem.parquet COL=0 RGS=8 TRIALS=5 \
//!     cargo run --release -p ematix-parquet-codec --example bench_downcast
//!
//! Defaults to the ematix-flow SF=100 lineitem; COL=0 is l_orderkey
//! (INT64, max ~600M at SF=100 → narrows to i32). RGS=0 means all row
//! groups.

use std::time::Instant;

use ematix_parquet_codec::read::{read_column_i64, read_column_i64_downcast};
use ematix_parquet_io::ParquetFile;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn time_it(label: &str, warmups: usize, trials: usize, mut f: impl FnMut() -> usize) -> f64 {
    for _ in 0..warmups {
        std::hint::black_box(f());
    }
    let mut samples: Vec<f64> = Vec::with_capacity(trials);
    let mut nvals = 0usize;
    for _ in 0..trials {
        let t = Instant::now();
        nvals = f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let in_gb = (nvals * 8) as f64 / 1e9;
    let gbps = in_gb / (p50 / 1000.0);
    println!("  {label:<30} p50 {p50:9.2} ms   {nvals:>12} vals   {gbps:6.2} GB/s (i64-input)");
    p50
}

fn main() {
    let path = std::env::var("PARQUET").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/RustroverProjects/ematix-flow/examples/tpch/data/sf100/lineitem.parquet")
    });
    let col = env_usize("COL", 0);
    let trials = env_usize("TRIALS", 5);
    let warmups = env_usize("WARMUPS", 2);

    let file = ParquetFile::open(&path).expect("open parquet file");
    let n_rg = file.cached_metadata().expect("metadata").row_groups.len();
    let rgs_req = env_usize("RGS", 8);
    let rg_count = if rgs_req == 0 { n_rg } else { rgs_req.min(n_rg) };

    println!("=== REV.12 decode-side microbench ===");
    println!("file   : {path}");
    println!("column : {col}   row_groups: {rg_count}/{n_rg}   trials: {trials} (+{warmups} warmup)\n");

    // Correctness + footprint on row group 0.
    let nd = read_column_i64_downcast(&file, 0, col).expect("downcast rg0");
    let plain = read_column_i64(&file, 0, col).expect("plain rg0");
    assert_eq!(
        nd.to_i64(),
        plain,
        "downcast values must match the plain i64 decode"
    );
    let dc_bytes = nd.byte_size();
    let i64_bytes = plain.len() * 8;
    println!(
        "rg0: {} values | target {:?} | footprint {:.1} MB (downcast) vs {:.1} MB (i64) = {:.2}x smaller\n",
        nd.len(),
        nd.target(),
        dc_bytes as f64 / 1e6,
        i64_bytes as f64 / 1e6,
        i64_bytes as f64 / dc_bytes.max(1) as f64,
    );

    let rgs: Vec<usize> = (0..rg_count).collect();

    println!("throughput over {rg_count} row groups:");
    let base = time_it("read_column_i64 (baseline)", warmups, trials, || {
        let mut total = 0;
        for &rg in &rgs {
            let v = read_column_i64(&file, rg, col).unwrap();
            total += v.len();
            std::hint::black_box(&v);
        }
        total
    });
    let dc = time_it("read_column_i64_downcast", warmups, trials, || {
        let mut total = 0;
        for &rg in &rgs {
            let nd = read_column_i64_downcast(&file, rg, col).unwrap();
            total += nd.len();
            std::hint::black_box(&nd);
        }
        total
    });

    let delta = (dc - base) / base * 100.0;
    println!(
        "\nverdict: downcast decode {} baseline by {:+.1}%  ({})",
        if delta <= 0.0 { "BEATS/ties" } else { "is slower than" },
        delta,
        if delta <= 5.0 {
            "no meaningful decode regression — footprint win is free"
        } else {
            "decode regression — narrowing cast costs more than the smaller output saves"
        }
    );
}
