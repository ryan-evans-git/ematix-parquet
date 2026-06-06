//! Σ.Q06.SF10.7 probe — what does a fresh ParquetFile::open +
//! footer-parse cost on SF=10 lineitem, and how much does the flow
//! scan waste re-opening ~232×/query (per-RG × per-pass)?

use ematix_parquet_io::ParquetFile;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/tpch/data/sf10/lineitem.parquet".to_string());
    let opens: usize = std::env::var("OPENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(232);

    // One open to report footer size + RG count.
    let f = ParquetFile::open(&path).unwrap();
    let md = f.cached_metadata().unwrap();
    println!(
        "file: {path}\nrow_groups: {}, columns: {}",
        md.row_groups.len(),
        md.row_groups.first().map(|r| r.columns.len()).unwrap_or(0),
    );

    // (A) open only (footer read + alloc, no parse).
    let t = Instant::now();
    for _ in 0..opens {
        let f = ParquetFile::open(&path).unwrap();
        std::hint::black_box(&f);
    }
    let a = t.elapsed().as_secs_f64() * 1000.0;

    // (B) open + first cached_metadata (footer read + alloc + parse).
    let t = Instant::now();
    for _ in 0..opens {
        let f = ParquetFile::open(&path).unwrap();
        let m = f.cached_metadata().unwrap();
        std::hint::black_box(m.row_groups.len());
    }
    let b = t.elapsed().as_secs_f64() * 1000.0;

    println!("\n{opens} iterations (single-threaded):");
    println!(
        "  (A) open only            : {a:.2} ms  ({:.1} µs/open)",
        a * 1000.0 / opens as f64
    );
    println!(
        "  (B) open + parse footer  : {b:.2} ms  ({:.1} µs/open)",
        b * 1000.0 / opens as f64
    );
    println!(
        "\nQ06 does ~232 opens/query across ~14 worker threads.\n  → ~{:.2} ms CPU, ~{:.2} ms wall-equiv (÷14) if all redundant",
        b,
        b / 14.0
    );
}
