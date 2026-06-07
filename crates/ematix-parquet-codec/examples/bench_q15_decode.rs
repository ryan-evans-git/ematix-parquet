//! Phase-0 de-risk (2026-06-03): raw per-column DENSE decode rate of the two
//! Q15-dominant lineitem columns, isolated from all query orchestration. Pairs
//! with a polars/pyarrow reference (scripts/q15_decode_polars.py) to decide
//! whether ematix-parquet's decoder is the Q15/Q06 gap — i.e. whether the
//! "bulk RLE/bitpack page-parse port" is worth a full effort.
//!
//!   - l_extendedprice (col 5, f64 PLAIN + Snappy): tests decompress + PLAIN decode.
//!   - l_shipdate      (col 10, Date32/i32 DICT + Snappy): tests dict-index
//!     RLE/bitpack UNPACK (the port's target) + decompress.
//!
//! Reports per column: median wall ms, values/s, and uncompressed MB/s
//! (= n_values * sizeof(T) / time) so it's directly comparable to polars.
//!
//! Run (against the LOCAL crate):
//!   cargo run --release -p ematix-parquet-codec --example bench_q15_decode -- \
//!     /abs/path/to/sf10/lineitem.parquet
//!   REPS=20 cargo run --release ... (default REPS=15, 2 warmups)

use ematix_parquet_codec::read::{read_column_f64, read_column_i32};
use ematix_parquet_io::ParquetFile;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/tpch/data/sf10/lineitem.parquet".to_string());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let warmups: usize = std::env::var("WARMUPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let file = ParquetFile::open(&path).expect("open");
    let md = file.cached_metadata().expect("meta");
    let n_rg = md.row_groups.len();

    let extprice_col = 5usize; // f64
    let shipdate_col = 10usize; // i32 (Date32)

    let mut n_vals = 0usize;
    for rg in 0..n_rg {
        n_vals += md.row_groups[rg].columns[extprice_col]
            .meta_data
            .as_ref()
            .unwrap()
            .num_values as usize;
    }
    println!("file: {path}");
    println!("row groups: {n_rg}, values/col: {n_vals}, reps: {reps} (+{warmups} warmup)\n");

    // ---- l_extendedprice (f64) ----
    let mut ms = Vec::new();
    let mut checksum = 0.0f64;
    for r in 0..(reps + warmups) {
        let t = Instant::now();
        let mut total = 0usize;
        for rg in 0..n_rg {
            let v = read_column_f64(&file, rg, extprice_col).unwrap();
            total += v.len();
            checksum += v[0] + v[v.len() - 1];
        }
        let e = t.elapsed().as_secs_f64() * 1000.0;
        if r >= warmups {
            ms.push(e);
        }
        debug_assert_eq!(total, n_vals);
    }
    let m = median(ms);
    let vps = n_vals as f64 / (m / 1000.0);
    let mbps = (n_vals as f64 * 8.0) / (m / 1000.0) / 1e6;
    println!("=== l_extendedprice (f64 PLAIN+Snappy) ===");
    println!(
        "  median {m:8.2} ms   {:.1} M values/s   {mbps:8.1} MB/s (uncompressed)",
        vps / 1e6
    );

    // ---- l_shipdate (i32 dict) ----
    let mut ms = Vec::new();
    let mut isum = 0i64;
    for r in 0..(reps + warmups) {
        let t = Instant::now();
        let mut total = 0usize;
        for rg in 0..n_rg {
            let v = read_column_i32(&file, rg, shipdate_col).unwrap();
            total += v.len();
            isum += v[0] as i64 + v[v.len() - 1] as i64;
        }
        let e = t.elapsed().as_secs_f64() * 1000.0;
        if r >= warmups {
            ms.push(e);
        }
        debug_assert_eq!(total, n_vals);
    }
    let m = median(ms);
    let vps = n_vals as f64 / (m / 1000.0);
    let mbps = (n_vals as f64 * 4.0) / (m / 1000.0) / 1e6;
    println!("=== l_shipdate (i32/Date32 DICT+Snappy) ===");
    println!(
        "  median {m:8.2} ms   {:.1} M values/s   {mbps:8.1} MB/s (uncompressed)",
        vps / 1e6
    );

    println!("\n(checksums {checksum:.1} / {isum} — prevent dead-code elim)");
}
