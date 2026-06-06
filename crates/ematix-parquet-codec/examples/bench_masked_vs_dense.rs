//! Σ.Q06.SF10.7 probe — is masked (sparse) PLAIN decode slower than
//! dense-decode-then-gather at Q06's ~2% selectivity?
//!
//! The flow profile shows `read_column_f64_masked_into` +
//! `plain_sparse_decode_f64_into` are ~9700 samples (on par with
//! Snappy) on production Q06 SF=10. The breakdown harness showed
//! dense full-column decode+sum of l_extendedprice is 3.62 ms while
//! the masked projection path is ~42 ms. This isolates the pure
//! decode primitive on the same bytes + same mask.
//!
//! Run:
//!   cargo run --release -p ematix-parquet-codec --example bench_masked_vs_dense -- \
//!     /path/to/sf10/lineitem.parquet

use ematix_parquet_codec::read::{
    read_column_f64, read_column_f64_masked_into, read_column_i32, read_column_i32_masked_into,
};
use ematix_parquet_io::ParquetFile;
use std::time::Instant;

/// Deterministic ~`pct`% bitmap sized to `n` rows. xorshift so the set
/// bits are scattered like Q06's shipdate-random selectivity (no whole
/// 8-row blocks empty/full clustering).
fn build_mask(n: usize, pct_num: u64, pct_den: u64) -> (Vec<u8>, usize) {
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut set = 0usize;
    for row in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state % pct_den < pct_num {
            bitmap[row >> 3] |= 1 << (row & 7);
            set += 1;
        }
    }
    (bitmap, set)
}

fn gather_f64(all: &[f64], mask: &[u8], out: &mut Vec<f64>) {
    for (row, v) in all.iter().enumerate() {
        if (mask[row >> 3] >> (row & 7)) & 1 == 1 {
            out.push(*v);
        }
    }
}

fn gather_i32(all: &[i32], mask: &[u8], out: &mut Vec<i32>) {
    for (row, v) in all.iter().enumerate() {
        if (mask[row >> 3] >> (row & 7)) & 1 == 1 {
            out.push(*v);
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/tpch/data/sf10/lineitem.parquet".to_string());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let file = ParquetFile::open(&path).expect("open");
    let md = file.cached_metadata().expect("meta");
    let n_rg = md.row_groups.len();
    println!("file: {path}\nrow groups: {n_rg}, reps: {reps}\n");

    // l_extendedprice = col 5 (f64), l_shipdate = col 10 (date32→i32).
    // ~1.9% Q06 selectivity (shipdate∧discount∧quantity).
    let f64_col = 5usize;
    let i32_col = 10usize;
    let (pct_num, pct_den) = (19u64, 1000u64);

    // Pre-build per-RG masks (so mask-build cost is out of the timed loop).
    let mut masks: Vec<(Vec<u8>, usize)> = Vec::with_capacity(n_rg);
    let mut total_rows = 0usize;
    let mut total_set = 0usize;
    for rg in 0..n_rg {
        let nv = md.row_groups[rg].columns[f64_col]
            .meta_data
            .as_ref()
            .unwrap()
            .num_values as usize;
        let m = build_mask(nv, pct_num, pct_den);
        total_rows += nv;
        total_set += m.1;
        masks.push(m);
    }
    println!(
        "total rows: {total_rows}, selected: {total_set} ({:.2}%)\n",
        100.0 * total_set as f64 / total_rows as f64
    );

    // ---- f64 (l_extendedprice) ----
    let mut a_ms = vec![];
    let mut b_ms = vec![];
    let (mut a_cnt, mut b_cnt) = (0usize, 0usize);
    for _ in 0..reps {
        // (A) masked sparse
        let t = Instant::now();
        let mut out = Vec::new();
        for rg in 0..n_rg {
            read_column_f64_masked_into(&file, rg, f64_col, &masks[rg].0, &mut out).unwrap();
        }
        a_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        a_cnt = out.len();

        // (B) dense decode + gather
        let t = Instant::now();
        let mut out = Vec::new();
        for rg in 0..n_rg {
            let all = read_column_f64(&file, rg, f64_col).unwrap();
            gather_f64(&all, &masks[rg].0, &mut out);
        }
        b_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        b_cnt = out.len();
    }
    a_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("=== f64 l_extendedprice (col {f64_col}) ===");
    println!(
        "  (A) masked sparse        : {:.2} ms  (n={a_cnt})",
        a_ms[reps / 2]
    );
    println!(
        "  (B) dense decode+gather  : {:.2} ms  (n={b_cnt})",
        b_ms[reps / 2]
    );
    println!("  speedup B/A: {:.2}x\n", a_ms[reps / 2] / b_ms[reps / 2]);

    // ---- i32 (l_shipdate) ----
    let mut a_ms = vec![];
    let mut b_ms = vec![];
    let (mut a_cnt, mut b_cnt) = (0usize, 0usize);
    for _ in 0..reps {
        let t = Instant::now();
        let mut out = Vec::new();
        for rg in 0..n_rg {
            read_column_i32_masked_into(&file, rg, i32_col, &masks[rg].0, &mut out).unwrap();
        }
        a_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        a_cnt = out.len();

        let t = Instant::now();
        let mut out = Vec::new();
        for rg in 0..n_rg {
            let all = read_column_i32(&file, rg, i32_col).unwrap();
            gather_i32(&all, &masks[rg].0, &mut out);
        }
        b_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        b_cnt = out.len();
    }
    a_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("=== i32 l_shipdate (col {i32_col}) ===");
    println!(
        "  (A) masked sparse        : {:.2} ms  (n={a_cnt})",
        a_ms[reps / 2]
    );
    println!(
        "  (B) dense decode+gather  : {:.2} ms  (n={b_cnt})",
        b_ms[reps / 2]
    );
    println!("  speedup B/A: {:.2}x", a_ms[reps / 2] / b_ms[reps / 2]);
}
