//! DOWNCAST.CONT — decode-layer cost of INT32-source narrowing.
//!
//! The footprint sweep (bench_downcast_taxonomy) showed DATE(INT32)->i16 and
//! l_linenumber(INT32)->i8 are the columns the new INT32 slice unlocks. The
//! gate question for any flow-side use is: does narrowing during decode COST
//! throughput (like the i64->i32 +4% scalar-cast tax) or is it free/faster?
//!
//! This isolates exactly that — same PLAIN input bytes, decode to the full
//! width (decode_plain_i32 -> Vec<i32>) vs narrow to the target
//! (decode_plain_i32_narrowed -> Vec<i16>/Vec<i8>). Decompression is identical
//! for both arms in a real read, so it's factored out. Synthetic but the cast
//! cost is the same operation as on a real column.
//!
//! Usage:
//!   N=16000000 TRIALS=9 cargo run --release -p ematix-parquet-codec \
//!     --example bench_i32_narrow_decode

use std::time::Instant;

use ematix_parquet_codec::downcast::{decode_plain_i32_narrowed, IntTarget};
use ematix_parquet_codec::plain::decode_plain_i32;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn plain_bytes_i32(vals: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn time_it(
    label: &str,
    warmups: usize,
    trials: usize,
    n: usize,
    mut f: impl FnMut() -> usize,
) -> f64 {
    for _ in 0..warmups {
        std::hint::black_box(f());
    }
    let mut samples: Vec<f64> = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t = Instant::now();
        let got = f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(got);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let in_gbps = (n * 4) as f64 / 1e9 / (p50 / 1000.0); // input is 4 bytes/value
    println!("  {label:<34} p50 {p50:8.3} ms   {in_gbps:6.2} GB/s (i32-input)");
    p50
}

fn run_case(name: &str, vals: &[i32], target: IntTarget, warmups: usize, trials: usize) {
    let n = vals.len();
    let bytes = plain_bytes_i32(vals);
    let out_w = target.width_bytes();
    println!(
        "\n{name}: {n} values  ({:.1} MB i32 input -> {:.1} MB {:?} output)",
        bytes.len() as f64 / 1e6,
        (n * out_w) as f64 / 1e6,
        target,
    );
    let base = time_it(
        "decode_plain_i32 (baseline ->i32)",
        warmups,
        trials,
        n,
        || decode_plain_i32(&bytes).unwrap().len(),
    );
    let narr = time_it(
        "decode_plain_i32_narrowed (->narrow)",
        warmups,
        trials,
        n,
        || decode_plain_i32_narrowed(&bytes, target).unwrap().len(),
    );
    let delta = (narr - base) / base * 100.0;
    println!(
        "  => narrowing {} baseline by {:+.1}%  ({})",
        if delta <= 0.0 {
            "BEATS/ties"
        } else {
            "is slower than"
        },
        delta,
        if delta <= 2.0 {
            "no decode tax — footprint win is free at the decode layer"
        } else {
            "decode tax — per-value cast costs more than the smaller write saves"
        }
    );
}

fn main() {
    let n = env_usize("N", 16_000_000);
    let trials = env_usize("TRIALS", 9);
    let warmups = env_usize("WARMUPS", 3);

    println!("=== INT32-source narrowing: decode-layer cost ===");
    println!("n: {n}   trials: {trials} (+{warmups} warmup)");

    // DATE(INT32) shape: days-since-epoch within TPC-H's 1992-1998 window.
    let dates: Vec<i32> = (0..n).map(|i| 8035 + (i % 2557) as i32).collect();
    run_case(
        "DATE(INT32) -> i16",
        &dates,
        IntTarget::I16,
        warmups,
        trials,
    );

    // l_linenumber shape: 1..=7.
    let lineno: Vec<i32> = (0..n).map(|i| 1 + (i % 7) as i32).collect();
    run_case(
        "l_linenumber(INT32) -> i8",
        &lineno,
        IntTarget::I8,
        warmups,
        trials,
    );
}
