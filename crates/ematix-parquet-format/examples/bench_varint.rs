//! Microbench: `read_uvarint` throughput on representative data.
//!
//! Run: `cargo run --release -p ematix-parquet-format --example bench_varint`
//!
//! Three input shapes:
//!   1. all-1-byte (values 0..127, the metadata field-ID common case)
//!   2. mixed (1-3 byte values, the RLE/DELTA chunk-header common case)
//!   3. all-10-byte (worst case, u64::MAX values)
//!
//! ## Negative-result note
//!
//! This bench was written during a 2026-05 hot-path opportunity survey.
//! Several alternative implementations were attempted (≥10-byte fast
//! path with `get_unchecked` indexing, 10-step manual unroll). All were
//! SLOWER than the existing per-byte `read_u8` loop:
//!
//! | Variant | 1byte ns/value | mixed ns/value | 10byte ns/value |
//! |---------|---------------:|---------------:|----------------:|
//! | baseline (per-byte read_u8) | 0.70 | 0.75 | 2.25 |
//! | fast-path `for i in 0..10` get_unchecked | 2.26 | 2.00 | 2.64 |
//! | manual unroll b0..b9 | 2.36 | 1.98 | 2.37 |
//!
//! Conclusion: LLVM already elides the per-byte bounds check via
//! inlining + CFG analysis. The baseline's tight 5-instruction inner
//! loop is at or near the M-series cycle floor for the 1-byte case
//! (~2.8 cycles at 4 GHz). Don't reattempt without a profile showing
//! varint decode is on the hot path of a specific workload.
//!
//! The bench stays as a regression guard — if varint perf regresses
//! below the baseline numbers above, something has changed in compiler
//! behavior or in the function body that warrants investigation.

use ematix_parquet_format::compact::{read_uvarint, Cursor};

fn encode_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
}

fn build_input(shape: &str, n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    match shape {
        "1byte" => {
            for i in 0..n {
                encode_uvarint(&mut out, (i & 0x7F) as u64);
            }
        }
        "mixed" => {
            // Mixture: roughly 60% 1-byte, 30% 2-byte, 10% 3-byte
            for i in 0..n {
                let v = match i % 10 {
                    0..=5 => (i & 0x7F) as u64,
                    6..=8 => 0x80 + (i as u64 & 0x3FFF),
                    _ => 0x4000 + (i as u64 & 0x1F_FFFF),
                };
                encode_uvarint(&mut out, v);
            }
        }
        "10byte" => {
            for _ in 0..n {
                encode_uvarint(&mut out, u64::MAX);
            }
        }
        _ => unreachable!(),
    }
    out
}

fn time_read(bytes: &[u8], n: usize) -> std::time::Duration {
    let mut cur = Cursor::new(bytes);
    let start = std::time::Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..n {
        let v = read_uvarint(&mut cur).unwrap();
        sink = sink.wrapping_add(v);
    }
    let elapsed = start.elapsed();
    std::hint::black_box(sink);
    elapsed
}

fn main() {
    let n = 1_000_000;
    let warmup_runs = 3;
    let timed_runs = 10;

    println!(
        "read_uvarint microbench — {} values per run, {} timed runs each",
        n, timed_runs
    );
    println!();

    for shape in &["1byte", "mixed", "10byte"] {
        let bytes = build_input(shape, n);
        // Warmup
        for _ in 0..warmup_runs {
            let _ = time_read(&bytes, n);
        }
        // Timed
        let mut samples: Vec<std::time::Duration> = (0..timed_runs)
            .map(|_| time_read(&bytes, n))
            .collect();
        samples.sort();
        let median = samples[timed_runs / 2];
        let ns_per_value = median.as_nanos() as f64 / n as f64;
        let mb_per_s = (bytes.len() as f64 / 1e6) / median.as_secs_f64();
        println!(
            "  {:6}  median {:>7.2} ms  ({:>5.2} ns/value, {:>7.1} MB/s in {:>9} bytes)",
            shape,
            median.as_secs_f64() * 1e3,
            ns_per_value,
            mb_per_s,
            bytes.len()
        );
    }
}
