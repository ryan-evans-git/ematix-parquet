//! Micro-bench: hand-rolled Snappy vs the `snap` crate.
//!
//! Tests three input shapes typical of parquet pages:
//!   1. Random/incompressible — mostly literal tags. Bulk memcpy hot path.
//!   2. Highly compressible (constant run) — back-references dominate.
//!   3. Realistic dict-encoded payload — synthetic but mimics real parquet.
//!
//! Output: GB/s decompressed, ns/page, comparison ratio.
//!
//! Usage: cargo run --release --example bench_snappy

use std::hint::black_box;
use std::time::Instant;

use ematix_parquet_codec::compression::{decompress_snappy_fast_into, decompress_snappy_into};

const ITERS: usize = 200;

fn snappy_encode(input: &[u8]) -> Vec<u8> {
    let mut enc = snap::raw::Encoder::new();
    enc.compress_vec(input).unwrap()
}

fn time_one(label: &str, compressed: &[u8], orig_len: usize) {
    // Warmup both paths.
    let mut buf: Vec<u8> = Vec::with_capacity(orig_len);
    for _ in 0..10 {
        buf.clear();
        decompress_snappy_into(black_box(compressed), &mut buf).unwrap();
    }
    for _ in 0..10 {
        buf.clear();
        decompress_snappy_fast_into(black_box(compressed), &mut buf).unwrap();
    }

    // snap crate.
    let mut best_snap = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            buf.clear();
            decompress_snappy_into(black_box(compressed), &mut buf).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_snap {
            best_snap = dt;
        }
    }

    // Hand-rolled.
    let mut best_fast = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            buf.clear();
            decompress_snappy_fast_into(black_box(compressed), &mut buf).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_fast {
            best_fast = dt;
        }
    }

    let bytes_out_snap = orig_len * ITERS;
    let gbps_snap = bytes_out_snap as f64 / best_snap / 1e9;
    let gbps_fast = bytes_out_snap as f64 / best_fast / 1e9;
    let speedup = best_snap / best_fast;
    let ns_per_page_snap = best_snap * 1e9 / ITERS as f64;
    let ns_per_page_fast = best_fast * 1e9 / ITERS as f64;
    println!(
        "  {label:<32} snap: {:>5.2} GB/s ({:>6.0} ns/page)  fast: {:>5.2} GB/s ({:>6.0} ns/page)  speedup: {:>4.2}×",
        gbps_snap, ns_per_page_snap, gbps_fast, ns_per_page_fast, speedup
    );
}

fn main() {
    println!("Snappy decompress microbench ({} iters, best-of-5)", ITERS);

    // Case 1: incompressible random ~80 KB (typical parquet bit-packed page).
    let mut seed: u64 = 0xDEADBEEF_CAFEBABE;
    let random: Vec<u8> = (0..80 * 1024)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        })
        .collect();
    let random_c = snappy_encode(&random);
    println!(
        "  (random 80KB: input {} bytes → compressed {} bytes, ratio {:.2})",
        random.len(),
        random_c.len(),
        random.len() as f64 / random_c.len() as f64
    );
    time_one("random 80 KB", &random_c, random.len());

    // Case 2: highly compressible (constant run) — heavy back-references.
    let constant: Vec<u8> = vec![42u8; 256 * 1024];
    let constant_c = snappy_encode(&constant);
    println!(
        "  (constant 256KB: input {} bytes → compressed {} bytes, ratio {:.0})",
        constant.len(),
        constant_c.len(),
        constant.len() as f64 / constant_c.len() as f64
    );
    time_one("constant 256 KB", &constant_c, constant.len());

    // Case 3: synthetic dict-encoded payload (mimics l_extendedprice
    // bw=17 RLE_DICTIONARY page after bit-packed indices). 8 KB of
    // mostly-random bytes punctuated by short repeating patterns.
    let mut sim: Vec<u8> = Vec::with_capacity(80 * 1024);
    let mut s: u64 = 0x12345678ABCDEF01;
    while sim.len() < 80 * 1024 {
        // 90% random, 10% repeats.
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (s >> 60) == 0 {
            // Insert a short pattern.
            sim.extend_from_slice(b"PATTERN_");
        } else {
            sim.push((s >> 33) as u8);
        }
    }
    let sim_c = snappy_encode(&sim);
    println!(
        "  (synthetic dict 80KB: input {} bytes → compressed {} bytes, ratio {:.2})",
        sim.len(),
        sim_c.len(),
        sim.len() as f64 / sim_c.len() as f64
    );
    time_one("synthetic dict 80 KB", &sim_c, sim.len());
}
