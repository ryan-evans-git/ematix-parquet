//! Π.9d — Microbench for `pld_l1` prefetch hints in
//! `gather_dict_at_bitmap_into`.
//!
//! Sized so the dict overflows L1 (typical 128–192 KB on M-series),
//! which is the regime where software prefetch can actually shave
//! cycles off the gather. For small dicts (< L1) the prefetch is
//! near-free and should be within noise.
//!
//! We can't A/B the prefetch at runtime (it's compiled in for
//! aarch64), but the report shows absolute throughput so you can
//! eyeball it vs the prior baseline reported in the plan.
//!
//! Run:
//!   cargo run --release --example bench_dict_gather_prefetch \
//!       -p ematix-parquet-codec

use std::hint::black_box;
use std::time::Instant;

use ematix_parquet_codec::dict::gather_dict_at_bitmap_into;

const ROWS: usize = 1_000_000;
const BIT_WIDTH: u8 = 16; // matches the typical TPC-H dict size band
const WARMUPS: usize = 3;
const ITERS: usize = 11;

/// Build a bit-packed RLE_DICTIONARY data-page body for `indices` at
/// the requested bit_width. One single bit-packed run (no RLE), to
/// keep the gather measurement focused on the unpack + dict path.
fn build_body(indices: &[u32], bit_width: u8) -> Vec<u8> {
    assert!(indices.len() % 8 == 0, "use a multiple of 8 indices");
    let count_8groups = indices.len() / 8;
    let header = ((count_8groups as u64) << 1) | 1; // bit-packed
    let mut out = vec![bit_width];
    // uvarint encode header
    let mut h = header;
    loop {
        let mut b = (h & 0x7F) as u8;
        h >>= 7;
        if h != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
    // bit-pack the indices LSB-first
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    for &idx in indices {
        acc |= (idx as u64) << bits;
        bits += bit_width as u32;
        while bits >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

fn run_bench(label: &str, selectivity_pct: u32, dict_size: usize) {
    // Build a `Vec<u64>` dict — 8 bytes per entry × dict_size →
    // 16 KB at dict_size = 2048 (fits L1) up to 1 MB at dict_size =
    // 131072 (L1-miss territory).
    let dict: Vec<u64> = (0..dict_size as u64)
        .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
        .collect();

    // Indices uniformly across dict — forces gather to chase
    // throughout the dict footprint.
    let mut seed: u64 = 0xCAFEBABE;
    let indices: Vec<u32> = (0..ROWS)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % dict_size as u32
        })
        .collect();

    let body = build_body(&indices, BIT_WIDTH);

    // Bitmap with the requested selectivity.
    let bitmap_bytes = ROWS.div_ceil(8);
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut s2: u64 = 0xDEADBEEF;
    for i in 0..ROWS {
        s2 = s2
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (s2 >> 32) % 100 < selectivity_pct as u64 {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }

    let mut out: Vec<u64> = Vec::with_capacity(ROWS);

    // Warmup
    for _ in 0..WARMUPS {
        out.clear();
        gather_dict_at_bitmap_into(&body, ROWS, &bitmap, 0, &dict, &mut out).unwrap();
        black_box(&out);
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        gather_dict_at_bitmap_into(&body, ROWS, &bitmap, 0, &dict, &mut out).unwrap();
        times.push(t0.elapsed());
        black_box(&out);
    }
    times.sort();
    let med = times[ITERS / 2];
    let min = times[0];
    let gathered = out.len();
    let rows_per_ms = ROWS as f64 / med.as_secs_f64() / 1000.0;
    println!(
        "  {label:<44} median {:>7.2} ms  min {:>7.2} ms  ({gathered:>7} gathered, {rows_per_ms:>8.1} K rows/ms)",
        med.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!("== Π.9d dict-gather prefetch bench ({WARMUPS} warmups + {ITERS} iters per cell) ==");
    println!("rows={ROWS}, bit_width={BIT_WIDTH}\n");

    for dict_size in [2_048usize, 16_384, 131_072] {
        let footprint_kb = dict_size * 8 / 1024;
        println!("dict_size={dict_size} ({footprint_kb} KB):");
        for sel in [1u32, 10, 50, 100] {
            run_bench(&format!("selectivity={sel}%"), sel, dict_size);
        }
        println!();
    }
}
