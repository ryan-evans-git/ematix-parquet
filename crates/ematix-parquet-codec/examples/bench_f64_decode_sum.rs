//! Q15 floor — "what does Polars do differently in f64 decode?" PV.M.0 saw
//! Polars f64-decode 4.3× faster (LZ4); PV.M.2 dismissed it on a flawed
//! our-dense≈our-masked compare. plain_sparse_decode_f64_into is a scalar
//! per-value `from_le_bytes + Vec::push` (push = vectorization barrier,
//! REV.14); Polars reduces over the dense PLAIN buffer with SIMD.
//!
//! Sources the REAL extprice values via read_column_f64 (strips def levels
//! correctly — the v1 hand-parse hit NaN), re-encodes to PLAIN little-endian
//! bytes (on LE that IS the wire form), then times decode+SUM single-thread
//! (parallelism removed) via:
//!   A ematix_sparse — plain_sparse_decode_f64_into (survivor Vec) + sum
//!   B fused_fromle  — masked sum via from_le_bytes, countable loop, NO Vec
//!   C zerocopy      — &[f64] view (8-aligned) + masked sum  (Polars-style)
//! in DENSE (all values) and 3.6%-MASKED (Q15 shipdate selectivity).
//!
//! Usage: cargo run --release -p ematix-parquet-codec --example bench_f64_decode_sum

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use ematix_parquet_codec::plain::plain_sparse_decode_f64_into;
use ematix_parquet_codec::read::read_column_f64;
use ematix_parquet_io::ParquetFile;

const ITERS: usize = 100;
const WARMUPS: usize = 10;
const PAGE: usize = 8192; // values per simulated page (cache-realistic chunking)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("TPCH_LINEITEM").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(
            "/Users/ryanevans/RustroverProjects/ematix-flow/examples/tpch/data/sf10/lineitem.parquet",
        )
    });
    let col_idx: usize = std::env::var("COL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let file = ParquetFile::open(&path)?;
    let md = file.metadata()?;

    // CORRECT dense values (read_column_f64 strips levels / handles pages).
    let mut vals: Vec<f64> = Vec::new();
    for rg in 0..md.row_groups.len() {
        vals.extend_from_slice(&read_column_f64(&file, rg, col_idx)?);
    }
    let n = vals.len();
    // PLAIN wire bytes = the f64 values as LE bytes (LE machine → identity).
    let bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, n * 8).to_vec() };
    let aligned = bytes.as_ptr() as usize % 8 == 0;
    // Deterministic ~3.6% mask (Q15 shipdate selectivity), 1 in 28.
    let mut mask = vec![0u8; n.div_ceil(8)];
    let mut i = 0;
    while i < n {
        mask[i / 8] |= 1 << (i % 8);
        i += 28;
    }
    let n_set: usize = (0..n)
        .filter(|&i| (mask[i / 8] >> (i % 8)) & 1 == 1)
        .count();
    println!("== f64 decode+sum mechanism (single-thread) ==  col {col_idx}");
    println!(
        "{:.1}M values, bytes 8-aligned={aligned}, mask {:.1}% ({} set)",
        n as f64 / 1e6,
        n_set as f64 / n as f64 * 100.0,
        n_set
    );

    let gbps = |s: f64| (n * 8) as f64 / s / 1e9;
    let bestof = |f: &mut dyn FnMut() -> f64| -> (f64, f64) {
        let mut best = f64::INFINITY;
        let mut cs = 0.0;
        for it in 0..(ITERS + WARMUPS) {
            let t0 = Instant::now();
            cs = f();
            let dt = t0.elapsed().as_secs_f64();
            if it >= WARMUPS {
                best = best.min(dt);
            }
            black_box(cs);
        }
        (best, cs)
    };

    for &(label, all_ones) in &[
        ("DENSE (all values)", true),
        ("3.6% MASKED (Q15 sel)", false),
    ] {
        println!("\n--- {label} ---");
        let mut scratch: Vec<f64> = Vec::with_capacity(PAGE);

        // A: ematix sparse decode (survivor Vec) + sum, per simulated page.
        let mut a = || {
            let mut s = 0.0f64;
            let mut off = 0usize;
            while off < n {
                let pn = PAGE.min(n - off);
                let pbytes = &bytes[off * 8..(off + pn) * 8];
                scratch.clear();
                if all_ones {
                    let ones = vec![0xFFu8; pn.div_ceil(8)];
                    plain_sparse_decode_f64_into(pbytes, pn, &ones, 0, &mut scratch).unwrap();
                } else {
                    // page-local mask slice (byte-aligned at multiples of 8)
                    let mbytes = &mask[off / 8..(off + pn).div_ceil(8)];
                    plain_sparse_decode_f64_into(pbytes, pn, mbytes, 0, &mut scratch).unwrap();
                }
                for &v in &scratch {
                    s += v;
                }
                off += pn;
            }
            s
        };
        let (ta, ca) = bestof(&mut a);

        // B: fused from_le_bytes masked sum, NO Vec.
        let mut b = || {
            let mut s = 0.0f64;
            if all_ones {
                for k in 0..n {
                    s += f64::from_le_bytes(bytes[k * 8..k * 8 + 8].try_into().unwrap());
                }
            } else {
                for k in 0..n {
                    if (mask[k / 8] >> (k % 8)) & 1 == 1 {
                        s += f64::from_le_bytes(bytes[k * 8..k * 8 + 8].try_into().unwrap());
                    }
                }
            }
            s
        };
        let (tb, cb) = bestof(&mut b);

        // C: zero-copy &[f64] view + masked sum (Polars mechanism).
        let mut c = || {
            let f: &[f64] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, n) };
            let mut s = 0.0f64;
            if all_ones {
                for &v in f {
                    s += v;
                }
            } else {
                for k in 0..n {
                    if (mask[k / 8] >> (k % 8)) & 1 == 1 {
                        s += f[k];
                    }
                }
            }
            s
        };
        let (tc, cc) = bestof(&mut c);

        println!(
            "{:<16}{:>9.3} ms{:>9.2} GB/s",
            "A ematix_sparse",
            ta * 1e3,
            gbps(ta)
        );
        println!(
            "{:<16}{:>9.3} ms{:>9.2} GB/s",
            "B fused_fromle",
            tb * 1e3,
            gbps(tb)
        );
        println!(
            "{:<16}{:>9.3} ms{:>9.2} GB/s",
            "C zerocopy",
            tc * 1e3,
            gbps(tc)
        );
        let ok = (ca - cb).abs() / ca.abs().max(1.0) < 1e-9
            && (ca - cc).abs() / ca.abs().max(1.0) < 1e-9;
        println!(
            "B/A={:.2}×  C/A={:.2}×  checksums_match={ok} (Σ={ca:.1})",
            ta / tb,
            ta / tc
        );
    }

    println!("\nQ15 is 3.6%-MASKED. If A (sparse-skip) already ≈/beats B,C there → our");
    println!("decode is NOT the gap in Q15's selective regime → residual is parallelism.");
    Ok(())
}
