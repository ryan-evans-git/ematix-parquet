//! Q15/Q06 floor de-risk — is the `snap` Snappy-decompress rate algorithm-bound
//! (→ a faster decoder is a real lever) or memory-bandwidth-bound (→ the
//! "Snappy floor" PV.M.5 assumed is physically real)?
//!
//! PV.M.3 proved ematix AND Polars both decompress via the pure-Rust `snap`
//! crate, so the Q15/Q06/Q07/Q08 residuals all bottom out at the SAME
//! decompress rate. Every prior phase treated that rate as fixed ("the Snappy
//! floor") WITHOUT testing whether `snap` itself is optimal — a classic
//! un-validated floor assumption. This measures it on real extprice pages.
//!
//! Three arms on real SF=10 canonical-Snappy l_extendedprice (col 5) pages:
//!   1. snap_into     — production `decompress_snappy_into` (incl. out.resize(_,0)
//!                      per-page zero-fill memset)
//!   2. snap_presized — `snap::raw::Decoder::decompress` into a slice resized
//!                      ONCE (no per-iter memset) — isolates the zero-fill tax
//!   3. memcpy        — copy `uncomp` bytes/page from a src buffer = pure
//!                      memory-move at output size = the bandwidth ceiling
//!
//! Reads:
//!   - snap_presized / memcpy ratio  → algorithmic headroom (how far below
//!     pure memory-move the decode runs; >~3× ⇒ algorithm-bound ⇒ libsnappy
//!     worth installing; ~1-2× ⇒ memory-bound ⇒ floor is physical, ACCEPT)
//!   - snap_into / snap_presized     → the zero-fill memset tax (a free lever
//!     via uninit out if it's material)
//!
//! Usage (gated behind the `libsnappy-bench` feature — links system libsnappy):
//!   cargo run --release -p ematix-parquet-codec --features libsnappy-bench \
//!     --example bench_snappy_headroom
//! Env: TPCH_LINEITEM=<abs path> (default sf10 canonical snappy), COL=<idx> (5)

use std::hint::black_box;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::time::Instant;

use ematix_parquet_codec::compression::decompress_snappy_into;
use ematix_parquet_io::pages::PageWalker;
use ematix_parquet_io::ParquetFile;

// Google libsnappy C API (brew install snappy). Build with
// RUSTFLAGS="-L /opt/homebrew/lib". snappy_uncompress writes into a
// caller buffer; returns 0 (SNAPPY_OK) on success.
#[link(name = "snappy")]
unsafe extern "C" {
    fn snappy_uncompress(
        compressed: *const c_char,
        compressed_length: usize,
        uncompressed: *mut c_char,
        uncompressed_length: *mut usize,
    ) -> c_int;
}

const ITERS: usize = 200;
const WARMUPS: usize = 10;

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
    let col_name = std::str::from_utf8(
        md.row_groups[0].columns[col_idx]
            .meta_data
            .as_ref()
            .unwrap()
            .path_in_schema[0],
    )
    .unwrap_or("?");
    println!("== Snappy decompress headroom probe ==");
    println!("file: {}", path.display());
    println!("col {col_idx}: {col_name}  (RGs={})", md.row_groups.len());

    // Collect real Snappy-compressed data-page bodies (skip dict pages).
    let mut pages: Vec<(Vec<u8>, usize)> = Vec::new();
    let (mut total_uncomp, mut total_comp) = (0usize, 0usize);
    for rg in &md.row_groups {
        let cm = rg.columns[col_idx].meta_data.as_ref().unwrap();
        let offset = cm.dictionary_page_offset.unwrap_or(cm.data_page_offset) as u64;
        let length = cm.total_compressed_size as u64;
        let chunk = file.read_range(offset, length)?;
        let mut w = PageWalker::with_byte_limit(&chunk, length as usize);
        while let Ok(Some((hdr, body))) = w.next_page() {
            if hdr.dictionary_page_header.is_some() {
                continue;
            }
            let uncomp = hdr.uncompressed_page_size as usize;
            total_comp += body.len();
            total_uncomp += uncomp;
            pages.push((body.to_vec(), uncomp));
        }
    }
    if pages.is_empty() {
        eprintln!(
            "no data pages for col {col_idx} — is this column Snappy? (LZ4 sibling won't match)"
        );
        std::process::exit(1);
    }
    println!(
        "collected {} pages, comp={:.1} MB uncomp={:.1} MB ratio={:.2}",
        pages.len(),
        total_comp as f64 / 1e6,
        total_uncomp as f64 / 1e6,
        total_comp as f64 / total_uncomp as f64,
    );

    let max_uncomp = pages.iter().map(|(_, u)| *u).max().unwrap_or(0);
    let gbps = |secs: f64| total_uncomp as f64 / secs / 1e9;
    let nspb = |secs: f64| secs * 1e9 / total_uncomp as f64;

    // ---- Arm 1: production decompress_snappy_into (includes resize zero-fill)
    let mut out: Vec<u8> = Vec::with_capacity(max_uncomp);
    for _ in 0..WARMUPS {
        for (body, _u) in &pages {
            decompress_snappy_into(body, &mut out).unwrap();
            black_box(&out);
        }
    }
    let mut best_into = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for (body, _u) in &pages {
            decompress_snappy_into(body, &mut out).unwrap();
            black_box(&out);
        }
        best_into = best_into.min(t0.elapsed().as_secs_f64());
    }

    // ---- Arm 2: snap decompress into a slice resized ONCE (no per-iter memset)
    let mut buf: Vec<u8> = vec![0u8; max_uncomp];
    let mut dec = snap::raw::Decoder::new();
    for _ in 0..WARMUPS {
        for (body, u) in &pages {
            let n = dec.decompress(body, &mut buf[..*u]).unwrap();
            black_box(n);
        }
    }
    let mut best_presized = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for (body, u) in &pages {
            let n = dec.decompress(body, &mut buf[..*u]).unwrap();
            black_box(n);
        }
        best_presized = best_presized.min(t0.elapsed().as_secs_f64());
    }

    // ---- Arm 2b: libsnappy (Google C, SIMD-tuned) into the same buffer.
    // out_len is in/out: init to available capacity, lib writes actual size.
    let mut ls_ok = true;
    for _ in 0..WARMUPS {
        for (body, u) in &pages {
            let mut out_len = buf.len();
            let st = unsafe {
                snappy_uncompress(
                    body.as_ptr() as *const c_char,
                    body.len(),
                    buf.as_mut_ptr() as *mut c_char,
                    &mut out_len,
                )
            };
            if st != 0 || out_len != *u {
                ls_ok = false;
            }
            black_box(out_len);
        }
    }
    let mut best_libsnappy = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for (body, _u) in &pages {
            let mut out_len = buf.len();
            let st = unsafe {
                snappy_uncompress(
                    body.as_ptr() as *const c_char,
                    body.len(),
                    buf.as_mut_ptr() as *mut c_char,
                    &mut out_len,
                )
            };
            black_box(st);
        }
        best_libsnappy = best_libsnappy.min(t0.elapsed().as_secs_f64());
    }

    // ---- Arm 3: memcpy uncomp bytes/page = pure memory-move ceiling
    let src: Vec<u8> = vec![7u8; max_uncomp];
    let mut dst: Vec<u8> = vec![0u8; max_uncomp];
    for _ in 0..WARMUPS {
        for (_b, u) in &pages {
            dst[..*u].copy_from_slice(&src[..*u]);
            black_box(dst[0]);
        }
    }
    let mut best_memcpy = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for (_b, u) in &pages {
            dst[..*u].copy_from_slice(&src[..*u]);
            black_box(dst[0]);
        }
        best_memcpy = best_memcpy.min(t0.elapsed().as_secs_f64());
    }

    println!(
        "\n{:<14} {:>9} {:>10} {:>12}",
        "arm", "ms", "GB/s_out", "ns/byte_out"
    );
    println!(
        "{:<14} {:>9.3} {:>10.2} {:>12.3}",
        "snap_into",
        best_into * 1e3,
        gbps(best_into),
        nspb(best_into)
    );
    println!(
        "{:<14} {:>9.3} {:>10.2} {:>12.3}",
        "snap_presized",
        best_presized * 1e3,
        gbps(best_presized),
        nspb(best_presized)
    );
    println!(
        "{:<14} {:>9.3} {:>10.2} {:>12.3}  ok={}",
        "libsnappy",
        best_libsnappy * 1e3,
        gbps(best_libsnappy),
        nspb(best_libsnappy),
        ls_ok
    );
    println!(
        "{:<14} {:>9.3} {:>10.2} {:>12.3}",
        "memcpy",
        best_memcpy * 1e3,
        gbps(best_memcpy),
        nspb(best_memcpy)
    );

    let vs_memcpy = best_presized / best_memcpy;
    let ls_speedup = (best_presized - best_libsnappy) / best_presized * 100.0;
    let memset_tax = (best_into - best_presized) / best_presized * 100.0;
    println!("\nsnap_presized / memcpy = {vs_memcpy:.1}× (rules OUT memory-bound: snap is compute-bound)");
    println!(
        "libsnappy vs snap_presized = {ls_speedup:+.1}% ({:.2}× snap rate)",
        best_presized / best_libsnappy
    );
    println!("zero-fill memset tax (snap_into vs snap_presized) = {memset_tax:+.1}%");
    println!(
        "VERDICT: {}",
        if !ls_ok {
            "libsnappy correctness mismatch — investigate before trusting the number."
        } else if ls_speedup >= 15.0 {
            "GO — libsnappy is materially faster than the `snap` crate. Swapping ematix-parquet's Snappy decode to a libsnappy FFI is a generalizable lever across EVERY Snappy-bound query (Q06/Q07/Q08/Q15) and beats Polars (stuck on `snap`). Next: wire it behind a feature, flow-side bench Q06+Q15."
        } else if ls_speedup >= 5.0 {
            "MARGINAL — libsnappy modestly faster; weigh the C++ build-dep cost vs the small generalized win. Re-measure at full-scan / multi-thread before committing."
        } else {
            "snap ≈ libsnappy — the `snap` crate is already at the Snappy ALGORITHMIC limit for this data (the 46× vs memcpy is the irreducible tag-parse + scattered back-reference cost, not a snap deficiency). Decompress is a TRUE floor (measured both ways). Pivot to parallel-efficiency (PV.M.5 work-steal, ~4ms, multi-session) or writer-side codec."
        }
    );
    Ok(())
}
