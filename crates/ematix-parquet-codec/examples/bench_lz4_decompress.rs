//! Phase A.1 audit probe — measure raw `lz4_flex::block::decompress_into`
//! throughput on a representative TPC-H page body, single-thread.
//!
//! Reads RG 0 of `lineitem_lz4.parquet`, walks the data-page bodies
//! of l_extendedprice (col 5), and times decompress-only across 50
//! iterations. Reports best-of GB/s of uncompressed output.
//!
//! Usage:
//!   cargo run --release -p ematix-parquet-codec --example bench_lz4_decompress
//! Env:
//!   TPCH_LINEITEM_LZ4=<abs path>  default examples/tpch/data/sf10/lineitem_lz4.parquet

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use ematix_parquet_codec::compression::decompress_lz4_raw_into_sized;
use ematix_parquet_io::pages::PageWalker;
use ematix_parquet_io::ParquetFile;

const ITERS: usize = 200;
const WARMUPS: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("TPCH_LINEITEM_LZ4").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/Users/ryanevans/RustroverProjects/ematix-flow/examples/tpch/data/sf10/lineitem_lz4.parquet")
    });
    let file = ParquetFile::open(&path)?;
    let md = file.metadata()?;
    if md.row_groups.is_empty() {
        eprintln!("no row groups");
        std::process::exit(1);
    }

    // Pick col_idx 5 (l_extendedprice) — known LZ4_RAW + high uncompressed
    // size, representative of the Q06 hot column.
    let col_idx: usize = std::env::var("COL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let col_name = std::str::from_utf8(
        md.row_groups[0].columns[col_idx]
            .meta_data
            .as_ref()
            .unwrap()
            .path_in_schema[0],
    )
    .unwrap_or("?");
    println!("== LZ4_RAW decompress probe ==");
    println!("file: {}", path.display());
    println!("col {}: {}", col_idx, col_name);

    // Walk pages across all row groups, collecting (body_bytes, uncompressed_size)
    let mut pages: Vec<(Vec<u8>, usize)> = Vec::new();
    let mut total_uncomp: usize = 0;
    let mut total_comp: usize = 0;
    for rg in &md.row_groups {
        let cm = rg.columns[col_idx].meta_data.as_ref().unwrap();
        let offset = cm.dictionary_page_offset.unwrap_or(cm.data_page_offset) as u64;
        let length = cm.total_compressed_size as u64;
        let chunk = file.read_range(offset, length)?;
        let mut w = PageWalker::with_byte_limit(&chunk, length as usize);
        loop {
            match w.next_page() {
                Ok(Some((hdr, body))) => {
                    // Skip dictionary pages; we only care about LZ4-compressed
                    // data pages here. Dict pages may also be LZ4-compressed
                    // but they're a smaller share of total bytes.
                    if hdr.dictionary_page_header.is_some() {
                        continue;
                    }
                    let uncomp = hdr.uncompressed_page_size as usize;
                    let body_vec = body.to_vec();
                    total_comp += body_vec.len();
                    total_uncomp += uncomp;
                    pages.push((body_vec, uncomp));
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("page walk error: {e}");
                    break;
                }
            }
        }
    }
    println!(
        "collected {} pages, total_comp={:.1} MB total_uncomp={:.1} MB ratio={:.2}",
        pages.len(),
        total_comp as f64 / 1e6,
        total_uncomp as f64 / 1e6,
        total_comp as f64 / total_uncomp as f64,
    );

    // Pre-allocate one output buffer the size of the largest page, reused
    // across all pages within an iteration.
    let max_uncomp = pages.iter().map(|(_, u)| *u).max().unwrap_or(0);
    let mut out: Vec<u8> = Vec::with_capacity(max_uncomp);

    // Warmup
    for _ in 0..WARMUPS {
        for (body, uncomp) in &pages {
            out.clear();
            decompress_lz4_raw_into_sized(body, *uncomp, &mut out).unwrap();
            black_box(&out);
        }
    }

    // Measure
    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for (body, uncomp) in &pages {
            out.clear();
            decompress_lz4_raw_into_sized(body, *uncomp, &mut out).unwrap();
            black_box(&out);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }

    let gbps_out = total_uncomp as f64 / best / 1e9;
    let gbps_in = total_comp as f64 / best / 1e9;
    println!(
        "best: {:.3} ms  out={:.2} GB/s  in={:.2} GB/s  ({:.1} ns/byte_out)",
        best * 1e3,
        gbps_out,
        gbps_in,
        best * 1e9 / total_uncomp as f64,
    );
    println!("(single-thread; multi-thread scaling assumed near-linear up to memory BW)",);
    Ok(())
}
