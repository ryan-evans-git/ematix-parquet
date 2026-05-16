//! Π.14c — adaptive dispatch threshold-tuning bench.
//!
//! For each target selectivity in a sweep, time three paths on
//! the same lineitem `l_shipdate` chunk:
//!
//!   bitmap_only          — `decode_rle_dictionary_predicate_bitmap`
//!                          page-by-page, bitmap output. The right
//!                          baseline for a bitmap-consuming caller
//!                          (filter chain, aggregator's COUNT).
//!   fused_then_gather    — same fused decode + a downstream gather
//!                          step that turns the bitmap into a values
//!                          vector. The right baseline for a
//!                          values-consuming caller.
//!   materialised         — `decode_rle_dictionary_into` + filter,
//!                          values output. Avoids the bitmap step.
//!   adaptive             — `run_adaptive_dict_chunk` with the
//!                          current `AdaptiveDictPredicate` defaults.
//!
//! The threshold question lives on the **values-consuming** side:
//! at what selectivity does `materialised` first beat
//! `fused_then_gather`?  That selectivity is what
//! `AdaptiveDictPredicate::DEFAULT_THRESHOLD` should approximate.
//!
//! Run:
//!   TPCH_DATA_DIR=/path cargo run --release \
//!       --example bench_adaptive_dispatch -p ematix-parquet-codec

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ematix_parquet_codec::adaptive::{
    run_adaptive_dict_chunk, AdaptiveDictPredicate, AdaptiveOutputKind, AdaptivePageInput, Dispatch,
};
use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::{
    decode_rle_dictionary_into, decode_rle_dictionary_predicate_bitmap, gather_dict_at_bitmap_into,
};
use ematix_parquet_codec::plain::decode_plain_i32;
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

const WARMUPS: usize = 5;
const ITERS: usize = 51;
const SELECTIVITIES: &[f32] = &[0.001, 0.01, 0.05, 0.10, 0.20, 0.50, 0.90];

fn data_dir() -> Option<PathBuf> {
    if let Ok(s) = std::env::var("TPCH_DATA_DIR") {
        let p = PathBuf::from(s);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .parent()?
        .parent()?
        .join("ematix-flow/examples/tpch/data/sf1");
    p.exists().then_some(p)
}

struct ChunkFixture {
    dict: Vec<i32>,
    bit_width: u8,
    /// Each page already decompressed; (num_values, body).
    pages: Vec<(usize, Vec<u8>)>,
    total_rows: usize,
}

fn load_l_shipdate(path: &Path) -> ChunkFixture {
    let file = ParquetFile::open(path).unwrap();
    let md = file.metadata().expect("metadata");
    // Column 10 = l_shipdate in TPC-H lineitem.
    let rg = &md.row_groups[0];
    let col = &rg.columns[10];
    let cm = col.meta_data.as_ref().expect("inline col meta");
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let bytes = file.read_range(start, length).expect("read chunk");
    let total_rows = cm.num_values as usize;

    let mut walker = PageWalker::new(&bytes);

    // Dict page first.
    let (dict_hdr, dict_body) = walker.next_page().unwrap().expect("dict page");
    assert!(dict_hdr.dictionary_page_header.is_some(), "expected dict");
    let dict_decompressed = decompress_snappy(dict_body).unwrap();
    let dict = decode_plain_i32(&dict_decompressed).unwrap();

    // Data pages, all decompressed up front.
    let mut pages: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut bit_width: u8 = 0;
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                assert!(!decompressed.is_empty(), "dict-encoded page empty");
                bit_width = decompressed[0];
                pages.push((n, decompressed));
            }
            other => panic!("bench expects dict-encoded data pages, got {other:?}"),
        }
    }

    ChunkFixture {
        dict,
        bit_width,
        pages,
        total_rows,
    }
}

/// Build a `dict_mask` that yields approximately `target` selectivity
/// against the given pages. Done deterministically: sort dict by value
/// and take the first N entries until the measured selectivity is
/// close to target. Returns `(mask, observed_selectivity)`.
fn dict_mask_at_target(fx: &ChunkFixture, target: f32) -> (Vec<u8>, f32) {
    // Run one fused pass over the chunk to count rows per dict index.
    // We approximate by counting indices via decode_rle_dictionary_into
    // with dict = (0..dict.len()) as identity, then tally.
    let id_dict: Vec<u32> = (0..fx.dict.len() as u32).collect();
    let mut idx_counts = vec![0usize; fx.dict.len()];
    let mut tmp: Vec<u32> = Vec::with_capacity(fx.total_rows);
    for (n, body) in &fx.pages {
        tmp.clear();
        decode_rle_dictionary_into(body, &id_dict, *n, &mut tmp).unwrap();
        for &idx in &tmp {
            idx_counts[idx as usize] += 1;
        }
    }

    // Sort dict entries by count (descending), greedily mark entries
    // until cumulative coverage ≈ target.
    let mut order: Vec<usize> = (0..fx.dict.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(idx_counts[i]));
    let target_rows = (fx.total_rows as f32 * target) as usize;
    let mut mask_indices: Vec<usize> = Vec::new();
    let mut acc: usize = 0;
    for &i in &order {
        if acc >= target_rows {
            break;
        }
        mask_indices.push(i);
        acc += idx_counts[i];
    }

    // Position-based mask: top-K most frequent dict entries until
    // selectivity ≈ target. We construct it directly rather than via
    // `build_dict_predicate_mask`, which is value-based.
    let mask_len = 1usize << fx.bit_width;
    let mut mask = vec![0u8; mask_len];
    for &i in &mask_indices {
        mask[i] = 1;
    }

    let observed = acc as f32 / fx.total_rows as f32;
    (mask, observed)
}

fn run_bitmap_only(fx: &ChunkFixture, dict_mask: &[u8]) -> usize {
    let mut bitmap: Vec<u8> = Vec::with_capacity(fx.total_rows / 8 + 1);
    for (n, body) in &fx.pages {
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut bitmap).unwrap();
    }
    bitmap.iter().map(|b| b.count_ones() as usize).sum()
}

/// Fused decode → per-page gather of surviving values. This is the
/// honest baseline for a values-consuming caller using the fused
/// kernel: the work that would happen in the late-mat consumer's
/// stack on top of `read_column_*_predicate_bitmap`.
fn run_fused_then_gather(fx: &ChunkFixture, dict_mask: &[u8]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut bm: Vec<u8> = Vec::with_capacity(fx.pages[0].0 / 8 + 1);
    for (n, body) in &fx.pages {
        bm.clear();
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut bm).unwrap();
        gather_dict_at_bitmap_into(body, *n, &bm, 0, &fx.dict, &mut out).unwrap();
    }
    out
}

fn run_static_materialised(fx: &ChunkFixture, dict_mask: &[u8]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut tmp: Vec<i32> = Vec::with_capacity(fx.pages[0].0);
    let mut bm: Vec<u8> = Vec::with_capacity(fx.pages[0].0 / 8 + 1);
    for (n, body) in &fx.pages {
        tmp.clear();
        decode_rle_dictionary_into(body, &fx.dict, *n, &mut tmp).unwrap();
        bm.clear();
        decode_rle_dictionary_predicate_bitmap(body, *n, dict_mask, &mut bm).unwrap();
        for row in 0..*n {
            let bit = (bm[row / 8] >> (row % 8)) & 1;
            if bit != 0 {
                out.push(tmp[row]);
            }
        }
    }
    out
}

fn run_adaptive(fx: &ChunkFixture, dict_mask: Vec<u8>) -> (Dispatch, usize) {
    let cfg = AdaptiveDictPredicate::new(dict_mask);
    let pages: Vec<AdaptivePageInput<'_>> = fx
        .pages
        .iter()
        .map(|(n, b)| AdaptivePageInput {
            body: b.as_slice(),
            num_values: *n,
        })
        .collect();
    let out = run_adaptive_dict_chunk::<i32>(&pages, &fx.dict, &cfg, None).unwrap();
    let count = match out.kind {
        AdaptiveOutputKind::Bitmap { set_bits, .. } => set_bits,
        AdaptiveOutputKind::Values(v) => v.len(),
    };
    (out.dispatch, count)
}

fn bench<R>(label: &str, mut f: impl FnMut() -> R) -> Duration {
    for _ in 0..WARMUPS {
        black_box(f());
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        black_box(f());
        times.push(start.elapsed());
    }
    times.sort();
    let med = times[ITERS / 2];
    println!(
        "    {label:<24} median {:>7.3} ms  min {:>7.3} ms",
        med.as_secs_f64() * 1000.0,
        times[0].as_secs_f64() * 1000.0,
    );
    med
}

fn main() {
    let Some(dir) = data_dir() else {
        eprintln!("TPC-H data not found; set TPCH_DATA_DIR");
        std::process::exit(1);
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("missing {}", path.display());
        std::process::exit(1);
    }
    // Pre-flight: ensure file readable.
    let mut probe_buf = [0u8; 4];
    File::open(&path)
        .unwrap()
        .read_exact(&mut probe_buf)
        .unwrap();
    assert_eq!(&probe_buf, b"PAR1", "not a parquet file");

    println!("== Π.14c adaptive-dispatch sweep ({WARMUPS} warmups + {ITERS} iters per cell) ==");
    println!("data: {}", path.display());

    let fx = load_l_shipdate(&path);
    println!(
        "fixture: {} rows, {} pages, dict_len={}, bit_width={}\n",
        fx.total_rows,
        fx.pages.len(),
        fx.dict.len(),
        fx.bit_width
    );

    println!(
        "{:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>13}",
        "target", "observed", "bitmap", "f+gather", "matr", "adapt", "adapt_chose"
    );
    println!("{}", "-".repeat(80));

    for &target in SELECTIVITIES {
        let (mask, observed) = dict_mask_at_target(&fx, target);

        // Sanity: same count across every path.
        let count_bitmap = run_bitmap_only(&fx, &mask);
        let count_fg = run_fused_then_gather(&fx, &mask).len();
        let count_matr = run_static_materialised(&fx, &mask).len();
        let (chose, count_adapt) = run_adaptive(&fx, mask.clone());
        assert_eq!(count_bitmap, count_fg, "count mismatch bitmap vs f+gather");
        assert_eq!(count_bitmap, count_matr, "count mismatch bitmap vs matr");
        assert_eq!(
            count_bitmap, count_adapt,
            "count mismatch bitmap vs adaptive"
        );

        let t_bm = bench("bitmap_only", || run_bitmap_only(&fx, &mask));
        let t_fg = bench("fused_then_gather", || run_fused_then_gather(&fx, &mask));
        let t_matr = bench("materialised", || run_static_materialised(&fx, &mask));
        let t_adapt = bench("adaptive", || run_adaptive(&fx, mask.clone()));

        println!(
            "{:>8.3}% {:>8.3}% {:>8.3} {:>8.3} {:>8.3} {:>8.3}  {:>12?}",
            target * 100.0,
            observed * 100.0,
            t_bm.as_secs_f64() * 1000.0,
            t_fg.as_secs_f64() * 1000.0,
            t_matr.as_secs_f64() * 1000.0,
            t_adapt.as_secs_f64() * 1000.0,
            chose,
        );
        println!();
    }

    println!(
        "Tuning rule:\n  For a values-consuming caller, pick `DEFAULT_THRESHOLD` ≈ the\n  observed selectivity where `matr` first becomes < `f+gather` in\n  the sweep above. At that selectivity, adaptive (with that\n  threshold) should match whichever values-output path is faster.\n  The `bitmap_only` column is the baseline for callers that consume\n  bitmaps directly (filter chain, COUNT aggregator) — for those,\n  fused always wins."
    );
}
