//! Q14-shape filter bench: l_shipdate ∈ [1995-09-01, 1995-10-01).
//!
//! Compares three implementations on lineitem row group 0:
//!
//!   late-mat:   build DictColumnChunk; predicate evaluated against
//!               the ~2,555-entry dict (once), then per-row index
//!               lookup. No value materialization in count path.
//!   eager:      decode chunk into Vec<i32>, then scan-filter.
//!   parquet-rs: typed Int32ColumnReader read_records, then scan.
//!
//! Q14's filter is `l_shipdate ∈ [DATE '1995-09-01', DATE '1995-10-01')`.
//! Date32 days-since-epoch: 1995-09-01 = 9374, 1995-10-01 = 9404.
//! Selectivity: 30 / (7 × 365) ≈ 1.17% of rows pass.
//!
//! Usage:
//!   cargo run --release --example bench_late_mat

use std::fs::File;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ematix_parquet_codec::column::{DictColumnChunk, Segment};
use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, decode_rle_dictionary_into};
use ematix_parquet_codec::plain::{decode_plain_i32, decode_plain_i32_n};
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::column::reader::ColumnReader;
use parquet::file::reader::{FileReader, SerializedFileReader};

// Q14 filter window in Date32 days-since-epoch.
const LO: i32 = 9374;
const HI: i32 = 9404;

const WARMUPS: usize = 3;
const ITERS: usize = 15;

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

fn read_chunk_bytes(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> (Vec<u8>, usize) {
    let md = file.metadata().expect("metadata");
    let rg = &md.row_groups[rg_idx];
    let col = &rg.columns[col_idx];
    let cm = col.meta_data.as_ref().expect("inline col meta");
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let bytes = file.read_range(start, length).expect("read chunk");
    (bytes, cm.num_values as usize)
}

/// Build a `DictColumnChunk<i32>` from a parquet column chunk. The
/// codec crate doesn't depend on io, so this composition lives in
/// the example for now; a future ematix-parquet top-level crate
/// would expose it.
fn build_dict_column_chunk_i32(path: &Path, col_idx: usize) -> DictColumnChunk<i32> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let dict: Arc<Vec<i32>> = if first_hdr.dictionary_page_header.is_some() {
        Arc::new(decode_plain_i32(&dict_decompressed).unwrap())
    } else {
        panic!("expected dictionary page first for l_shipdate")
    };

    let mut segments: Vec<Segment<i32>> = Vec::new();
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices = decode_rle_dictionary_indices(&decompressed, n).unwrap();
                segments.push(Segment::DictIndices(indices));
            }
            Encoding::Plain => {
                segments.push(Segment::Plain(
                    decode_plain_i32_n(&decompressed, n).unwrap(),
                ));
            }
            _ => panic!(),
        }
    }
    DictColumnChunk::new(Some(dict), segments, total)
}

/// Stream-fused chunk filter for INT32 columns: walks pages,
/// decompresses, and writes `dict_mask[idx]` (or `predicate(plain)`)
/// directly into a `Vec<bool>` bitmap. No intermediate `Vec<u32>`
/// indices, no `DictColumnChunk` materialization.
///
/// This is the "Phase 4" target — Q14's filter without the chunk-
/// build cost.
fn filter_dict_chunk_i32_into_bitmap(
    path: &Path,
    col_idx: usize,
    predicate: impl Fn(i32) -> bool,
) -> Vec<bool> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    // First page: dict.
    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let (dict_mask, _has_dict): (Vec<bool>, bool) = if first_hdr.dictionary_page_header.is_some() {
        let dict = decode_plain_i32(&dict_decompressed).unwrap();
        (dict.iter().copied().map(&predicate).collect(), true)
    } else {
        (Vec::new(), false)
    };

    let mut out: Vec<bool> = Vec::with_capacity(total);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                // decode_rle_dictionary_into handles the RLE/bit-
                // packed framing AND the dict lookup. By passing
                // `dict_mask: &[bool]` as the dict, the output is
                // dict_mask[idx] per row — i.e. the filter bitmap
                // for this page. No Vec<u32> indices materialized.
                decode_rle_dictionary_into(&decompressed, &dict_mask, n, &mut out).unwrap();
            }
            Encoding::Plain => {
                for chunk in decompressed.chunks_exact(4).take(n) {
                    let v = i32::from_le_bytes(chunk.try_into().unwrap());
                    out.push(predicate(v));
                }
            }
            _ => panic!(),
        }
        if out.len() >= total {
            break;
        }
    }
    out
}

fn parquet_rs_decode_i32(path: &Path, col_idx: usize) -> Vec<i32> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(col_idx).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::Int32ColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<i32> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn bench<R>(label: &str, mut f: impl FnMut() -> R) -> (Duration, Duration, Duration) {
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
    let min = times[0];
    let max = times[ITERS - 1];
    println!(
        "  {label:<48} median {:>7.2} ms  min {:>7.2} ms  max {:>7.2} ms",
        med.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
    (med, min, max)
}

fn pretty_ratio(label: &str, ours: Duration, ref_: Duration) {
    let r = ours.as_secs_f64() / ref_.as_secs_f64();
    if r < 1.0 {
        println!("  ✓ {label}: {:.2}× (we're {:.0}% faster)", r, (1.0 - r) * 100.0);
    } else {
        println!("  ✗ {label}: {:.2}× (we're {:.0}% slower)", r, (r - 1.0) * 100.0);
    }
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

    println!("== Q14-shape filter on l_shipdate ({WARMUPS} warmups + {ITERS} iters) ==");
    println!("data: {}", path.display());
    println!("filter: l_shipdate ∈ [{LO}, {HI}) i.e. [1995-09-01, 1995-10-01)\n");

    // Pre-build the chunk once outside the timing loop (parquet-rs
    // does the equivalent work per call, so this is a generous
    // assumption for late-mat — but matches a real query engine
    // that reads the column chunk once and runs many ops on it).
    let prebuilt = build_dict_column_chunk_i32(&path, 10);
    let prebuilt_count_pre = prebuilt.count_matching(|d| d >= LO && d < HI);
    println!(
        "(sanity check: {prebuilt_count_pre} of {} rows pass filter, selectivity {:.2}%)\n",
        prebuilt.num_values,
        100.0 * prebuilt_count_pre as f64 / prebuilt.num_values as f64
    );

    println!("Phase 1: count rows matching filter (no materialization)");
    let (lm_count, _, _) = bench("late-mat (count_matching on pre-built chunk)", || {
        prebuilt.count_matching(|d| d >= LO && d < HI)
    });
    let (eg_count, _, _) = bench("eager (collect + filter().count() on chunk)", || {
        prebuilt
            .collect()
            .into_iter()
            .filter(|&d| d >= LO && d < HI)
            .count()
    });
    let (pr_count, _, _) = bench("parquet-rs (typed read + filter().count())", || {
        parquet_rs_decode_i32(&path, 10)
            .into_iter()
            .filter(|&d| d >= LO && d < HI)
            .count()
    });
    pretty_ratio("late-mat vs eager", lm_count, eg_count);
    pretty_ratio("late-mat vs parquet-rs", lm_count, pr_count);
    println!();

    println!("Phase 2: materialize the matching rows (filter then keep matches)");
    let (lm_gather, _, _) = bench("late-mat (filter + gather)", || {
        let mask = prebuilt.filter(|d| d >= LO && d < HI);
        prebuilt.gather(&mask)
    });
    let (eg_gather, _, _) = bench("eager (collect + filter().collect())", || {
        prebuilt
            .collect()
            .into_iter()
            .filter(|&d| d >= LO && d < HI)
            .collect::<Vec<i32>>()
    });
    let (pr_gather, _, _) = bench("parquet-rs (typed read + filter().collect())", || {
        parquet_rs_decode_i32(&path, 10)
            .into_iter()
            .filter(|&d| d >= LO && d < HI)
            .collect::<Vec<i32>>()
    });
    pretty_ratio("late-mat vs eager", lm_gather, eg_gather);
    pretty_ratio("late-mat vs parquet-rs", lm_gather, pr_gather);
    println!();

    println!("Phase 3: end-to-end (open + build chunk + filter + count)");
    let (lm_full, _, _) = bench("late-mat (open + build + count_matching)", || {
        let c = build_dict_column_chunk_i32(&path, 10);
        c.count_matching(|d| d >= LO && d < HI)
    });
    let (pr_full, _, _) = bench("parquet-rs (open + read + filter().count())", || {
        parquet_rs_decode_i32(&path, 10)
            .into_iter()
            .filter(|&d| d >= LO && d < HI)
            .count()
    });
    pretty_ratio("end-to-end late-mat vs parquet-rs", lm_full, pr_full);
    println!();

    println!("Phase 4: stream-fused — open + page-walk + dict-mask → bitmap");
    println!("(no DictColumnChunk, no Vec<u32> indices, direct to Vec<bool>)");
    let (sf_count, _, _) = bench("stream-fused (bitmap + count)", || {
        let bitmap = filter_dict_chunk_i32_into_bitmap(&path, 10, |d| d >= LO && d < HI);
        bitmap.iter().filter(|&&b| b).count()
    });
    pretty_ratio("end-to-end stream-fused vs parquet-rs", sf_count, pr_full);
}
