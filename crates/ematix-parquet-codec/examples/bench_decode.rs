//! End-to-end decode bench: ours vs parquet-rs.
//!
//! Times one full column decode per iteration (open file → walk
//! pages → decompress → decode → materialize Vec). Both sides
//! include file-open cost so the comparison is apples to apples;
//! the OS page cache warms up during warmup runs and then both
//! sides hit cached data.
//!
//! Columns covered:
//!   l_orderkey   INT64   (dict + plain mix — most complex shape)
//!   l_shipdate   INT32   (mostly dict, the Q14 filter column)
//!   l_returnflag BYTE_ARRAY (3 distinct values, all dict-encoded)
//!
//! Usage:
//!   cargo run --release --example bench_decode

use std::fs::File;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, decode_rle_dictionary_into};
use ematix_parquet_codec::plain::{
    decode_plain_byte_array, decode_plain_byte_array_n, decode_plain_f64, decode_plain_i32,
    decode_plain_i64,
};
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

use polars::prelude::*;
use polars_io::prelude::*;

const WARMUPS: usize = 3;
const ITERS: usize = 12;

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

// -------- Ours --------------------------------------------------------------

fn ours_decode_i64(path: &Path, col_idx: usize) -> Vec<i64> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    let (_first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let dict: Vec<i64> = decode_plain_i64(&dict_decompressed).unwrap();

    let mut out: Vec<i64> = Vec::with_capacity(total);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                decode_rle_dictionary_into(&decompressed, &dict, n, &mut out).unwrap();
            }
            Encoding::Plain => {
                out.extend(decode_plain_i64(&decompressed).unwrap());
            }
            _ => panic!(),
        }
        if out.len() >= total {
            break;
        }
    }
    out
}

fn ours_decode_i32(path: &Path, col_idx: usize) -> Vec<i32> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    let (_first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let dict: Vec<i32> = decode_plain_i32(&dict_decompressed).unwrap();

    let mut out: Vec<i32> = Vec::with_capacity(total);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                decode_rle_dictionary_into(&decompressed, &dict, n, &mut out).unwrap();
            }
            Encoding::Plain => {
                out.extend(decode_plain_i32(&decompressed).unwrap());
            }
            _ => panic!(),
        }
        if out.len() >= total {
            break;
        }
    }
    out
}

#[allow(dead_code)]
fn ours_decode_f64(path: &Path, col_idx: usize) -> Vec<f64> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let dict: Vec<f64> = if first_hdr.dictionary_page_header.is_some() {
        decode_plain_f64(&dict_decompressed).unwrap()
    } else {
        Vec::new()
    };

    let mut out: Vec<f64> = Vec::with_capacity(total);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                decode_rle_dictionary_into(&decompressed, &dict, n, &mut out).unwrap();
            }
            Encoding::Plain => {
                out.extend(decode_plain_f64(&decompressed).unwrap());
            }
            _ => panic!(),
        }
        if out.len() >= total {
            break;
        }
    }
    out
}

fn ours_decode_byte_array(path: &Path, col_idx: usize) -> Vec<Vec<u8>> {
    let file = ParquetFile::open(path).unwrap();
    let (chunk, total) = read_chunk_bytes(&file, 0, col_idx);
    let mut walker = PageWalker::new(&chunk);

    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).unwrap();
    let dict: Vec<Vec<u8>> = if first_hdr.dictionary_page_header.is_some() {
        decode_plain_byte_array(&dict_decompressed)
            .unwrap()
            .into_iter()
            .map(|s| s.to_vec())
            .collect()
    } else {
        Vec::new()
    };

    let mut out: Vec<Vec<u8>> = Vec::with_capacity(total);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().unwrap();
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices = decode_rle_dictionary_indices(&decompressed, n).unwrap();
                for &idx in &indices {
                    out.push(dict[idx as usize].clone());
                }
            }
            Encoding::Plain => {
                let values = decode_plain_byte_array_n(&decompressed, n).unwrap();
                for v in values {
                    out.push(v.to_vec());
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

// -------- parquet-rs --------------------------------------------------------

fn pr_decode_i64(path: &Path, col_idx: usize) -> Vec<i64> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(col_idx).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::Int64ColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pr_decode_i32(path: &Path, col_idx: usize) -> Vec<i32> {
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

// -------- polars (eager ParquetReader from polars-io) -----------------------
//
// `ParquetReader::new(file).finish()` bypasses LazyFrame's plan-builder /
// optimizer overhead — pure decode + DataFrame construction. This is the
// closest apples-to-apples comparison with parquet-rs's
// `read_records`-into-Vec.

fn polars_decode_i64(path: &Path, col_name: &str) -> Vec<i64> {
    let file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(file)
        .with_columns(Some(vec![col_name.into()]))
        .finish()
        .unwrap();
    let s = df.column(col_name).unwrap().as_materialized_series();
    s.i64().unwrap().into_no_null_iter().collect()
}

fn polars_decode_i32(path: &Path, col_name: &str) -> Vec<i32> {
    let file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(file)
        .with_columns(Some(vec![col_name.into()]))
        .finish()
        .unwrap();
    let s = df.column(col_name).unwrap().as_materialized_series();
    let cast = s.cast(&DataType::Int32).unwrap();
    cast.i32().unwrap().into_no_null_iter().collect()
}

fn polars_decode_byte_array(path: &Path, col_name: &str) -> Vec<Vec<u8>> {
    let file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(file)
        .with_columns(Some(vec![col_name.into()]))
        .finish()
        .unwrap();
    let s = df.column(col_name).unwrap().as_materialized_series();
    s.str()
        .unwrap()
        .into_no_null_iter()
        .map(|s| s.as_bytes().to_vec())
        .collect()
}

fn pr_decode_byte_array(path: &Path, col_idx: usize) -> Vec<Vec<u8>> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(col_idx).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::ByteArrayColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

// -------- Bench driver ------------------------------------------------------

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
        "  {label:<22} median {:>7.2} ms  min {:>7.2} ms  max {:>7.2} ms",
        med.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
    (med, min, max)
}

fn compare(name: &str, ours_med: Duration, theirs_med: Duration) {
    let r = ours_med.as_secs_f64() / theirs_med.as_secs_f64();
    let symbol = if r < 1.0 { "✓" } else { "✗" };
    println!(
        "  {symbol} {name}: ours/theirs = {:.2}× ({})",
        r,
        if r < 1.0 {
            format!("we're {:.0}% faster", (1.0 - r) * 100.0)
        } else {
            format!("we're {:.0}% slower", (r - 1.0) * 100.0)
        }
    );
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

    println!(
        "== ematix-parquet vs parquet-rs vs polars ({WARMUPS} warmups + {ITERS} iters) =="
    );
    println!("data: {}\n", path.display());

    println!("l_orderkey  INT64  (dict + plain mix, 1,048,576 values)");
    let (o_med, _, _) = bench("ours", || ours_decode_i64(&path, 0));
    let (pr_med, _, _) = bench("parquet-rs", || pr_decode_i64(&path, 0));
    let (po_med, _, _) = bench("polars (eager)", || polars_decode_i64(&path, "l_orderkey"));
    compare("ours vs parquet-rs", o_med, pr_med);
    compare("ours vs polars    ", o_med, po_med);
    println!();

    println!("l_shipdate  INT32  (dict, 1,048,576 values)");
    let (o_med, _, _) = bench("ours", || ours_decode_i32(&path, 10));
    let (pr_med, _, _) = bench("parquet-rs", || pr_decode_i32(&path, 10));
    let (po_med, _, _) = bench("polars (eager)", || polars_decode_i32(&path, "l_shipdate"));
    compare("ours vs parquet-rs", o_med, pr_med);
    compare("ours vs polars    ", o_med, po_med);
    println!();

    println!("l_returnflag  BYTE_ARRAY  (3 distinct, all dict, 1,048,576 values)");
    let (o_med, _, _) = bench("ours", || ours_decode_byte_array(&path, 8));
    let (pr_med, _, _) = bench("parquet-rs", || pr_decode_byte_array(&path, 8));
    let (po_med, _, _) = bench("polars (eager)", || polars_decode_byte_array(&path, "l_returnflag"));
    compare("ours vs parquet-rs", o_med, pr_med);
    compare("ours vs polars    ", o_med, po_med);
}
