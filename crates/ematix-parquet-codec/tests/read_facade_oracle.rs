//! High-level facade oracle. Validates that `read_column_*(file, rg, col)`
//! returns the same values as parquet-rs on real TPC-H SF=1 lineitem.
//!
//! The point of these tests isn't to find decoder bugs — the existing
//! `lineitem_*_oracle` tests already cover the low-level path. These
//! tests check that the facade *plumbs* the low-level decoders correctly:
//! that it picks the right encoding per page, decompresses with a
//! reused buffer, and concatenates pages into one column-shaped Vec.

use std::path::PathBuf;

use ematix_parquet_codec::read::{
    read_column_byte_array, read_column_f64, read_column_i32, read_column_i64,
};
use ematix_parquet_io::ParquetFile;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

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

fn lineitem_path() -> Option<PathBuf> {
    let p = data_dir()?.join("lineitem.parquet");
    p.exists().then_some(p)
}

fn col_idx_by_name(name: &str) -> usize {
    // lineitem schema, in TPC-H standard order
    match name {
        "l_orderkey" => 0,
        "l_partkey" => 1,
        "l_suppkey" => 2,
        "l_linenumber" => 3,
        "l_quantity" => 4,
        "l_extendedprice" => 5,
        "l_discount" => 6,
        "l_tax" => 7,
        "l_returnflag" => 8,
        "l_linestatus" => 9,
        "l_shipdate" => 10,
        "l_commitdate" => 11,
        "l_receiptdate" => 12,
        "l_shipinstruct" => 13,
        "l_shipmode" => 14,
        "l_comment" => 15,
        _ => panic!("unknown lineitem column: {name}"),
    }
}

fn parquet_rs_read_i64(path: &PathBuf, rg: usize, col: usize) -> Vec<i64> {
    let f = std::fs::File::open(path).expect("open");
    let r = SerializedFileReader::new(f).expect("reader");
    let rg_reader = r.get_row_group(rg).expect("rg");
    let cr = rg_reader.get_column_reader(col).expect("col reader");
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected INT64 column at {col}");
    };
    let total = rg_reader.metadata().column(col).num_values() as usize;
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed
        .read_records(total, None, None, &mut out)
        .expect("read_records");
    out
}

fn parquet_rs_read_i32(path: &PathBuf, rg: usize, col: usize) -> Vec<i32> {
    let f = std::fs::File::open(path).expect("open");
    let r = SerializedFileReader::new(f).expect("reader");
    let rg_reader = r.get_row_group(rg).expect("rg");
    let cr = rg_reader.get_column_reader(col).expect("col reader");
    let ColumnReader::Int32ColumnReader(mut typed) = cr else {
        panic!("expected INT32 column at {col}");
    };
    let total = rg_reader.metadata().column(col).num_values() as usize;
    let mut out: Vec<i32> = Vec::with_capacity(total);
    typed
        .read_records(total, None, None, &mut out)
        .expect("read_records");
    out
}

fn parquet_rs_read_f64(path: &PathBuf, rg: usize, col: usize) -> Vec<f64> {
    let f = std::fs::File::open(path).expect("open");
    let r = SerializedFileReader::new(f).expect("reader");
    let rg_reader = r.get_row_group(rg).expect("rg");
    let cr = rg_reader.get_column_reader(col).expect("col reader");
    let ColumnReader::DoubleColumnReader(mut typed) = cr else {
        panic!("expected DOUBLE column at {col}");
    };
    let total = rg_reader.metadata().column(col).num_values() as usize;
    let mut out: Vec<f64> = Vec::with_capacity(total);
    typed
        .read_records(total, None, None, &mut out)
        .expect("read_records");
    out
}

fn parquet_rs_read_byte_array(path: &PathBuf, rg: usize, col: usize) -> Vec<Vec<u8>> {
    let f = std::fs::File::open(path).expect("open");
    let r = SerializedFileReader::new(f).expect("reader");
    let rg_reader = r.get_row_group(rg).expect("rg");
    let cr = rg_reader.get_column_reader(col).expect("col reader");
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected BYTE_ARRAY column at {col}");
    };
    let total = rg_reader.metadata().column(col).num_values() as usize;
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed
        .read_records(total, None, None, &mut out)
        .expect("read_records");
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

#[test]
fn facade_i64_matches_parquet_rs_orderkey() {
    let Some(path) = lineitem_path() else {
        eprintln!("SKIP: TPC-H lineitem not found");
        return;
    };
    let file = ParquetFile::open(&path).expect("open");
    let col = col_idx_by_name("l_orderkey");

    let ours = read_column_i64(&file, 0, col).expect("facade");
    let oracle = parquet_rs_read_i64(&path, 0, col);

    assert_eq!(ours.len(), oracle.len(), "row count mismatch");
    assert_eq!(ours, oracle, "values mismatch");
}

#[test]
fn facade_i32_matches_parquet_rs_shipdate() {
    let Some(path) = lineitem_path() else {
        eprintln!("SKIP: TPC-H lineitem not found");
        return;
    };
    let file = ParquetFile::open(&path).expect("open");
    let col = col_idx_by_name("l_shipdate");

    let ours = read_column_i32(&file, 0, col).expect("facade");
    let oracle = parquet_rs_read_i32(&path, 0, col);

    assert_eq!(ours.len(), oracle.len());
    assert_eq!(ours, oracle);
}

#[test]
fn facade_f64_matches_parquet_rs_extendedprice() {
    let Some(path) = lineitem_path() else {
        eprintln!("SKIP: TPC-H lineitem not found");
        return;
    };
    let file = ParquetFile::open(&path).expect("open");
    let col = col_idx_by_name("l_extendedprice");

    let ours = read_column_f64(&file, 0, col).expect("facade");
    let oracle = parquet_rs_read_f64(&path, 0, col);

    assert_eq!(ours.len(), oracle.len());
    // Exact equality is fine — DOUBLE PLAIN is bit-for-bit and our path
    // does no arithmetic.
    assert_eq!(ours, oracle);
}

#[test]
fn facade_byte_array_matches_parquet_rs_returnflag() {
    let Some(path) = lineitem_path() else {
        eprintln!("SKIP: TPC-H lineitem not found");
        return;
    };
    let file = ParquetFile::open(&path).expect("open");
    let col = col_idx_by_name("l_returnflag");

    let ours = read_column_byte_array(&file, 0, col).expect("facade");
    let oracle = parquet_rs_read_byte_array(&path, 0, col);

    assert_eq!(ours.len(), oracle.len());
    assert_eq!(ours, oracle);
}
