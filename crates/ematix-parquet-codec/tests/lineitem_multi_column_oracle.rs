//! Multi-column oracle: exercise the new PLAIN decoders (Int32,
//! Float64) end-to-end against real lineitem columns. l_partkey is
//! INT32 in every TPC-H writer we've seen. For Float64 we probe the
//! file's schema to find a DOUBLE column; if the writer encoded
//! decimals as FIXED_LEN_BYTE_ARRAY (common), we skip the Float64
//! sub-test with a log line rather than fail.

use std::fs::File;
use std::path::PathBuf;

use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, lookup_dict};
use ematix_parquet_codec::plain::{decode_plain_f64, decode_plain_i32};
use ematix_parquet_format::types::{Encoding, ParquetType};
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::basic::Type as PrType;
use parquet::column::reader::ColumnReader;
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

/// Read the entire column-chunk byte range for one (rg, col).
fn read_chunk_bytes(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> (Vec<u8>, i64) {
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
    (bytes, cm.num_values)
}

fn decode_int32_column(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> Vec<i32> {
    let (chunk_bytes, total_values) = read_chunk_bytes(file, rg_idx, col_idx);
    let total_values = total_values as usize;
    let mut walker = PageWalker::new(&chunk_bytes);

    // Dict page first (PLAIN i32).
    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).expect("dict snappy");
    let dict: Vec<i32> = if first_hdr.dictionary_page_header.is_some() {
        decode_plain_i32(&dict_decompressed).expect("dict PLAIN i32")
    } else {
        // No dict page — treat first page as a normal data page below.
        // For now we require columns with dict pages (lineitem's int32
        // columns always have them).
        panic!("expected dictionary page first for int32 column")
    };

    let mut out: Vec<i32> = Vec::with_capacity(total_values);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).expect("data snappy");
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices = decode_rle_dictionary_indices(&decompressed, n).expect("indices");
                let values = lookup_dict(&dict, &indices).expect("dict lookup");
                out.extend(values);
            }
            Encoding::Plain => {
                let values = decode_plain_i32(&decompressed).expect("plain i32");
                out.extend(values);
            }
            other => panic!("unhandled int32 data page encoding {other:?}"),
        }
        if out.len() >= total_values {
            break;
        }
    }
    out
}

fn decode_float64_column(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> Vec<f64> {
    let (chunk_bytes, total_values) = read_chunk_bytes(file, rg_idx, col_idx);
    let total_values = total_values as usize;
    let mut walker = PageWalker::new(&chunk_bytes);

    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).expect("dict snappy");
    let dict: Vec<f64> = if first_hdr.dictionary_page_header.is_some() {
        decode_plain_f64(&dict_decompressed).expect("dict PLAIN f64")
    } else {
        // First page is itself a data page (no dict). Process it then
        // continue.
        let dph = first_hdr.data_page_header.as_ref().expect("v1 data page");
        assert_eq!(
            dph.encoding,
            Encoding::Plain,
            "non-dict-led f64 column must use PLAIN"
        );
        return decode_plain_f64(&dict_decompressed).expect("plain f64");
    };

    let mut out: Vec<f64> = Vec::with_capacity(total_values);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).expect("data snappy");
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices = decode_rle_dictionary_indices(&decompressed, n).expect("indices");
                let values = lookup_dict(&dict, &indices).expect("dict lookup");
                out.extend(values);
            }
            Encoding::Plain => {
                let values = decode_plain_f64(&decompressed).expect("plain f64");
                out.extend(values);
            }
            other => panic!("unhandled f64 data page encoding {other:?}"),
        }
        if out.len() >= total_values {
            break;
        }
    }
    out
}

fn parquet_rs_read_i32(path: &PathBuf, rg_idx: usize, col_idx: usize, n: usize) -> Vec<i32> {
    let pr_reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(rg_idx).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::Int32ColumnReader(t) => t,
        _ => panic!("expected Int32ColumnReader"),
    };
    let mut out: Vec<i32> = Vec::with_capacity(n);
    typed.read_records(n, None, None, &mut out).unwrap();
    out
}

fn parquet_rs_read_f64(path: &PathBuf, rg_idx: usize, col_idx: usize, n: usize) -> Vec<f64> {
    let pr_reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(rg_idx).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::DoubleColumnReader(t) => t,
        _ => panic!("expected DoubleColumnReader"),
    };
    let mut out: Vec<f64> = Vec::with_capacity(n);
    typed.read_records(n, None, None, &mut out).unwrap();
    out
}

#[test]
fn lineitem_rg0_l_shipdate_int32_matches_parquet_rs() {
    // l_shipdate is INT32 (logical type Date) — col index 10.
    // Also the column Q14's filter operates on, so a correct decoder
    // here is on the critical path for that benchmark.
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open");
    let ours = decode_int32_column(&file, 0, 10);
    let theirs = parquet_rs_read_i32(&path, 0, 10, ours.len());
    assert_eq!(ours.len(), theirs.len());
    assert_eq!(ours, theirs);
    // l_shipdate values are Date32 days-since-epoch. dbgen ranges
    // approximately [1992-01-02, 1998-12-01] = [8036, 10561].
    let min = *ours.iter().min().unwrap();
    let max = *ours.iter().max().unwrap();
    eprintln!(
        "PASS l_shipdate: {} i32 values match parquet-rs \
         (range [{min}, {max}], first 5: {:?})",
        ours.len(),
        &ours[..5]
    );
    assert!(
        min >= 8000 && max <= 10600,
        "values outside expected dbgen range"
    );
}

#[test]
fn lineitem_first_float64_column_matches_parquet_rs() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    // Probe schema for the first DOUBLE column.
    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let descr = pr_reader.metadata().file_metadata().schema_descr();
    let mut f64_col: Option<usize> = None;
    for i in 0..descr.num_columns() {
        if descr.column(i).physical_type() == PrType::DOUBLE {
            f64_col = Some(i);
            break;
        }
    }
    let Some(col_idx) = f64_col else {
        eprintln!(
            "SKIP: lineitem has no DOUBLE physical column \
             (decimals stored as FIXED_LEN_BYTE_ARRAY in this writer's output)"
        );
        return;
    };

    let column_name = descr.column(col_idx).name().to_string();
    eprintln!("Float64 column: {column_name} (col index {col_idx})");

    let file = ParquetFile::open(&path).expect("open");
    let ours = decode_float64_column(&file, 0, col_idx);
    let theirs = parquet_rs_read_f64(&path, 0, col_idx, ours.len());
    assert_eq!(ours.len(), theirs.len());
    // f64 equality is bit-precise; PLAIN decode preserves the exact
    // bit pattern, so this should always match without epsilon tolerance.
    let mismatches: Vec<(usize, f64, f64)> = ours
        .iter()
        .zip(theirs.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, (a, b))| (i, *a, *b))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} value mismatches, first: {:?}",
        mismatches.len(),
        mismatches.first()
    );
    eprintln!(
        "PASS {column_name}: {} f64 values match parquet-rs (first 5: {:?})",
        ours.len(),
        &ours[..5]
    );
}

/// Optional sanity check: the *type* of the column we identify with our
/// format crate matches what parquet-rs sees.
#[test]
fn lineitem_physical_types_agree_with_parquet_rs() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open");
    let md = file.metadata().expect("metadata");

    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let descr = pr_reader.metadata().file_metadata().schema_descr();

    for col_idx in 0..descr.num_columns() {
        let theirs = descr.column(col_idx).physical_type();
        // Find the matching schema element in our flat list. parquet-rs
        // column ordering matches a depth-first walk; for flat schemas
        // (lineitem has no nesting), the col_idx-th non-root schema
        // element corresponds to leaf col_idx.
        let ours = md.schema[col_idx + 1].column_type.unwrap();
        let expect = match theirs {
            PrType::BOOLEAN => ParquetType::Boolean,
            PrType::INT32 => ParquetType::Int32,
            PrType::INT64 => ParquetType::Int64,
            PrType::INT96 => ParquetType::Int96,
            PrType::FLOAT => ParquetType::Float,
            PrType::DOUBLE => ParquetType::Double,
            PrType::BYTE_ARRAY => ParquetType::ByteArray,
            PrType::FIXED_LEN_BYTE_ARRAY => ParquetType::FixedLenByteArray,
        };
        assert_eq!(
            ours, expect,
            "col {col_idx}: ours={ours:?}, theirs={theirs:?}"
        );
    }
}
