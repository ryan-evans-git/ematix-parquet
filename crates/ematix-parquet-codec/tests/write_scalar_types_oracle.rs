//! Oracle tests for the Π.2b write path: i32, f64, bool, byte_array.
//!
//! For every type:
//!   1. Write a known Vec<T> through our writer.
//!   2. Read back via parquet-rs (the oracle).
//!   3. Read back via our own façade (the symmetric check).
//!   4. Assert equality on both.

use ematix_parquet_codec::read::{read_column_byte_array, read_column_f64, read_column_i32};
use ematix_parquet_codec::write::{
    write_bool_column_to_path, write_byte_array_column_to_path, write_f64_column_to_path,
    write_i32_column_to_path,
};
use ematix_parquet_io::ParquetFile;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

// ---- parquet-rs read helpers ----------------------------------------

fn pq_read_i32(path: &std::path::Path) -> Vec<i32> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int32ColumnReader(mut typed) = cr else {
        panic!("expected INT32");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<i32> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_f64(path: &std::path::Path) -> Vec<f64> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::DoubleColumnReader(mut typed) = cr else {
        panic!("expected DOUBLE");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<f64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_bool(path: &std::path::Path) -> Vec<bool> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::BoolColumnReader(mut typed) = cr else {
        panic!("expected BOOLEAN");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<bool> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_byte_array(path: &std::path::Path) -> Vec<Vec<u8>> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected BYTE_ARRAY");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

// ---- i32 ------------------------------------------------------------

#[test]
fn i32_writes_and_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.parquet");
    let values: Vec<i32> = (0..1000i32).map(|i| i - 500).collect();
    write_i32_column_to_path(&path, "v", &values).unwrap();
    assert_eq!(pq_read_i32(&path), values);
}

#[test]
fn i32_writes_and_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32_self.parquet");
    let values: Vec<i32> = vec![0, 1, -1, i32::MAX, i32::MIN, 42, -42];
    write_i32_column_to_path(&path, "v", &values).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i32(&file, 0, 0).unwrap(), values);
}

// ---- f64 ------------------------------------------------------------

#[test]
fn f64_writes_and_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.parquet");
    let values: Vec<f64> = vec![
        0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        std::f64::consts::E,
        f64::MAX,
        f64::MIN_POSITIVE,
        -f64::INFINITY,
    ];
    write_f64_column_to_path(&path, "v", &values).unwrap();
    let read = pq_read_f64(&path);
    // Use bit equality — IEEE doubles round-trip exactly when we just
    // copy the bytes through.
    for (a, b) in read.iter().zip(values.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn f64_writes_and_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64_self.parquet");
    let values: Vec<f64> = (0..256).map(|i| i as f64 * 0.123).collect();
    write_f64_column_to_path(&path, "v", &values).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    let read_back = read_column_f64(&file, 0, 0).unwrap();
    for (a, b) in read_back.iter().zip(values.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ---- bool -----------------------------------------------------------

#[test]
fn bool_writes_and_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bool.parquet");
    // Mix of alternating + runs + length-not-multiple-of-8 to exercise
    // the bit-packing + tail-padding path.
    let values: Vec<bool> = (0..73).map(|i| matches!(i % 5, 0 | 2 | 3)).collect();
    write_bool_column_to_path(&path, "v", &values).unwrap();
    assert_eq!(pq_read_bool(&path), values);
}

#[test]
fn bool_writes_and_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bool_self.parquet");
    let values: Vec<bool> = vec![true, false, true, true, false, false, true, false, true];
    write_bool_column_to_path(&path, "v", &values).unwrap();
    // Our façade doesn't have a read_column_bool yet — call the
    // decoder directly. (Adding the bool façade is independent work.)
    use ematix_parquet_codec::compression::decompress_snappy_into;
    use ematix_parquet_codec::plain::decode_plain_bool;
    use ematix_parquet_format::types::CompressionCodec;
    use ematix_parquet_io::PageWalker;

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let cm = md.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let chunk = file
        .read_range(start, cm.total_compressed_size as u64)
        .unwrap();
    let mut walker = PageWalker::new(&chunk);
    let (hdr, body) = walker.next_page().unwrap().unwrap();
    let n = hdr.data_page_header.unwrap().num_values as usize;
    assert_eq!(cm.codec, CompressionCodec::Uncompressed);
    let _ = decompress_snappy_into; // suppress unused-import warning
    let read_back = decode_plain_bool(body, n).unwrap();
    assert_eq!(read_back, values);
}

// ---- byte_array -----------------------------------------------------

#[test]
fn byte_array_writes_and_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba.parquet");
    let values: Vec<Vec<u8>> = vec![
        b"hello".to_vec(),
        b"".to_vec(),
        b"world!".to_vec(),
        vec![0, 1, 2, 3, 255],
        b"\xe2\x9c\x93 unicode-ish".to_vec(),
    ];
    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();
    assert_eq!(pq_read_byte_array(&path), values);
}

#[test]
fn byte_array_writes_and_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_self.parquet");
    let values: Vec<Vec<u8>> = (0..50)
        .map(|i| format!("row-{:03}", i).into_bytes())
        .collect();
    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_byte_array(&file, 0, 0).unwrap(), values);
}
