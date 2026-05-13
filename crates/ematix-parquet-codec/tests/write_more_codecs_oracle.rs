//! Oracle tests for GZIP / Brotli / LZ4_RAW round-trip.
//!
//! Each codec is exercised twice:
//!   - Ours writes → parquet-rs reads (the external oracle)
//!   - parquet-rs writes → ours reads (the read-side check)
//!
//! Bodies are sized so the codec actually compresses; tiny inputs
//! can produce output larger than the input on Brotli, which hides
//! header bookkeeping bugs.

use std::sync::Arc;

use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::write_i64_column_to_path_with_codec;
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

use parquet::basic::{Compression, GzipLevel};
use parquet::column::reader::ColumnReader;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn pq_read_i64(path: &std::path::Path) -> Vec<i64> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected INT64");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn parquet_rs_write_i64(path: &std::path::Path, values: &[i64], compression: Compression) {
    let schema = parse_message_type("message s { REQUIRED INT64 v; }").unwrap();
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, Arc::new(schema), Arc::new(props)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<parquet::data_type::Int64Type>()
        .write_batch(values, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

// ---- GZIP ----------------------------------------------------------

#[test]
fn gzip_ours_writes_parquet_rs_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gzip.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 64).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Gzip).unwrap();
    assert_eq!(pq_read_i64(&path), values);
}

#[test]
fn gzip_parquet_rs_writes_ours_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gzip_in.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| (i * 7) % 128).collect();
    parquet_rs_write_i64(&path, &values, Compression::GZIP(GzipLevel::default()));
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i64(&file, 0, 0).unwrap(), values);
}

// ---- Brotli --------------------------------------------------------

#[test]
fn brotli_ours_writes_parquet_rs_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brotli.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 64).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Brotli).unwrap();
    assert_eq!(pq_read_i64(&path), values);
}

#[test]
fn brotli_parquet_rs_writes_ours_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brotli_in.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| (i * 11) % 256).collect();
    parquet_rs_write_i64(
        &path,
        &values,
        Compression::BROTLI(parquet::basic::BrotliLevel::default()),
    );
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i64(&file, 0, 0).unwrap(), values);
}

// ---- LZ4_RAW -------------------------------------------------------

#[test]
fn lz4_raw_ours_writes_parquet_rs_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lz4raw.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 64).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Lz4Raw).unwrap();
    assert_eq!(pq_read_i64(&path), values);
}

#[test]
fn lz4_raw_parquet_rs_writes_ours_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lz4raw_in.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| (i * 13) % 256).collect();
    parquet_rs_write_i64(&path, &values, Compression::LZ4_RAW);
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i64(&file, 0, 0).unwrap(), values);
}
