//! End-to-end real-file oracle for DELTA_BINARY_PACKED.
//!
//! 1. Write a tiny parquet file via parquet-rs with INT32 + INT64
//!    columns forced to DELTA_BINARY_PACKED encoding.
//! 2. Open it through our pipeline (`ParquetFile::open`, `PageWalker`,
//!    snappy, `decode_delta_i32`/`_i64`).
//! 3. Read the same file via parquet-rs's typed column reader.
//! 4. Assert value-by-value match.
//!
//! Validates the full stack on a real Parquet file (footer, page
//! headers, page bodies, encoding dispatch) — not just an in-memory
//! encoder roundtrip.

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::compression::decompress_snappy_into;
use ematix_parquet_codec::delta::{decode_delta_i32, decode_delta_i64};
use ematix_parquet_format::types::Encoding as EmEncoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::basic::{Compression, Encoding, Repetition, Type as PhysicalType};
use parquet::column::reader::ColumnReader;
use parquet::column::writer::ColumnWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn write_delta_file(path: &std::path::Path, ints: &[i32], longs: &[i64]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(
                    SchemaType::primitive_type_builder("col_i32", PhysicalType::INT32)
                        .with_repetition(Repetition::REQUIRED)
                        .build()
                        .unwrap(),
                ),
                Arc::new(
                    SchemaType::primitive_type_builder("col_i64", PhysicalType::INT64)
                        .with_repetition(Repetition::REQUIRED)
                        .build()
                        .unwrap(),
                ),
            ])
            .build()
            .unwrap(),
    );

    // Force DELTA + disable dict so every data page is DELTA-encoded.
    let props = Arc::new(
        WriterProperties::builder()
            .set_encoding(Encoding::DELTA_BINARY_PACKED)
            .set_dictionary_enabled(false)
            .set_compression(Compression::SNAPPY)
            .build(),
    );

    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();

    // Column 0: INT32
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int32ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(ints, None, None).unwrap();
    }
    col.close().unwrap();

    // Column 1: INT64
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(longs, None, None).unwrap();
    }
    col.close().unwrap();

    row_group.close().unwrap();
    writer.close().unwrap();
}

/// Walk pages of one column chunk and decode each DELTA-encoded data
/// page via our codec. Returns the concatenated values.
fn ours_decode_delta_i32_column(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> Vec<i32> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[rg_idx].columns[col_idx]
        .meta_data
        .as_ref()
        .unwrap();
    let start = cm.data_page_offset as u64;
    let length = cm.total_compressed_size as u64;
    let chunk = file.read_range(start, length).unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp: Vec<u8> = Vec::new();
    let mut out = Vec::new();
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        decompress_snappy_into(body, &mut decomp).unwrap();
        match dph.encoding {
            EmEncoding::DeltaBinaryPacked => {
                out.extend(decode_delta_i32(&decomp).unwrap());
            }
            other => panic!("expected DELTA encoding, got {other:?}"),
        }
        if out.len() >= cm.num_values as usize {
            break;
        }
    }
    out
}

fn ours_decode_delta_i64_column(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> Vec<i64> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[rg_idx].columns[col_idx]
        .meta_data
        .as_ref()
        .unwrap();
    let start = cm.data_page_offset as u64;
    let length = cm.total_compressed_size as u64;
    let chunk = file.read_range(start, length).unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp: Vec<u8> = Vec::new();
    let mut out = Vec::new();
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        decompress_snappy_into(body, &mut decomp).unwrap();
        match dph.encoding {
            EmEncoding::DeltaBinaryPacked => {
                out.extend(decode_delta_i64(&decomp).unwrap());
            }
            other => panic!("expected DELTA encoding, got {other:?}"),
        }
        if out.len() >= cm.num_values as usize {
            break;
        }
    }
    out
}

fn pr_read_i32(path: &std::path::Path) -> Vec<i32> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(0).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(0).unwrap() {
        ColumnReader::Int32ColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<i32> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pr_read_i64(path: &std::path::Path) -> Vec<i64> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(1).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(1).unwrap() {
        ColumnReader::Int64ColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

#[test]
fn delta_real_file_roundtrips_through_our_pipeline() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Realistic shapes: i32 monotonic, i64 large-with-small-deltas.
    let ints: Vec<i32> = (0..10_000).map(|i| 1_000_000 + i * 7).collect();
    let longs: Vec<i64> = (0..10_000)
        .map(|i| 1_700_000_000_000_000_000i64 + (i as i64) * 1_000_000)
        .collect();

    write_delta_file(&path, &ints, &longs);

    let file = ParquetFile::open(&path).expect("open");
    let ours_i32 = ours_decode_delta_i32_column(&file, 0, 0);
    let ours_i64 = ours_decode_delta_i64_column(&file, 0, 1);

    let theirs_i32 = pr_read_i32(&path);
    let theirs_i64 = pr_read_i64(&path);

    assert_eq!(ours_i32.len(), ints.len());
    assert_eq!(ours_i32, ints, "ours i32 vs original");
    assert_eq!(ours_i32, theirs_i32, "ours vs parquet-rs i32");
    assert_eq!(ours_i64.len(), longs.len());
    assert_eq!(ours_i64, longs, "ours i64 vs original");
    assert_eq!(ours_i64, theirs_i64, "ours vs parquet-rs i64");

    eprintln!(
        "PASS: {} i32 + {} i64 values decoded via our pipeline match parquet-rs byte-for-byte",
        ours_i32.len(),
        ours_i64.len()
    );
}
