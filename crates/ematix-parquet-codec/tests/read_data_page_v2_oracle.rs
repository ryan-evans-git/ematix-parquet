//! Oracle: parquet-rs writes DataPageV2 → our high-level façade
//! reads values back correctly.
//!
//! Π.6a contract: the read façade dispatches both DataPageV1 and
//! DataPageV2. V2 differs in:
//!   - rep + def levels are stored UNCOMPRESSED at the start of the
//!     page body, with explicit byte lengths in the header
//!   - only the values portion may be compressed, gated by the
//!     `is_compressed` flag
//!   - num_values / encoding live in `data_page_header_v2`, not the
//!     V1 `data_page_header` field

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::read::{
    read_column_byte_array, read_column_i32, read_column_i64,
};
use ematix_parquet_io::ParquetFile;

use parquet::basic::{Compression, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn writer_props_v2(compression: Compression) -> Arc<WriterProperties> {
    // Force V2: PARQUET_2_0 + dict disabled (so the data pages are
    // PLAIN-encoded V2). Multiple pages so we exercise more than the
    // header-once code path.
    Arc::new(
        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(compression)
            .set_dictionary_enabled(false)
            .set_data_page_size_limit(2 * 1024)
            .set_encoding(parquet::basic::Encoding::PLAIN)
            .build(),
    )
}

#[test]
fn data_page_v2_i32_uncompressed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_i32.parquet");

    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::INT32)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let values: Vec<i32> = (0..5_000).collect();

    let f = File::create(&path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, writer_props_v2(Compression::UNCOMPRESSED)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::Int32ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(&values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let got = read_column_i32(&file, 0, 0).unwrap();
    assert_eq!(got, values);
}

#[test]
fn data_page_v2_i64_snappy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_i64.parquet");

    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::INT64)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let values: Vec<i64> = (0..3_000i64).map(|i| i * 17 - 1000).collect();

    let f = File::create(&path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, writer_props_v2(Compression::SNAPPY)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(&values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let got = read_column_i64(&file, 0, 0).unwrap();
    assert_eq!(got, values);
}

#[test]
fn data_page_v2_byte_array_uncompressed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_ba.parquet");

    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let values: Vec<PqByteArray> = (0..1_000)
        .map(|i| PqByteArray::from(format!("row-{i:06}").into_bytes()))
        .collect();

    let f = File::create(&path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, writer_props_v2(Compression::UNCOMPRESSED)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::ByteArrayColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(&values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let got = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(got.len(), values.len());
    for (g, w) in got.iter().zip(values.iter()) {
        assert_eq!(g.as_slice(), w.data());
    }
}
