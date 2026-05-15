//! Oracle: parquet-rs writes BYTE_STREAM_SPLIT for f32 / f64;
//! our decoder reads the page body and recovers the values bit-exact.
//!
//! Π.7 contract: spec-completeness for the BYTE_STREAM_SPLIT
//! encoding. parquet-rs's writer (with the right encoding hint) is
//! the cross-check.

use std::sync::Arc;

use ematix_parquet_codec::byte_stream_split::{
    decode_byte_stream_split_f32, decode_byte_stream_split_f64,
};
use ematix_parquet_codec::compression::decompress_snappy_into;
use ematix_parquet_format::types::{CompressionCodec, Encoding as EmEncoding, PageType};
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::basic::{Compression, Encoding as PqEncoding, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn write_f32_byte_stream_split(path: &std::path::Path, values: &[f32]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::FLOAT)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_encoding(PqEncoding::BYTE_STREAM_SPLIT)
            .set_compression(Compression::SNAPPY)
            .build(),
    );
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::FloatColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_f64_byte_stream_split(path: &std::path::Path, values: &[f64]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::DOUBLE)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_encoding(PqEncoding::BYTE_STREAM_SPLIT)
            .set_compression(Compression::UNCOMPRESSED)
            .build(),
    );
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::DoubleColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

/// Pull the decompressed body of the first data page of column 0
/// from a single-row-group file. Returns (body bytes, num_values,
/// encoding) — enough for the per-encoding oracle to dispatch.
fn first_data_page(path: &std::path::Path) -> (Vec<u8>, usize, EmEncoding) {
    let file = ParquetFile::open(path).unwrap();
    let md = file.metadata().unwrap();
    let cm = md.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let chunk = file
        .read_range(cm.data_page_offset as u64, cm.total_compressed_size as u64)
        .unwrap();
    let codec = cm.codec;
    drop(md);

    let mut walker = PageWalker::new(&chunk);
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        if matches!(hdr.page_type, PageType::DataPage | PageType::DataPageV2) {
            // Decompress whatever the codec is — for v1 the whole body
            // is one compressed unit; the BSS oracle uses v1.
            let mut decomp = Vec::new();
            match codec {
                CompressionCodec::Uncompressed => decomp.extend_from_slice(body),
                CompressionCodec::Snappy => decompress_snappy_into(body, &mut decomp).unwrap(),
                _ => panic!("test should pin a known codec"),
            };
            let dph = hdr
                .data_page_header
                .as_ref()
                .expect("v1 page header expected for BSS oracle");
            return (decomp, dph.num_values as usize, dph.encoding);
        }
    }
    panic!("no data page found");
}

#[test]
fn f32_byte_stream_split_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bss_f32.parquet");

    let values: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5 - 100.0).collect();
    write_f32_byte_stream_split(&path, &values);

    let (body, n, encoding) = first_data_page(&path);
    assert_eq!(encoding, EmEncoding::ByteStreamSplit);
    assert_eq!(n, values.len());

    let decoded = decode_byte_stream_split_f32(&body, n).unwrap();
    for (a, b) in decoded.iter().zip(values.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn f64_byte_stream_split_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bss_f64.parquet");

    let values: Vec<f64> = (0..512).map(|i| (i as f64) / 7.0 - 50.0).collect();
    write_f64_byte_stream_split(&path, &values);

    let (body, n, encoding) = first_data_page(&path);
    assert_eq!(encoding, EmEncoding::ByteStreamSplit);
    assert_eq!(n, values.len());

    let decoded = decode_byte_stream_split_f64(&body, n).unwrap();
    for (a, b) in decoded.iter().zip(values.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
