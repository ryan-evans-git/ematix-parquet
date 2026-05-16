//! Oracle: the `_into` family of read façade entry points
//! correctly reuses caller-provided buffers (clear-then-fill
//! semantics) and produces values byte-identical to the
//! allocating variants.
//!
//! Π.9a contract: `read_column_*_into(file, rg, col, &mut Vec<T>)`
//! mirrors `read_column_*` but writes into the caller's buffer.
//! Reusing the buffer across calls is the supported steady-state
//! pattern — no allocation past the first call once the buffer
//! has grown to the chunk size.

use ematix_parquet_codec::read::{
    read_column_byte_array, read_column_byte_array_into, read_column_byte_array_offsets,
    read_column_byte_array_offsets_into, read_column_f64, read_column_f64_into,
    read_column_i32, read_column_i32_into, read_column_i32_with_range,
    read_column_i32_with_range_into, read_column_i64, read_column_i64_into,
    read_column_i64_with_range, read_column_i64_with_range_into,
};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_f64_column_to_path,
    write_i32_column_to_path, write_i64_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

#[test]
fn i64_into_matches_allocating_variant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| i * 13 - 1000).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let allocating = read_column_i64(&file, 0, 0).unwrap();
    let mut buf: Vec<i64> = Vec::new();
    read_column_i64_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, allocating);
    assert_eq!(buf, values);
}

#[test]
fn i64_into_clears_buffer_on_each_call() {
    // Calling _into twice with the same buffer must overwrite, not
    // append. Pre-fill with garbage to make sure the clear happens.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_clear.parquet");
    let values: Vec<i64> = (0..1_000i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mut buf: Vec<i64> = vec![999_999; 50_000]; // garbage prefill, larger than chunk
    read_column_i64_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, values, "first read must clear the prefill");

    // Reread: must produce the same result, not append.
    read_column_i64_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, values, "second read must overwrite, not append");
}

#[test]
fn i64_into_buffer_reuse_no_growth_past_first_call() {
    // After the first call grows the buffer to the chunk size,
    // subsequent calls must not change the capacity.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_reuse.parquet");
    let values: Vec<i64> = (0..2_048i64).map(|i| i * 7).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mut buf: Vec<i64> = Vec::new();
    read_column_i64_into(&file, 0, 0, &mut buf).unwrap();
    let cap_after_first = buf.capacity();
    assert!(cap_after_first >= values.len());

    for _ in 0..9 {
        read_column_i64_into(&file, 0, 0, &mut buf).unwrap();
        assert_eq!(buf, values);
    }
    let cap_after_ten = buf.capacity();
    assert_eq!(
        cap_after_first, cap_after_ten,
        "10 successive reads must not grow capacity past the first"
    );
}

#[test]
fn i32_into_matches_allocating_variant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.parquet");
    let values: Vec<i32> = (0..3_000i32).map(|i| i * 17 - 500).collect();
    write_i32_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let allocating = read_column_i32(&file, 0, 0).unwrap();
    let mut buf: Vec<i32> = Vec::new();
    read_column_i32_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, allocating);
}

#[test]
fn f64_into_matches_allocating_variant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.parquet");
    let values: Vec<f64> = (0..1_000).map(|i| i as f64 * 0.5 - 100.0).collect();
    write_f64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let allocating = read_column_f64(&file, 0, 0).unwrap();
    let mut buf: Vec<f64> = Vec::new();
    read_column_f64_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, allocating);
}

#[test]
fn byte_array_into_matches_allocating_variant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba.parquet");
    let palette: [&[u8]; 4] = [b"alpha", b"bravo", b"charlie", b"delta"];
    let values: Vec<&[u8]> = (0..600).map(|i| palette[i % 4]).collect();
    write_byte_array_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let allocating = read_column_byte_array(&file, 0, 0).unwrap();
    let mut buf: Vec<Vec<u8>> = Vec::new();
    read_column_byte_array_into(&file, 0, 0, &mut buf).unwrap();
    assert_eq!(buf, allocating);
}

#[test]
fn byte_array_offsets_into_matches_allocating_variant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_off.parquet");
    let palette: [&[u8]; 3] = [b"A", b"R", b"N"];
    let values: Vec<&[u8]> = (0..1_000).map(|i| palette[i % 3]).collect();
    write_byte_array_column_dict_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Uncompressed,
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let (alloc_bytes, alloc_offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();

    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    read_column_byte_array_offsets_into(&file, 0, 0, &mut bytes, &mut offsets).unwrap();
    assert_eq!(bytes, alloc_bytes);
    assert_eq!(offsets, alloc_offsets);
}

#[test]
fn byte_array_offsets_into_buffer_reuse() {
    // Same dict file, called 10 times into the same buffers — must
    // not grow capacity past the first call.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_reuse.parquet");
    let palette: [&[u8]; 5] = [b"alpha", b"bravo", b"charlie", b"delta", b"echo"];
    let values: Vec<&[u8]> = (0..5_000).map(|i| palette[i % 5]).collect();
    write_byte_array_column_dict_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Uncompressed,
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    read_column_byte_array_offsets_into(&file, 0, 0, &mut bytes, &mut offsets).unwrap();
    let bytes_cap_first = bytes.capacity();
    let offsets_cap_first = offsets.capacity();

    for _ in 0..9 {
        read_column_byte_array_offsets_into(&file, 0, 0, &mut bytes, &mut offsets).unwrap();
    }
    assert_eq!(bytes.capacity(), bytes_cap_first);
    assert_eq!(offsets.capacity(), offsets_cap_first);
    assert_eq!(offsets.len(), values.len() + 1);
}

#[test]
fn i64_with_range_into_matches_allocating_variant() {
    use std::sync::Arc;
    use parquet::basic::{Compression, Repetition, Type as PhysicalType};
    use parquet::column::writer::ColumnWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as SchemaType;

    // Need a paged file with column index — use parquet-rs.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_paged.parquet");
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
    let props = Arc::new(
        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .set_compression(Compression::SNAPPY)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_size_limit(4 * 1024)
            .build(),
    );
    let f = std::fs::File::create(&path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
    let values: Vec<i64> = (0..5_000i64).collect();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(&values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let allocating = read_column_i64_with_range(&file, 0, 0, 1000, 2000).unwrap();
    let mut buf: Vec<i64> = Vec::new();
    read_column_i64_with_range_into(&file, 0, 0, 1000, 2000, &mut buf).unwrap();
    assert_eq!(buf, allocating);

    // i32 with range, same shape.
    let _ = read_column_i32_with_range;
    let _ = read_column_i32_with_range_into;
}
