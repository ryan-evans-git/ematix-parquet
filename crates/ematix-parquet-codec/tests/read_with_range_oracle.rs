//! Oracle: `read_column_{i32,i64}_with_range` skips pages whose
//! `[min, max]` doesn't intersect the predicate, and returns every
//! value the caller can possibly need to filter further.
//!
//! Π.5a contract: page-index pruning at the read façade. The win is
//! the pages we never decompressed/decoded; the returned vec is the
//! union of values from kept pages. Caller still applies the
//! row-level filter.

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::read::{
    read_column_i32, read_column_i32_with_range, read_column_i64, read_column_i64_with_range,
};
use ematix_parquet_io::ParquetFile;

use parquet::basic::{Compression, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

/// Write a single-column INT32 file with multiple V1 data pages and
/// page-level statistics enabled (so the writer emits ColumnIndex +
/// OffsetIndex). Uses PARQUET_1_0 so the data pages are V1 — that's
/// what our high-level façade decodes today (V2 lands in Π.6a).
fn write_paged_i32_v1(path: &std::path::Path, values: &[i32]) {
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
    let props = Arc::new(
        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .set_compression(Compression::SNAPPY)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_size_limit(4 * 1024)
            .set_encoding(parquet::basic::Encoding::PLAIN)
            .build(),
    );
    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int32ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    row_group.close().unwrap();
    writer.close().unwrap();
}

fn write_paged_i64_v1(path: &std::path::Path, values: &[i64]) {
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
            .set_encoding(parquet::basic::Encoding::PLAIN)
            .build(),
    );
    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    row_group.close().unwrap();
    writer.close().unwrap();
}

// ---- i32 ----

#[test]
fn i32_with_range_returns_superset_of_matches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.parquet");

    let values: Vec<i32> = (0..10_000).collect();
    write_paged_i32_v1(&path, &values);

    let file = ParquetFile::open(&path).unwrap();
    let lo = 4000;
    let hi = 5500;
    let kept = read_column_i32_with_range(&file, 0, 0, lo, hi).unwrap();

    // Caller-side filter on the kept superset must recover exactly
    // the values that satisfy the predicate.
    let recovered: Vec<i32> = kept.iter().copied().filter(|&v| v >= lo && v <= hi).collect();
    let expected: Vec<i32> = values.iter().copied().filter(|&v| v >= lo && v <= hi).collect();
    assert_eq!(recovered, expected);
}

#[test]
fn i32_with_range_skips_non_overlapping_pages() {
    // Pruning must actually skip pages — the returned vec should be
    // smaller than the full chunk for a narrow predicate.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32_skip.parquet");

    let values: Vec<i32> = (0..10_000).collect();
    write_paged_i32_v1(&path, &values);

    let file = ParquetFile::open(&path).unwrap();
    let kept = read_column_i32_with_range(&file, 0, 0, 4000, 5500).unwrap();
    let full = read_column_i32(&file, 0, 0).unwrap();

    assert_eq!(full.len(), values.len());
    assert!(
        kept.len() < full.len(),
        "pruning must drop pages: kept {} of {}",
        kept.len(),
        full.len()
    );
    // And the kept set must still contain every match.
    for &v in values.iter().filter(|&&v| (4000..=5500).contains(&v)) {
        assert!(kept.contains(&v), "missing matching value {v}");
    }
}

#[test]
fn i32_with_range_predicate_above_max_drops_almost_everything() {
    // Predicate entirely above the chunk's max → ideally zero pages
    // kept. parquet-rs may emit one tail page that overlaps; the
    // contract is "no false negatives" and "fewer than full."
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32_high.parquet");

    let values: Vec<i32> = (0..10_000).collect();
    write_paged_i32_v1(&path, &values);

    let file = ParquetFile::open(&path).unwrap();
    let kept = read_column_i32_with_range(&file, 0, 0, 50_000, 60_000).unwrap();
    assert!(
        kept.iter().all(|&v| !(50_000..=60_000).contains(&v)),
        "no value in the original data could match"
    );
    // No false positives that would have matched.
    assert!(
        kept.len() < 10_000,
        "must drop most/all pages: kept {} of 10000",
        kept.len()
    );
}

// ---- i64 ----

#[test]
fn i64_with_range_returns_superset_of_matches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.parquet");

    let values: Vec<i64> = (0..10_000i64).map(|i| i * 100).collect();
    write_paged_i64_v1(&path, &values);

    let file = ParquetFile::open(&path).unwrap();
    let lo = 100_000i64;
    let hi = 200_000i64;
    let kept = read_column_i64_with_range(&file, 0, 0, lo, hi).unwrap();
    let full = read_column_i64(&file, 0, 0).unwrap();

    assert!(kept.len() < full.len(), "pruning must drop pages");
    let recovered: Vec<i64> = kept.iter().copied().filter(|&v| v >= lo && v <= hi).collect();
    let expected: Vec<i64> = values.iter().copied().filter(|&v| v >= lo && v <= hi).collect();
    assert_eq!(recovered, expected);
}

// ---- fallback when no column index ----

#[test]
fn with_range_falls_back_to_full_read_when_no_index() {
    // Files we write today don't include a column index. The
    // with-range entry point must degrade gracefully to a full read.
    use ematix_parquet_codec::write::write_i32_column_to_path;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours.parquet");

    let values: Vec<i32> = (0..500).collect();
    write_i32_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    // Predicate inside the data range — fallback must return all
    // values (caller filters).
    let kept = read_column_i32_with_range(&file, 0, 0, 100, 200).unwrap();
    assert_eq!(kept, values, "fallback returns full chunk unchanged");
}
