//! Oracle: parquet-rs writes bloom filters → our SBBF decoder
//! answers membership queries correctly.
//!
//! Π.6c contract: given a column chunk with `bloom_filter_offset` set,
//! we can read the bloom filter bytes, decode the header, attach the
//! bitset, and answer `contains(x)` for any value the writer inserted
//! (true positive — must always return true), as well as for values
//! the writer didn't insert (most should return false; a few false
//! positives are allowed by bloom semantics).

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::bloom::{parquet_xxh64, SplitBlockBloomFilter};
use ematix_parquet_io::ParquetFile;

use parquet::basic::{Compression, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::properties::{BloomFilterPosition, WriterProperties};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn write_i64_with_bloom(path: &std::path::Path, values: &[i64]) {
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
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_bloom_filter_enabled(true)
            .set_bloom_filter_position(BloomFilterPosition::End)
            .build(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_byte_array_with_bloom(path: &std::path::Path, values: &[PqByteArray]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("s", PhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_bloom_filter_enabled(true)
            .set_bloom_filter_position(BloomFilterPosition::End)
            .build(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    if let ColumnWriter::ByteArrayColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn load_bloom_bytes(file: &ParquetFile, row_group: usize, column: usize) -> Vec<u8> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[row_group].columns[column]
        .meta_data
        .as_ref()
        .unwrap();
    let off = cm.bloom_filter_offset.expect("bloom_filter_offset");
    // length is optional — when absent we read enough to cover header
    // + the largest plausible bitset and let the decoder bound by
    // `num_bytes`.
    let approx_len = cm.bloom_filter_length.unwrap_or(64 * 1024) as u64;
    file.read_range(off as u64, approx_len).unwrap()
}

#[test]
fn bloom_i64_known_present_and_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bloom_i64.parquet");

    // 2000 distinct i64s. Every present value must round-trip true.
    let present: Vec<i64> = (0i64..2_000).map(|i| i * 31 + 7).collect();
    write_i64_with_bloom(&path, &present);

    let file = ParquetFile::open(&path).unwrap();
    let bytes = load_bloom_bytes(&file, 0, 0);
    let bf = SplitBlockBloomFilter::from_bytes(&bytes).unwrap();

    // True positives: every inserted value must report present.
    for &v in &present {
        let h = parquet_xxh64(&v.to_le_bytes());
        assert!(
            bf.contains_hash(h),
            "false negative for present value {v}"
        );
    }

    // False-positive rate check: probe 10 000 absent values, expect
    // most to report absent. The default fpp is ~1%, so a 5% upper
    // bound is generous and tolerant of the exact filter sizing the
    // writer chose.
    let mut false_positives = 0usize;
    let absent_probes = 10_000;
    for i in 0..absent_probes {
        // Pick values that can't collide with anything in `present`
        // (they're in [0, 2000*31+7); pick a different residue).
        let v = i64::MAX - i as i64;
        let h = parquet_xxh64(&v.to_le_bytes());
        if bf.contains_hash(h) {
            false_positives += 1;
        }
    }
    assert!(
        false_positives * 20 < absent_probes,
        "fp rate too high: {} of {}",
        false_positives,
        absent_probes
    );
}

#[test]
fn bloom_byte_array_known_present_and_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bloom_ba.parquet");

    let present: Vec<PqByteArray> = (0..1_000)
        .map(|i| PqByteArray::from(format!("user_{i:06}").into_bytes()))
        .collect();
    write_byte_array_with_bloom(&path, &present);

    let file = ParquetFile::open(&path).unwrap();
    let bytes = load_bloom_bytes(&file, 0, 0);
    let bf = SplitBlockBloomFilter::from_bytes(&bytes).unwrap();

    // Every inserted value present.
    for v in &present {
        let h = parquet_xxh64(v.data());
        assert!(bf.contains_hash(h), "false negative for {:?}", v.data());
    }

    // Most absent values absent. Pick a non-overlapping namespace.
    let mut false_positives = 0usize;
    let probes = 5_000;
    for i in 0..probes {
        let probe = format!("absent_{i:06}");
        let h = parquet_xxh64(probe.as_bytes());
        if bf.contains_hash(h) {
            false_positives += 1;
        }
    }
    assert!(
        false_positives * 20 < probes,
        "fp rate too high: {} of {}",
        false_positives,
        probes
    );
}

#[test]
fn bloom_contains_bytes_helper_matches_hash_path() {
    // Sanity: contains_bytes(b) must equal contains_hash(xxh64(b)).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bloom_helper.parquet");
    let present: Vec<PqByteArray> = (0..50)
        .map(|i| PqByteArray::from(format!("k{i}").into_bytes()))
        .collect();
    write_byte_array_with_bloom(&path, &present);

    let file = ParquetFile::open(&path).unwrap();
    let bytes = load_bloom_bytes(&file, 0, 0);
    let bf = SplitBlockBloomFilter::from_bytes(&bytes).unwrap();

    for k in &present {
        assert!(bf.contains_bytes(k.data()));
        assert_eq!(
            bf.contains_bytes(k.data()),
            bf.contains_hash(parquet_xxh64(k.data()))
        );
    }
}
