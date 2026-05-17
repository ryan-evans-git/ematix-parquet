//! PLAIN (non-dict) bloom writer × parquet-rs oracle.
//!
//! For each of the four PLAIN bloom entry points
//! (`write_{i32,i64,f64,byte_array}_column_with_bloom_to_path`):
//!
//! 1. Write a single-column file with an attached SBBF at fpp 0.01.
//! 2. Open via parquet-rs with `set_read_bloom_filter(true)`.
//! 3. Confirm every written value reports present (no false negatives).
//! 4. Confirm an out-of-range / never-inserted value reports absent
//!    most of the time (sanity check that the filter is real and
//!    actually narrows the space).

use ematix_parquet_codec::write::{
    write_byte_array_column_with_bloom_to_path, write_f64_column_with_bloom_to_path,
    write_i32_column_with_bloom_to_path, write_i64_column_with_bloom_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use parquet::data_type::ByteArray;
use parquet::file::properties::ReaderProperties;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
use std::fs::File;

fn open_with_bloom(path: &std::path::Path) -> SerializedFileReader<File> {
    let reader_props = ReaderProperties::builder()
        .set_read_bloom_filter(true)
        .build();
    let read_opts = ReadOptionsBuilder::new()
        .with_reader_properties(reader_props)
        .build();
    SerializedFileReader::new_with_options(File::open(path).unwrap(), read_opts).unwrap()
}

#[test]
fn plain_i32_bloom_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_i32_bloom.parquet");

    // Mix duplicates in — the PLAIN bloom path hashes every value
    // (not distinct values), so this also exercises the duplicate
    // insert path. Distinct cardinality ≈ 1024.
    let values: Vec<i32> = (0..4096).map(|i| (i % 1024) * 7 - 3).collect();
    write_i32_column_with_bloom_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Snappy,
        0.01,
    )
    .unwrap();

    let reader = open_with_bloom(&path);
    let rg = reader.get_row_group(0).unwrap();
    let bf = rg.get_column_bloom_filter(0).expect("bloom missing");
    for v in &values {
        assert!(bf.check(v), "value {v} reported absent");
    }
    // Sanity: a value provably not in the column space.
    let never: i32 = 1024 * 7 + 100;
    assert!(!values.contains(&never));
    // We can't assert `bf.check(never) == false` deterministically
    // (false positives are allowed), but the SBBF sized at fpp 0.01
    // should fail this often enough to make the assertion meaningful
    // across many distinct probes. Probe 32 disjoint never-inserted
    // values and require at least one absent.
    let mut any_absent = false;
    for k in 0i32..32 {
        let probe = i32::MAX - k;
        assert!(!values.contains(&probe));
        if !bf.check(&probe) {
            any_absent = true;
            break;
        }
    }
    assert!(
        any_absent,
        "SBBF reported every probe present — fpp budget blown"
    );
}

#[test]
fn plain_i64_bloom_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_i64_bloom.parquet");

    let values: Vec<i64> = (0..2048).map(|i| (i as i64) * 1_000_003 - 17).collect();
    write_i64_column_with_bloom_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Snappy,
        0.01,
    )
    .unwrap();

    let reader = open_with_bloom(&path);
    let rg = reader.get_row_group(0).unwrap();
    let bf = rg.get_column_bloom_filter(0).expect("bloom missing");
    for v in &values {
        assert!(bf.check(v), "i64 value {v} reported absent");
    }
}

#[test]
fn plain_f64_bloom_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_f64_bloom.parquet");

    let values: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.5 - 100.0).collect();
    write_f64_column_with_bloom_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Snappy,
        0.01,
    )
    .unwrap();

    let reader = open_with_bloom(&path);
    let rg = reader.get_row_group(0).unwrap();
    let bf = rg.get_column_bloom_filter(0).expect("bloom missing");
    for v in &values {
        assert!(bf.check(v), "f64 value {v} reported absent");
    }
}

#[test]
fn plain_byte_array_bloom_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_ba_bloom.parquet");

    // High-cardinality strings — the typical reason you want a
    // PLAIN-encoded BYTE_ARRAY bloom in the first place (dict encoding
    // breaks down past a few thousand uniques).
    let owned: Vec<String> = (0..512).map(|i| format!("string-{i:04}")).collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();

    write_byte_array_column_with_bloom_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Snappy,
        0.01,
    )
    .unwrap();

    let reader = open_with_bloom(&path);
    let rg = reader.get_row_group(0).unwrap();
    let bf = rg.get_column_bloom_filter(0).expect("bloom missing");
    for s in &owned {
        let ba = ByteArray::from(s.as_bytes().to_vec());
        assert!(bf.check(&ba), "string {s:?} reported absent");
    }

    // Sanity: a string we never inserted should usually report absent.
    let mut any_absent = false;
    for k in 0..32 {
        let probe = format!("not-present-{k}");
        let ba = ByteArray::from(probe.as_bytes().to_vec());
        if !bf.check(&ba) {
            any_absent = true;
            break;
        }
    }
    assert!(any_absent, "SBBF reported every never-inserted probe present");
}

#[test]
fn plain_bloom_offset_recorded_in_metadata() {
    // bloom_filter_offset / bloom_filter_length must be present in
    // the ColumnMetaData — this is what tells the reader where to
    // find the SBBF blob on disk. Regression guard for the earlier
    // case where the metadata writer panicked on these fields.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("offset_check.parquet");
    let values: Vec<i32> = (0..256).collect();
    write_i32_column_with_bloom_to_path(
        &path,
        "v",
        &values,
        CompressionCodec::Uncompressed,
        0.01,
    )
    .unwrap();

    let file = ematix_parquet_io::ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 1);
    let cm = md.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let off = cm
        .bloom_filter_offset
        .expect("bloom_filter_offset must be set on PLAIN-with-bloom files");
    let len = cm
        .bloom_filter_length
        .expect("bloom_filter_length must be set on PLAIN-with-bloom files");
    assert!(off > 0, "bloom offset should be > 0");
    assert!(len > 0, "bloom length should be > 0");
    // Bloom blob must end before the footer starts.
    assert!(
        off as u64 + len as u64 <= file.footer_offset(),
        "bloom blob extends into the footer region"
    );
}
