//! Bloom-filter writer integration oracle.
//!
//! `write_i32_column_dict_with_bloom_to_path` round-trip:
//!
//! 1. Write an i32 column with a bloom filter.
//! 2. Parse the file via our reader; read the bytes at
//!    `ColumnMetaData.bloom_filter_offset`; decode via
//!    `SplitBlockBloomFilter::from_bytes`; confirm every distinct
//!    value reports present and a 10K-probe of absent values has
//!    a false-positive rate below 5%.
//! 3. Parse the file via parquet-rs; pull the bloom filter via
//!    its native API; confirm parquet-rs's filter agrees with
//!    ours on the same membership queries.

use ematix_parquet_codec::bloom::{parquet_xxh64, SplitBlockBloomFilter};
use ematix_parquet_codec::read::read_column_i32;
use ematix_parquet_codec::write::write_i32_column_dict_with_bloom_to_path;
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

fn load_bloom_bytes(file: &ParquetFile) -> Vec<u8> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let off = cm
        .bloom_filter_offset
        .expect("bloom_filter_offset must be present after write_*_with_bloom");
    let length =
        cm.bloom_filter_length
            .expect("bloom_filter_length must be present after write_*_with_bloom") as u64;
    file.read_range(off as u64, length).unwrap()
}

#[test]
fn writer_round_trips_through_our_decoder() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("with_bloom.parquet");

    // 1024 distinct i32s — keeps the filter small but exercises
    // enough block diversity that the false-positive rate is
    // honestly measurable.
    let present: Vec<i32> = (0i32..1024).map(|i| i * 17 + 3).collect();
    write_i32_column_dict_with_bloom_to_path(&path, "v", &present, CompressionCodec::Snappy, 0.01)
        .unwrap();

    let file = ParquetFile::open(&path).unwrap();

    // Data round-trips unaffected by the bloom-filter addition.
    let read_back = read_column_i32(&file, 0, 0).unwrap();
    assert_eq!(read_back, present);

    // Bloom-filter membership.
    let bytes = load_bloom_bytes(&file);
    let bf = SplitBlockBloomFilter::from_bytes(&bytes).unwrap();
    for &v in &present {
        let h = parquet_xxh64(&v.to_le_bytes());
        assert!(bf.contains_hash(h), "false negative on present value {v}");
    }

    // FPR sanity: probe 10K absent values, expect << 5% positives.
    let mut fp = 0usize;
    for i in 100_000i32..110_000 {
        let h = parquet_xxh64(&i.to_le_bytes());
        if bf.contains_hash(h) {
            fp += 1;
        }
    }
    assert!(
        fp < 500,
        "false-positive rate too high: {fp} / 10000 (target fpp 1%)"
    );
}

#[test]
fn parquet_rs_can_read_the_bloom_filter_we_wrote() {
    use parquet::file::properties::ReaderProperties;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::file::serialized_reader::ReadOptionsBuilder;
    use std::fs::File;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("interop_bloom.parquet");

    let present: Vec<i32> = (0i32..512).map(|i| i * 5 + 11).collect();
    write_i32_column_dict_with_bloom_to_path(&path, "id", &present, CompressionCodec::Snappy, 0.01)
        .unwrap();

    // First, sanity-check metadata-only access without bloom enabled.
    {
        let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        let md = reader.metadata();
        let col_md = md.row_group(0).column(0);
        assert!(
            col_md.bloom_filter_offset().is_some(),
            "parquet-rs must see the bloom_filter_offset we wrote"
        );
        assert!(
            col_md.bloom_filter_length().is_some(),
            "parquet-rs must see the bloom_filter_length we wrote"
        );
    }

    // Now actually load the bloom filter — parquet-rs requires this
    // to be opted in via ReaderProperties::set_read_bloom_filter(true).
    let reader_props = ReaderProperties::builder()
        .set_read_bloom_filter(true)
        .build();
    let read_opts = ReadOptionsBuilder::new()
        .with_reader_properties(reader_props)
        .build();
    let reader =
        SerializedFileReader::new_with_options(File::open(&path).unwrap(), read_opts).unwrap();
    let rg_reader = reader.get_row_group(0).unwrap();
    let bf = rg_reader
        .get_column_bloom_filter(0)
        .expect("parquet-rs reads the filter we wrote");

    // Every present value should be reported "possibly present".
    for &v in &present {
        assert!(
            bf.check(&v),
            "parquet-rs reported false negative for present value {v}"
        );
    }
}
