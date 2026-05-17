//! Multi-column / multi-row-group bloom writer oracle.
//!
//! `write_table_with_blooms_to_path` round-trip:
//!
//! 1. Build a 2-column table (i32 + byte_array) with 4 row groups.
//! 2. Request bloom filters on both columns at fpp 0.01.
//! 3. Open via parquet-rs with `set_read_bloom_filter(true)`;
//!    confirm each (RG, col) has a bloom filter that reports every
//!    value in its RG slice as present.
//! 4. A column with bloom disabled gets no bloom_filter_offset.

use ematix_parquet_codec::write::{write_table_with_blooms_to_path, ColumnData};
use ematix_parquet_format::types::CompressionCodec;

#[test]
fn per_rg_per_column_blooms_interop_with_parquet_rs() {
    use parquet::data_type::ByteArray;
    use parquet::file::properties::ReaderProperties;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
    use std::fs::File;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_rg_bloom.parquet");

    let rows: usize = 4096;
    let ids: Vec<i32> = (0..rows as i32).map(|i| i * 13 - 7).collect();
    let tags_dict: [&[u8]; 8] = [
        b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf", b"hotel",
    ];
    let tags: Vec<&[u8]> = (0..rows).map(|i| tags_dict[i % 8]).collect();

    let rg_size = 1024;
    write_table_with_blooms_to_path(
        &path,
        &[
            ("id", ColumnData::I32(&ids)),
            ("tag", ColumnData::ByteArray(&tags)),
        ],
        CompressionCodec::Snappy,
        rg_size,
        &[Some(0.01), Some(0.01)],
    )
    .unwrap();

    let reader_props = ReaderProperties::builder()
        .set_read_bloom_filter(true)
        .build();
    let read_opts = ReadOptionsBuilder::new()
        .with_reader_properties(reader_props)
        .build();
    let reader =
        SerializedFileReader::new_with_options(File::open(&path).unwrap(), read_opts).unwrap();
    assert_eq!(reader.num_row_groups(), 4, "expected 4 row groups");

    for rg_ix in 0..4 {
        let rg = reader.get_row_group(rg_ix).unwrap();
        let lo = rg_ix * rg_size;
        let hi = (lo + rg_size).min(rows);

        // Column 0 (id) bloom filter: every id in [lo, hi) must
        // report present.
        let bf_id = rg.get_column_bloom_filter(0).unwrap_or_else(|| {
            panic!("rg {rg_ix} col 0 bloom missing");
        });
        for v in &ids[lo..hi] {
            assert!(
                bf_id.check(v),
                "rg {rg_ix} id {v} reported absent by parquet-rs"
            );
        }

        // Column 1 (tag) bloom filter: every tag in [lo, hi) must
        // report present.
        let bf_tag = rg
            .get_column_bloom_filter(1)
            .unwrap_or_else(|| panic!("rg {rg_ix} col 1 bloom missing"));
        for &t in &tags[lo..hi] {
            let ba = ByteArray::from(t.to_vec());
            assert!(
                bf_tag.check(&ba),
                "rg {rg_ix} tag {t:?} reported absent by parquet-rs"
            );
        }
    }
}

#[test]
fn opt_out_columns_get_no_bloom() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_bloom.parquet");

    let rows: usize = 256;
    let ids: Vec<i32> = (0..rows as i32).collect();
    let scores: Vec<f64> = (0..rows).map(|i| i as f64 * 0.5).collect();

    write_table_with_blooms_to_path(
        &path,
        &[
            ("id", ColumnData::I32(&ids)),
            ("score", ColumnData::F64(&scores)),
        ],
        CompressionCodec::Snappy,
        128,
        &[Some(0.01), None], // bloom only on `id`
    )
    .unwrap();

    // Inspect ColumnMetaData via our metadata reader.
    let file = ematix_parquet_io::ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 2);
    for (rg_ix, rg) in md.row_groups.iter().enumerate() {
        let id_cm = rg.columns[0].meta_data.as_ref().unwrap();
        let score_cm = rg.columns[1].meta_data.as_ref().unwrap();
        assert!(
            id_cm.bloom_filter_offset.is_some(),
            "rg {rg_ix} id should have bloom"
        );
        assert!(
            score_cm.bloom_filter_offset.is_none(),
            "rg {rg_ix} score must NOT have bloom"
        );
    }
}

#[test]
fn bloom_fpps_length_mismatch_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("err.parquet");
    let ids: Vec<i32> = vec![1, 2, 3];
    let r = write_table_with_blooms_to_path(
        &path,
        &[("id", ColumnData::I32(&ids))],
        CompressionCodec::Uncompressed,
        16,
        &[Some(0.01), Some(0.01)], // 2 fpps for 1 column
    );
    assert!(r.is_err());
}
