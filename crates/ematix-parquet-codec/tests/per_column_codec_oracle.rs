//! `write_table_with_options_to_path` per-column codec oracle.
//!
//! Confirms:
//!   * Different columns in the same row group can use different
//!     codecs (Snappy, Zstd, Uncompressed) and parquet-rs decodes
//!     each per its column's declared codec.
//!   * ColumnMetaData.codec is recorded per column (not the
//!     default).
//!   * codec_per_column combines with dict_per_column and bloom_fpps
//!     correctly.
//!   * Length-mismatch on codec_per_column is rejected.
//!   * Default options round-trip equivalently to the existing
//!     uncompressed-PLAIN entry point.

use ematix_parquet_codec::write::{write_table_with_options_to_path, ColumnData, WriteOptions};
use ematix_parquet_format::types::CompressionCodec;
use parquet::data_type::ByteArray;
use parquet::file::properties::ReaderProperties;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
use parquet::record::Field;
use std::fs::File;

#[test]
fn three_columns_three_codecs_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("three_codecs.parquet");

    let ids: Vec<i32> = (0..1024).collect();
    let scores: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.5).collect();
    let owned: Vec<String> = (0..1024).map(|i| format!("row-{i:04}")).collect();
    let tags: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();

    let codecs = [
        CompressionCodec::Snappy,
        CompressionCodec::Zstd,
        CompressionCodec::Uncompressed,
    ];
    let opts = WriteOptions {
        row_group_size: 256,
        default_codec: CompressionCodec::Snappy,
        codec_per_column: Some(&codecs),
        ..Default::default()
    };

    write_table_with_options_to_path(
        &path,
        &[
            ("id", ColumnData::I32(&ids)),
            ("score", ColumnData::F64(&scores)),
            ("tag", ColumnData::ByteArray(&tags)),
        ],
        &opts,
    )
    .unwrap();

    // Metadata: every column carries its own codec.
    let file = ematix_parquet_io::ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 4);
    for rg in &md.row_groups {
        assert_eq!(
            rg.columns[0].meta_data.as_ref().unwrap().codec,
            CompressionCodec::Snappy
        );
        assert_eq!(
            rg.columns[1].meta_data.as_ref().unwrap().codec,
            CompressionCodec::Zstd
        );
        assert_eq!(
            rg.columns[2].meta_data.as_ref().unwrap().codec,
            CompressionCodec::Uncompressed
        );
    }

    // Data: parquet-rs decompresses each column with its own codec.
    let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let mut row_ix = 0usize;
    for rg_ix in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(rg_ix).unwrap();
        let iter = rg.get_row_iter(None).unwrap();
        for row in iter {
            let row = row.unwrap();
            let mut it = row.get_column_iter();
            match it.next().unwrap().1 {
                Field::Int(v) => assert_eq!(*v, ids[row_ix]),
                other => panic!("id col: {other:?}"),
            }
            match it.next().unwrap().1 {
                Field::Double(v) => assert_eq!(*v, scores[row_ix]),
                other => panic!("score col: {other:?}"),
            }
            match it.next().unwrap().1 {
                Field::Bytes(b) => assert_eq!(b.data(), tags[row_ix]),
                Field::Str(s) => assert_eq!(s.as_bytes(), tags[row_ix]),
                other => panic!("tag col: {other:?}"),
            }
            row_ix += 1;
        }
    }
    assert_eq!(row_ix, 1024);
}

#[test]
fn codec_plus_dict_plus_bloom_combined() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("combo.parquet");

    let rows: usize = 512;
    let ids: Vec<i64> = (0..rows as i64).map(|i| i * 13).collect();
    let tags_dict: [&[u8]; 4] = [b"alpha", b"bravo", b"charlie", b"delta"];
    let tags: Vec<&[u8]> = (0..rows).map(|i| tags_dict[i % 4]).collect();

    let codecs = [CompressionCodec::Snappy, CompressionCodec::Zstd];
    let dicts = [false, true];
    let blooms: [Option<f64>; 2] = [Some(0.01), Some(0.01)];

    let opts = WriteOptions {
        row_group_size: 128,
        codec_per_column: Some(&codecs),
        dict_per_column: Some(&dicts),
        bloom_fpps: Some(&blooms),
        ..Default::default()
    };

    write_table_with_options_to_path(
        &path,
        &[
            ("id", ColumnData::I64(&ids)),
            ("tag", ColumnData::ByteArray(&tags)),
        ],
        &opts,
    )
    .unwrap();

    // Bloom is consultable through parquet-rs, and each column has
    // its declared codec / dict shape.
    let reader_props = ReaderProperties::builder()
        .set_read_bloom_filter(true)
        .build();
    let read_opts = ReadOptionsBuilder::new()
        .with_reader_properties(reader_props)
        .build();
    let reader =
        SerializedFileReader::new_with_options(File::open(&path).unwrap(), read_opts).unwrap();
    assert_eq!(reader.num_row_groups(), 4);

    for rg_ix in 0..4 {
        let rg = reader.get_row_group(rg_ix).unwrap();
        let lo = rg_ix * 128;
        let hi = (lo + 128).min(rows);
        let bf_id = rg.get_column_bloom_filter(0).expect("id bloom missing");
        for v in &ids[lo..hi] {
            assert!(bf_id.check(v), "rg {rg_ix} id {v}");
        }
        let bf_tag = rg.get_column_bloom_filter(1).expect("tag bloom missing");
        for &t in &tags[lo..hi] {
            assert!(bf_tag.check(&ByteArray::from(t.to_vec())), "rg {rg_ix}");
        }
    }

    // Metadata: id is PLAIN+Snappy, tag is DICT+Zstd.
    let file = ematix_parquet_io::ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    for rg in &md.row_groups {
        let id_cm = rg.columns[0].meta_data.as_ref().unwrap();
        let tag_cm = rg.columns[1].meta_data.as_ref().unwrap();
        assert_eq!(id_cm.codec, CompressionCodec::Snappy);
        assert!(id_cm.dictionary_page_offset.is_none());
        assert_eq!(tag_cm.codec, CompressionCodec::Zstd);
        assert!(tag_cm.dictionary_page_offset.is_some());
    }
}

#[test]
fn codec_per_column_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("err.parquet");
    let ids = vec![1i32, 2, 3];
    let codecs = [CompressionCodec::Snappy, CompressionCodec::Zstd]; // length 2 for a 1-column table
    let opts = WriteOptions {
        codec_per_column: Some(&codecs),
        ..Default::default()
    };
    let r = write_table_with_options_to_path(&path, &[("id", ColumnData::I32(&ids))], &opts);
    assert!(r.is_err(), "length mismatch must be rejected");
}

#[test]
fn default_options_match_existing_writer_shape() {
    // WriteOptions::default() == uncompressed PLAIN, single RG, V1 —
    // should round-trip cleanly through the basic reader path.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("defaults.parquet");
    let ids: Vec<i32> = (0..64).collect();
    let opts = WriteOptions::default();
    write_table_with_options_to_path(&path, &[("id", ColumnData::I32(&ids))], &opts).unwrap();

    let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    assert_eq!(reader.num_row_groups(), 1, "single RG expected");
    let rg = reader.get_row_group(0).unwrap();
    let mut got = Vec::new();
    for row in rg.get_row_iter(None).unwrap() {
        let row = row.unwrap();
        if let Field::Int(v) = row.get_column_iter().next().unwrap().1 {
            got.push(*v);
        }
    }
    assert_eq!(got, ids);
}
