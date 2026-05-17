//! Multi-column / multi-RG dict-encoding writer oracle.
//!
//! Exercises `write_table_with_dict_to_path` (and the bloom-combined
//! sibling) round-tripping through parquet-rs to confirm:
//!   * Dict-encoded columns and PLAIN columns coexist in the same RG.
//!   * Per-RG dictionaries are written (not file-wide), so each RG
//!     reproduces the right values.
//!   * `dictionary_page_offset` is set on dict columns, absent on
//!     PLAIN columns.
//!   * Combined with bloom: every value in the column reports present
//!     via `get_column_bloom_filter`.
//!   * Validation: dict on a Bool column is rejected.

use ematix_parquet_codec::write::{
    write_table_with_dict_and_blooms_to_path, write_table_with_dict_to_path, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::record::Field;
use std::fs::File;

fn read_column<F>(path: &std::path::Path, col_ix: usize, mut extract: F) -> Vec<Field>
where
    F: FnMut(&Field) -> Field,
{
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let mut out = Vec::new();
    for rg_ix in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(rg_ix).unwrap();
        let iter = rg.get_row_iter(None).unwrap();
        for row in iter {
            let row = row.unwrap();
            let f = row.get_column_iter().nth(col_ix).unwrap().1;
            out.push(extract(f));
        }
    }
    out
}

#[test]
fn mixed_dict_and_plain_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_dict_plain.parquet");

    // 4 row groups × 256 rows = 1024 rows.
    // Column 0 (id, i32): PLAIN. Column 1 (tag, byte_array): DICT.
    // Column 2 (score, f64): PLAIN. Column 3 (region, byte_array): DICT.
    let rows: usize = 1024;
    let ids: Vec<i32> = (0..rows as i32).collect();
    let tags_dict: [&[u8]; 4] = [b"red", b"green", b"blue", b"amber"];
    let tags: Vec<&[u8]> = (0..rows).map(|i| tags_dict[i % 4]).collect();
    let scores: Vec<f64> = (0..rows).map(|i| (i as f64) * 0.25).collect();
    let regions_dict: [&[u8]; 3] = [b"us-east", b"us-west", b"eu-west"];
    let regions: Vec<&[u8]> = (0..rows).map(|i| regions_dict[i % 3]).collect();

    write_table_with_dict_to_path(
        &path,
        &[
            ("id", ColumnData::I32(&ids)),
            ("tag", ColumnData::ByteArray(&tags)),
            ("score", ColumnData::F64(&scores)),
            ("region", ColumnData::ByteArray(&regions)),
        ],
        CompressionCodec::Snappy,
        256,
        &[false, true, false, true],
    )
    .unwrap();

    // ---- 1. Metadata shape: dict columns have dict_page_offset set ----
    let file = ematix_parquet_io::ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert_eq!(md.row_groups.len(), 4, "expected 4 row groups");
    for rg in &md.row_groups {
        let id_cm = rg.columns[0].meta_data.as_ref().unwrap();
        let tag_cm = rg.columns[1].meta_data.as_ref().unwrap();
        let score_cm = rg.columns[2].meta_data.as_ref().unwrap();
        let region_cm = rg.columns[3].meta_data.as_ref().unwrap();
        assert!(
            id_cm.dictionary_page_offset.is_none(),
            "PLAIN i32 should have no dict offset"
        );
        assert!(
            tag_cm.dictionary_page_offset.is_some(),
            "DICT byte_array (tag) should have dict offset"
        );
        assert!(
            score_cm.dictionary_page_offset.is_none(),
            "PLAIN f64 should have no dict offset"
        );
        assert!(
            region_cm.dictionary_page_offset.is_some(),
            "DICT byte_array (region) should have dict offset"
        );
    }

    // ---- 2. Row data: every column round-trips through parquet-rs ----
    let id_back = read_column(&path, 0, |f| f.clone());
    let tag_back = read_column(&path, 1, |f| f.clone());
    let score_back = read_column(&path, 2, |f| f.clone());
    let region_back = read_column(&path, 3, |f| f.clone());
    assert_eq!(id_back.len(), rows);

    for (i, f) in id_back.iter().enumerate() {
        match f {
            Field::Int(v) => assert_eq!(*v, ids[i], "id row {i}"),
            other => panic!("id row {i}: expected Int got {other:?}"),
        }
    }
    for (i, f) in tag_back.iter().enumerate() {
        match f {
            Field::Bytes(b) => assert_eq!(b.data(), tags[i], "tag row {i}"),
            Field::Str(s) => assert_eq!(s.as_bytes(), tags[i], "tag row {i}"),
            other => panic!("tag row {i}: unexpected {other:?}"),
        }
    }
    for (i, f) in score_back.iter().enumerate() {
        match f {
            Field::Double(v) => assert_eq!(*v, scores[i], "score row {i}"),
            other => panic!("score row {i}: expected Double got {other:?}"),
        }
    }
    for (i, f) in region_back.iter().enumerate() {
        match f {
            Field::Bytes(b) => assert_eq!(b.data(), regions[i], "region row {i}"),
            Field::Str(s) => assert_eq!(s.as_bytes(), regions[i], "region row {i}"),
            other => panic!("region row {i}: unexpected {other:?}"),
        }
    }
}

#[test]
fn dict_plus_bloom_combined() {
    use parquet::data_type::ByteArray;
    use parquet::file::properties::ReaderProperties;
    use parquet::file::serialized_reader::ReadOptionsBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dict_bloom.parquet");

    let rows: usize = 512;
    let ids: Vec<i64> = (0..rows as i64).map(|i| i * 31).collect();
    let tags_dict: [&[u8]; 5] = [b"alpha", b"bravo", b"charlie", b"delta", b"echo"];
    let tags: Vec<&[u8]> = (0..rows).map(|i| tags_dict[i % 5]).collect();

    // i64 stays PLAIN with bloom; tag is DICT with bloom.
    write_table_with_dict_and_blooms_to_path(
        &path,
        &[
            ("id", ColumnData::I64(&ids)),
            ("tag", ColumnData::ByteArray(&tags)),
        ],
        CompressionCodec::Snappy,
        128,
        &[false, true],
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
    assert_eq!(reader.num_row_groups(), 4);

    for rg_ix in 0..4 {
        let rg = reader.get_row_group(rg_ix).unwrap();
        let lo = rg_ix * 128;
        let hi = (lo + 128).min(rows);

        let bf_id = rg.get_column_bloom_filter(0).expect("id bloom missing");
        for v in &ids[lo..hi] {
            assert!(bf_id.check(v), "rg {rg_ix} id {v} absent");
        }
        let bf_tag = rg.get_column_bloom_filter(1).expect("tag bloom missing");
        for &t in &tags[lo..hi] {
            let ba = ByteArray::from(t.to_vec());
            assert!(bf_tag.check(&ba), "rg {rg_ix} tag {t:?} absent");
        }
    }
}

#[test]
fn dict_on_bool_column_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("err.parquet");
    let flags = vec![true, false, true];
    let r = write_table_with_dict_to_path(
        &path,
        &[("flag", ColumnData::Bool(&flags))],
        CompressionCodec::Uncompressed,
        4,
        &[true],
    );
    assert!(r.is_err(), "BOOLEAN dict must be rejected");
}

#[test]
fn dict_per_column_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("err2.parquet");
    let ids = vec![1i32, 2, 3];
    let r = write_table_with_dict_to_path(
        &path,
        &[("id", ColumnData::I32(&ids))],
        CompressionCodec::Uncompressed,
        4,
        &[true, true], // length 2 for a 1-column table
    );
    assert!(r.is_err());
}
