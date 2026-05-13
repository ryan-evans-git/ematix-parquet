//! TDD pin for `PageEncodingStats`, `ColumnMetaData`, `ColumnChunk`.
//!
//! These are the first structs in the suite that exercise lists.
//!   PageEncodingStats { page_type, encoding, count }
//!   ColumnMetaData    { type, list<Encoding>, list<string>, codec,
//!                       num_values, ..., list<PageEncodingStats>, ... }
//!   ColumnChunk       { file_path?, file_offset, ColumnMetaData?,
//!                       offset/column index offsets+lengths }

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_column_chunk, read_column_metadata, read_page_encoding_stats, ColumnChunk,
    PageEncodingStats,
};
use ematix_parquet_format::types::{CompressionCodec, Encoding, PageType, ParquetType};

// ---- PageEncodingStats ------------------------------------------------------

#[test]
fn page_encoding_stats_round_trip() {
    let bytes = CompactBuilder::new()
        .enum_field(1, 0) // DATA_PAGE
        .enum_field(2, 2) // PLAIN_DICTIONARY
        .i32_field(3, 42)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_page_encoding_stats(&mut cur).unwrap(),
        PageEncodingStats {
            page_type: PageType::DataPage,
            encoding: Encoding::PlainDictionary,
            count: 42,
        }
    );
}

#[test]
fn page_encoding_stats_missing_required() {
    let bytes = CompactBuilder::new().enum_field(1, 0).stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_page_encoding_stats(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "PageEncodingStats",
            field_id: 2,
        })
    ));
}

// ---- ColumnMetaData: minimal required-only --------------------------------

#[test]
fn column_metadata_minimal_required_fields_only() {
    // type=INT64(2), encodings=[PLAIN(0), RLE(3)], path=["foo","bar"],
    // codec=SNAPPY(1), num_values=10000,
    // total_uncompressed=80000, total_compressed=40000,
    // data_page_offset=4
    let bytes = CompactBuilder::new()
        .enum_field(1, 2)
        .list_i32_field(2, &[0, 3])
        .list_binary_field(3, &[b"foo", b"bar"])
        .enum_field(4, 1)
        .i64_field(5, 10_000)
        .i64_field(6, 80_000)
        .i64_field(7, 40_000)
        .i64_field(9, 4)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let m = read_column_metadata(&mut cur).unwrap();
    assert_eq!(m.column_type, ParquetType::Int64);
    assert_eq!(m.encodings, vec![Encoding::Plain, Encoding::Rle]);
    assert_eq!(m.path_in_schema, vec![&b"foo"[..], &b"bar"[..]]);
    assert_eq!(m.codec, CompressionCodec::Snappy);
    assert_eq!(m.num_values, 10_000);
    assert_eq!(m.total_uncompressed_size, 80_000);
    assert_eq!(m.total_compressed_size, 40_000);
    assert_eq!(m.data_page_offset, 4);
    assert_eq!(m.key_value_metadata, None);
    assert_eq!(m.index_page_offset, None);
    assert_eq!(m.dictionary_page_offset, None);
    assert_eq!(m.statistics, None);
    assert_eq!(m.encoding_stats, None);
    assert_eq!(m.bloom_filter_offset, None);
    assert_eq!(m.bloom_filter_length, None);
}

#[test]
fn column_metadata_with_encoding_stats_list() {
    let pes_elem = CompactBuilder::new()
        .enum_field(1, 0) // DATA_PAGE
        .enum_field(2, 0) // PLAIN
        .i32_field(3, 5)
        .stop();

    let bytes = CompactBuilder::new()
        .enum_field(1, 2)
        .list_i32_field(2, &[0])
        .list_binary_field(3, &[b"col"])
        .enum_field(4, 0) // UNCOMPRESSED
        .i64_field(5, 100)
        .i64_field(6, 800)
        .i64_field(7, 800)
        .i64_field(9, 0)
        .list_struct_field(13, &[pes_elem])
        .stop();

    let mut cur = Cursor::new(&bytes);
    let m = read_column_metadata(&mut cur).unwrap();
    let stats = m.encoding_stats.unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].page_type, PageType::DataPage);
    assert_eq!(stats[0].encoding, Encoding::Plain);
    assert_eq!(stats[0].count, 5);
}

#[test]
fn column_metadata_with_nested_statistics() {
    let stats = CompactBuilder::new().i64_field(3, 99).stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 5) // DOUBLE
        .list_i32_field(2, &[0])
        .list_binary_field(3, &[b"x"])
        .enum_field(4, 0)
        .i64_field(5, 1)
        .i64_field(6, 8)
        .i64_field(7, 8)
        .i64_field(9, 16)
        .struct_field(12, &stats)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let m = read_column_metadata(&mut cur).unwrap();
    assert_eq!(m.statistics.unwrap().null_count, Some(99));
}

#[test]
fn column_metadata_with_key_value_metadata_and_bloom() {
    let kv = CompactBuilder::new()
        .binary(1, b"writer")
        .binary(2, b"ematix")
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 1) // INT32
        .list_i32_field(2, &[0])
        .list_binary_field(3, &[b"id"])
        .enum_field(4, 6) // ZSTD
        .i64_field(5, 50)
        .i64_field(6, 200)
        .i64_field(7, 80)
        .list_struct_field(8, &[kv])
        .i64_field(9, 4)
        .i64_field(10, 1024)
        .i64_field(11, 2048)
        .i64_field(14, 9999)
        .i32_field(15, 128)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let m = read_column_metadata(&mut cur).unwrap();
    assert_eq!(m.codec, CompressionCodec::Zstd);
    assert_eq!(m.index_page_offset, Some(1024));
    assert_eq!(m.dictionary_page_offset, Some(2048));
    assert_eq!(m.bloom_filter_offset, Some(9999));
    assert_eq!(m.bloom_filter_length, Some(128));
    let kv = m.key_value_metadata.unwrap();
    assert_eq!(kv.len(), 1);
    assert_eq!(kv[0].key, b"writer");
    assert_eq!(kv[0].value, Some(&b"ematix"[..]));
}

#[test]
fn column_metadata_missing_required_type() {
    // Skip field 1 (type), keep the rest.
    let bytes = CompactBuilder::new()
        .list_i32_field(2, &[0])
        .list_binary_field(3, &[b"x"])
        .enum_field(4, 0)
        .i64_field(5, 1)
        .i64_field(6, 1)
        .i64_field(7, 1)
        .i64_field(9, 0)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_column_metadata(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "ColumnMetaData",
            field_id: 1,
        })
    ));
}

// ---- ColumnChunk wrapper ----------------------------------------------------

#[test]
fn column_chunk_with_inline_metadata() {
    let meta = CompactBuilder::new()
        .enum_field(1, 2) // INT64
        .list_i32_field(2, &[0])
        .list_binary_field(3, &[b"x"])
        .enum_field(4, 0)
        .i64_field(5, 10)
        .i64_field(6, 80)
        .i64_field(7, 80)
        .i64_field(9, 4)
        .stop();
    let bytes = CompactBuilder::new()
        .i64_field(2, 12345)
        .struct_field(3, &meta)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let cc = read_column_chunk(&mut cur).unwrap();
    assert_eq!(cc.file_path, None);
    assert_eq!(cc.file_offset, 12345);
    let m = cc.meta_data.unwrap();
    assert_eq!(m.column_type, ParquetType::Int64);
    assert_eq!(m.num_values, 10);
}

#[test]
fn column_chunk_with_file_path_and_index_offsets() {
    let bytes = CompactBuilder::new()
        .binary(1, b"part-0.parquet")
        .i64_field(2, 555)
        .i64_field(4, 100) // offset_index_offset
        .i32_field(5, 50)  // offset_index_length
        .i64_field(6, 200) // column_index_offset
        .i32_field(7, 75)  // column_index_length
        .stop();
    let mut cur = Cursor::new(&bytes);
    let cc = read_column_chunk(&mut cur).unwrap();
    assert_eq!(cc.file_path, Some(&b"part-0.parquet"[..]));
    assert_eq!(cc.file_offset, 555);
    assert_eq!(cc.meta_data, None);
    assert_eq!(cc.offset_index_offset, Some(100));
    assert_eq!(cc.offset_index_length, Some(50));
    assert_eq!(cc.column_index_offset, Some(200));
    assert_eq!(cc.column_index_length, Some(75));
}

#[test]
fn column_chunk_missing_required_file_offset() {
    let bytes = CompactBuilder::new().binary(1, b"x").stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_column_chunk(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "ColumnChunk",
            field_id: 2,
        })
    ));
}

#[test]
fn empty_column_chunk_default_state() {
    // Just to confirm Default derive lines up.
    let cc = ColumnChunk::default();
    assert_eq!(cc.file_offset, 0);
    assert_eq!(cc.file_path, None);
}
