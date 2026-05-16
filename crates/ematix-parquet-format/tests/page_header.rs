//! TDD pin for `PageHeader` and its four nested page-kind structs.
//!
//! From parquet.thrift:
//!   struct PageHeader {
//!     1: required PageType  type;
//!     2: required i32       uncompressed_page_size;
//!     3: required i32       compressed_page_size;
//!     4: optional i32       crc;
//!     5: optional DataPageHeader        data_page_header;
//!     6: optional IndexPageHeader       index_page_header;
//!     7: optional DictionaryPageHeader  dictionary_page_header;
//!     8: optional DataPageHeaderV2      data_page_header_v2;
//!   }
//!
//! And the four payload structs (see metadata.rs for full layout).
//!
//! Notable wrinkle: `DataPageHeaderV2.is_compressed` has spec default
//! `true`, so absence means `true`, not `None`. We model it as plain
//! `bool` (not `Option<bool>`) for that reason.

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_page_header, DataPageHeader, DataPageHeaderV2, DictionaryPageHeader, PageHeader,
};
use ematix_parquet_format::types::{Encoding, PageType};

// ---- DictionaryPageHeader ---------------------------------------------------

#[test]
fn dictionary_page_header_basic_no_is_sorted() {
    // num_values=100, encoding=PLAIN_DICTIONARY(2)
    let inner = CompactBuilder::new()
        .i32_field(1, 100)
        .enum_field(2, 2)
        .stop();
    // Wrap in a PageHeader: type=DICTIONARY_PAGE(2), uncompressed=4096,
    // compressed=2048, dictionary_page_header at field 7.
    let bytes = CompactBuilder::new()
        .enum_field(1, 2)
        .i32_field(2, 4096)
        .i32_field(3, 2048)
        .struct_field(7, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    assert_eq!(ph.page_type, PageType::DictionaryPage);
    assert_eq!(ph.uncompressed_page_size, 4096);
    assert_eq!(ph.compressed_page_size, 2048);
    assert!(ph.data_page_header.is_none());
    assert!(ph.data_page_header_v2.is_none());
    let dph = ph
        .dictionary_page_header
        .expect("dictionary header present");
    assert_eq!(dph.num_values, 100);
    assert_eq!(dph.encoding, Encoding::PlainDictionary);
    assert_eq!(dph.is_sorted, None);
}

#[test]
fn dictionary_page_header_is_sorted_true() {
    let inner = CompactBuilder::new()
        .i32_field(1, 50)
        .enum_field(2, 8) // RLE_DICTIONARY
        .bool_field(3, true)
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 2)
        .i32_field(2, 100)
        .i32_field(3, 80)
        .struct_field(7, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    let dph = ph.dictionary_page_header.unwrap();
    assert_eq!(dph.is_sorted, Some(true));
    assert_eq!(dph.encoding, Encoding::RleDictionary);
}

// ---- DataPageHeader ---------------------------------------------------------

#[test]
fn data_page_header_v1_without_statistics() {
    // num_values=1000, encoding=PLAIN(0), def_level_enc=RLE(3),
    // rep_level_enc=RLE(3).
    let inner = CompactBuilder::new()
        .i32_field(1, 1000)
        .enum_field(2, 0)
        .enum_field(3, 3)
        .enum_field(4, 3)
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 0) // DATA_PAGE
        .i32_field(2, 8192)
        .i32_field(3, 4096)
        .struct_field(5, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    assert_eq!(ph.page_type, PageType::DataPage);
    let dph = ph.data_page_header.unwrap();
    assert_eq!(dph.num_values, 1000);
    assert_eq!(dph.encoding, Encoding::Plain);
    assert_eq!(dph.definition_level_encoding, Encoding::Rle);
    assert_eq!(dph.repetition_level_encoding, Encoding::Rle);
    assert!(dph.statistics.is_none());
}

#[test]
fn data_page_header_v1_with_nested_statistics() {
    // Inner stats: null_count=7
    let stats = CompactBuilder::new().i64_field(3, 7).stop();
    let inner = CompactBuilder::new()
        .i32_field(1, 2000)
        .enum_field(2, 8) // RLE_DICTIONARY
        .enum_field(3, 3) // RLE
        .enum_field(4, 3)
        .struct_field(5, &stats)
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 0)
        .i32_field(2, 16384)
        .i32_field(3, 8192)
        .struct_field(5, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    let dph = ph.data_page_header.unwrap();
    let s = dph.statistics.expect("nested statistics present");
    assert_eq!(s.null_count, Some(7));
}

// ---- DataPageHeaderV2 -------------------------------------------------------

#[test]
fn data_page_header_v2_default_is_compressed_when_absent() {
    // No field 7 → is_compressed should default to true per spec.
    let inner = CompactBuilder::new()
        .i32_field(1, 500)
        .i32_field(2, 50)
        .i32_field(3, 450)
        .enum_field(4, 0)
        .i32_field(5, 32)
        .i32_field(6, 16)
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 3) // DATA_PAGE_V2
        .i32_field(2, 1024)
        .i32_field(3, 1024)
        .struct_field(8, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    let v2 = ph.data_page_header_v2.unwrap();
    assert_eq!(v2.num_values, 500);
    assert_eq!(v2.num_nulls, 50);
    assert_eq!(v2.num_rows, 450);
    assert_eq!(v2.encoding, Encoding::Plain);
    assert_eq!(v2.definition_levels_byte_length, 32);
    assert_eq!(v2.repetition_levels_byte_length, 16);
    assert!(
        v2.is_compressed,
        "spec default for is_compressed is true when field is absent"
    );
}

#[test]
fn data_page_header_v2_is_compressed_explicit_false() {
    let inner = CompactBuilder::new()
        .i32_field(1, 10)
        .i32_field(2, 0)
        .i32_field(3, 10)
        .enum_field(4, 0)
        .i32_field(5, 0)
        .i32_field(6, 0)
        .bool_field(7, false)
        .stop();
    let bytes = CompactBuilder::new()
        .enum_field(1, 3)
        .i32_field(2, 256)
        .i32_field(3, 256)
        .struct_field(8, &inner)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let v2 = read_page_header(&mut cur)
        .unwrap()
        .data_page_header_v2
        .unwrap();
    assert!(!v2.is_compressed);
}

// ---- PageHeader required fields --------------------------------------------

#[test]
fn missing_required_page_type_errors() {
    // No field 1 (type), but page sizes present.
    let bytes = CompactBuilder::new()
        .i32_field(2, 100)
        .i32_field(3, 50)
        .stop();
    let mut cur = Cursor::new(&bytes);
    match read_page_header(&mut cur) {
        Err(FormatError::MissingRequiredField {
            struct_name: "PageHeader",
            field_id: 1,
        }) => {}
        other => panic!("expected MissingRequiredField id=1, got {other:?}"),
    }
}

#[test]
fn missing_required_page_sizes_errors() {
    let bytes = CompactBuilder::new().enum_field(1, 0).stop();
    let mut cur = Cursor::new(&bytes);
    match read_page_header(&mut cur) {
        Err(FormatError::MissingRequiredField { field_id: 2, .. }) => {}
        other => panic!("expected MissingRequiredField id=2, got {other:?}"),
    }
}

#[test]
fn optional_crc_field() {
    let bytes = CompactBuilder::new()
        .enum_field(1, 0)
        .i32_field(2, 100)
        .i32_field(3, 100)
        .i32_field(4, 0x12345678)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    assert_eq!(ph.crc, Some(0x12345678));
}

// ---- Cross-construction: nested types compose via metadata.rs API ----------

#[test]
fn page_header_has_minimal_required_only() {
    let bytes = CompactBuilder::new()
        .enum_field(1, 1) // INDEX_PAGE (Parquet's "TODO" page kind)
        .i32_field(2, 0)
        .i32_field(3, 0)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let ph = read_page_header(&mut cur).unwrap();
    assert_eq!(
        ph,
        PageHeader {
            page_type: PageType::IndexPage,
            uncompressed_page_size: 0,
            compressed_page_size: 0,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        }
    );
    // Type checks: confirm struct-typed fields compose.
    let _: Option<DataPageHeader<'_>> = ph.data_page_header;
    let _: Option<DataPageHeaderV2<'_>> = ph.data_page_header_v2;
    let _: Option<DictionaryPageHeader> = ph.dictionary_page_header;
}
