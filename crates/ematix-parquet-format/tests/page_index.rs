//! TDD pin for the page-index types: `PageLocation`, `OffsetIndex`,
//! `ColumnIndex`. These are stored at byte offsets recorded in
//! `ColumnChunk.{offset,column}_index_{offset,length}` and are the
//! mechanism by which row-group page-skipping works (the lever we
//! already probed empirically on the ematix-flow side — see
//! ematix-flow's `parquet_page_index_probe.rs` example).

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::{read_list_bool, Cursor, FieldType};
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_column_index, read_offset_index, read_page_location, ColumnIndex, OffsetIndex,
    PageLocation,
};
use ematix_parquet_format::types::BoundaryOrder;

// ---- read_list_bool primitive ---------------------------------------------

#[test]
fn read_list_bool_empty() {
    // count=0, elem_type=2 → 0x02
    let bytes = [0x02u8];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_list_bool(&mut cur).unwrap(), Vec::<bool>::new());
}

#[test]
fn read_list_bool_mixed() {
    // count=4, elem_type=2 → 0x42, then [true, false, true, true]
    let bytes = [0x42u8, 0x01, 0x02, 0x01, 0x01];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_list_bool(&mut cur).unwrap(),
        vec![true, false, true, true]
    );
}

#[test]
fn read_list_bool_accepts_both_canonical_type_codes() {
    // Some thrift writers emit type code 1 (BoolTrue) instead of 2 in
    // the list header. Per spec both are valid for boolean lists.
    let bytes = [0x21u8, 0x01, 0x02]; // count=2, type=1
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_list_bool(&mut cur).unwrap(), vec![true, false]);
}

#[test]
fn read_list_bool_rejects_invalid_value_byte() {
    let bytes = [0x12u8, 0x03]; // count=1, elem_type=2, value byte = 0x03 invalid
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_list_bool(&mut cur),
        Err(FormatError::InvalidBoolByte(0x03))
    ));
}

#[test]
fn read_list_bool_rejects_non_bool_elem_type() {
    // count=1, elem_type=5 (I32) — invalid for a list<bool>.
    let bytes = [0x15u8, 0x00];
    let mut cur = Cursor::new(&bytes);
    let err = read_list_bool(&mut cur).unwrap_err();
    match err {
        FormatError::UnexpectedListElementType { actual, .. } => {
            assert_eq!(actual, FieldType::I32);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ---- PageLocation ---------------------------------------------------------

#[test]
fn page_location_round_trip() {
    let bytes = CompactBuilder::new()
        .i64_field(1, 1024)
        .i32_field(2, 8192)
        .i64_field(3, 500)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_page_location(&mut cur).unwrap(),
        PageLocation {
            offset: 1024,
            compressed_page_size: 8192,
            first_row_index: 500,
        }
    );
}

#[test]
fn page_location_missing_required_compressed_size() {
    let bytes = CompactBuilder::new()
        .i64_field(1, 1024)
        .i64_field(3, 0)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_page_location(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "PageLocation",
            field_id: 2,
        })
    ));
}

// ---- OffsetIndex ----------------------------------------------------------

#[test]
fn offset_index_three_pages() {
    let p1 = CompactBuilder::new()
        .i64_field(1, 0)
        .i32_field(2, 1024)
        .i64_field(3, 0)
        .stop();
    let p2 = CompactBuilder::new()
        .i64_field(1, 1024)
        .i32_field(2, 2048)
        .i64_field(3, 100)
        .stop();
    let p3 = CompactBuilder::new()
        .i64_field(1, 3072)
        .i32_field(2, 512)
        .i64_field(3, 250)
        .stop();
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[p1, p2, p3])
        .stop();
    let mut cur = Cursor::new(&bytes);
    let oi = read_offset_index(&mut cur).unwrap();
    assert_eq!(oi.page_locations.len(), 3);
    assert_eq!(oi.page_locations[0].offset, 0);
    assert_eq!(oi.page_locations[1].first_row_index, 100);
    assert_eq!(oi.page_locations[2].compressed_page_size, 512);
    assert_eq!(oi.unencoded_byte_array_data_bytes, None);
}

#[test]
fn offset_index_with_unencoded_bytes_per_page() {
    let p1 = CompactBuilder::new()
        .i64_field(1, 0)
        .i32_field(2, 100)
        .i64_field(3, 0)
        .stop();
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[p1])
        .list_i64_field(2, &[1234])
        .stop();
    let mut cur = Cursor::new(&bytes);
    let oi = read_offset_index(&mut cur).unwrap();
    assert_eq!(oi.unencoded_byte_array_data_bytes, Some(vec![1234]));
}

// ---- ColumnIndex ----------------------------------------------------------

#[test]
fn column_index_3_pages_ascending() {
    // 3 pages, none all-null. Per spec, null_pages[i]=true means
    // min/max for page i are not meaningful.
    let bytes = CompactBuilder::new()
        .list_bool_field(1, &[false, false, false])
        .list_binary_field(2, &[&[0x00, 0x00], &[0x01, 0x00], &[0x02, 0x00]])
        .list_binary_field(3, &[&[0x00, 0xFF], &[0x01, 0xFF], &[0x02, 0xFF]])
        .enum_field(4, 1) // BoundaryOrder::Ascending
        .stop();
    let mut cur = Cursor::new(&bytes);
    let ci = read_column_index(&mut cur).unwrap();
    assert_eq!(ci.null_pages, vec![false, false, false]);
    assert_eq!(ci.min_values.len(), 3);
    assert_eq!(ci.max_values.len(), 3);
    assert_eq!(ci.min_values[1], &[0x01, 0x00][..]);
    assert_eq!(ci.boundary_order, BoundaryOrder::Ascending);
    assert_eq!(ci.null_counts, None);
}

#[test]
fn column_index_with_null_page_and_null_counts() {
    let bytes = CompactBuilder::new()
        .list_bool_field(1, &[false, true, false])
        .list_binary_field(2, &[&[0x00], &[], &[0x02]])
        .list_binary_field(3, &[&[0x10], &[], &[0x20]])
        .enum_field(4, 0) // Unordered
        .list_i64_field(5, &[0, 100, 5])
        .stop();
    let mut cur = Cursor::new(&bytes);
    let ci = read_column_index(&mut cur).unwrap();
    assert_eq!(ci.null_pages, vec![false, true, false]);
    assert_eq!(ci.boundary_order, BoundaryOrder::Unordered);
    assert_eq!(ci.null_counts, Some(vec![0, 100, 5]));
}

#[test]
fn column_index_missing_required_min_values() {
    let bytes = CompactBuilder::new()
        .list_bool_field(1, &[false])
        .list_binary_field(3, &[&[0x00]])
        .enum_field(4, 0)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_column_index(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "ColumnIndex",
            field_id: 2,
        })
    ));
}
