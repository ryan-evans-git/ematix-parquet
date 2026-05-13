//! TDD pin for `SortingColumn` and `RowGroup`.
//!
//!   struct SortingColumn {
//!     1: required i32  column_idx
//!     2: required bool descending
//!     3: required bool nulls_first
//!   }
//!
//!   struct RowGroup {
//!     1: required list<ColumnChunk> columns
//!     2: required i64               total_byte_size
//!     3: required i64               num_rows
//!     4: optional list<SortingColumn> sorting_columns
//!     5: optional i64               file_offset
//!     6: optional i64               total_compressed_size
//!     7: optional i16               ordinal
//!   }

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::{read_zigzag_i16, Cursor};
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_row_group, read_sorting_column, RowGroup, SortingColumn,
};

// ---- read_zigzag_i16 -------------------------------------------------------

#[test]
fn zigzag_i16_boundary_values() {
    fn encode(v: i16) -> Vec<u8> {
        let mut u = (((v as i32) << 1) ^ ((v as i32) >> 31)) as u32 as u64;
        let mut out = vec![];
        loop {
            if u < 0x80 {
                out.push(u as u8);
                return out;
            }
            out.push(((u & 0x7F) | 0x80) as u8);
            u >>= 7;
        }
    }
    for v in [0i16, 1, -1, 100, -100, i16::MAX, i16::MIN] {
        let bytes = encode(v);
        let mut cur = Cursor::new(&bytes);
        assert_eq!(read_zigzag_i16(&mut cur).unwrap(), v);
    }
}

// ---- SortingColumn ---------------------------------------------------------

#[test]
fn sorting_column_round_trip() {
    let bytes = CompactBuilder::new()
        .i32_field(1, 3)
        .bool_field(2, true)
        .bool_field(3, false)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_sorting_column(&mut cur).unwrap(),
        SortingColumn {
            column_idx: 3,
            descending: true,
            nulls_first: false,
        }
    );
}

#[test]
fn sorting_column_missing_required_descending() {
    let bytes = CompactBuilder::new()
        .i32_field(1, 0)
        .bool_field(3, false)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_sorting_column(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "SortingColumn",
            field_id: 2,
        })
    ));
}

// ---- RowGroup --------------------------------------------------------------

/// Build a minimal ColumnChunk payload for testing the wrapping list.
fn minimal_column_chunk_bytes(file_offset: i64) -> Vec<u8> {
    CompactBuilder::new().i64_field(2, file_offset).stop()
}

#[test]
fn row_group_required_fields_only() {
    let cc1 = minimal_column_chunk_bytes(100);
    let cc2 = minimal_column_chunk_bytes(200);
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[cc1, cc2])
        .i64_field(2, 10_000_000)
        .i64_field(3, 60_000)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let rg = read_row_group(&mut cur).unwrap();
    assert_eq!(rg.columns.len(), 2);
    assert_eq!(rg.columns[0].file_offset, 100);
    assert_eq!(rg.columns[1].file_offset, 200);
    assert_eq!(rg.total_byte_size, 10_000_000);
    assert_eq!(rg.num_rows, 60_000);
    assert_eq!(rg.sorting_columns, None);
    assert_eq!(rg.file_offset, None);
    assert_eq!(rg.total_compressed_size, None);
    assert_eq!(rg.ordinal, None);
}

#[test]
fn row_group_with_sorting_columns_and_file_offset() {
    let cc = minimal_column_chunk_bytes(4);
    let sc = CompactBuilder::new()
        .i32_field(1, 0)
        .bool_field(2, false)
        .bool_field(3, true)
        .stop();
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[cc])
        .i64_field(2, 1024)
        .i64_field(3, 100)
        .list_struct_field(4, &[sc])
        .i64_field(5, 1_000_000)
        .i64_field(6, 800_000)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let rg = read_row_group(&mut cur).unwrap();
    let scs = rg.sorting_columns.unwrap();
    assert_eq!(scs.len(), 1);
    assert_eq!(scs[0].column_idx, 0);
    assert_eq!(scs[0].nulls_first, true);
    assert_eq!(rg.file_offset, Some(1_000_000));
    assert_eq!(rg.total_compressed_size, Some(800_000));
}

#[test]
fn row_group_ordinal_i16() {
    let cc = minimal_column_chunk_bytes(0);
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[cc])
        .i64_field(2, 0)
        .i64_field(3, 0)
        .i16_field(7, 42)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let rg = read_row_group(&mut cur).unwrap();
    assert_eq!(rg.ordinal, Some(42));
}

#[test]
fn row_group_missing_required_num_rows() {
    let cc = minimal_column_chunk_bytes(0);
    let bytes = CompactBuilder::new()
        .list_struct_field(1, &[cc])
        .i64_field(2, 1)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_row_group(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "RowGroup",
            field_id: 3,
        })
    ));
}

#[test]
fn row_group_default_state() {
    let rg = RowGroup::default();
    assert_eq!(rg.num_rows, 0);
    assert!(rg.columns.is_empty());
}
