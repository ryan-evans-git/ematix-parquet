//! TDD pin for `FileMetaData` (the top-level Parquet footer struct)
//! and the `ColumnOrder` union.
//!
//!   union ColumnOrder { 1: TypeDefinedOrder TYPE_ORDER }
//!
//!   struct FileMetaData {
//!     1: required i32                    version
//!     2: required list<SchemaElement>    schema
//!     3: required i64                    num_rows
//!     4: required list<RowGroup>         row_groups
//!     5: optional list<KeyValue>         key_value_metadata
//!     6: optional string                 created_by
//!     7: optional list<ColumnOrder>      column_orders
//!     // 8/9 (encryption) deferred — strict error
//!   }

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_column_order, read_file_metadata, ColumnOrder, FileMetaData,
};

// Reusable minimal payloads.

fn root_schema_element_bytes(num_children: i32) -> Vec<u8> {
    CompactBuilder::new()
        .binary(4, b"schema")
        .i32_field(5, num_children)
        .stop()
}

fn primitive_schema_element_bytes(name: &[u8], parquet_type: i32) -> Vec<u8> {
    CompactBuilder::new()
        .enum_field(1, parquet_type)
        .enum_field(3, 1) // OPTIONAL
        .binary(4, name)
        .stop()
}

fn minimal_row_group_bytes(num_rows: i64) -> Vec<u8> {
    let cc = CompactBuilder::new().i64_field(2, 4).stop();
    CompactBuilder::new()
        .list_struct_field(1, &[cc])
        .i64_field(2, 1024)
        .i64_field(3, num_rows)
        .stop()
}

fn type_defined_column_order_bytes() -> Vec<u8> {
    let inner = CompactBuilder::empty_struct();
    CompactBuilder::new().struct_field(1, &inner).stop()
}

// ---- ColumnOrder union -----------------------------------------------------

#[test]
fn column_order_type_defined() {
    let bytes = type_defined_column_order_bytes();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_column_order(&mut cur).unwrap(),
        ColumnOrder::TypeDefinedOrder
    );
}

#[test]
fn column_order_empty_union_errors() {
    let bytes = [0x00u8];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_column_order(&mut cur),
        Err(FormatError::EmptyUnion {
            union_name: "ColumnOrder"
        })
    ));
}

#[test]
fn column_order_unknown_variant_id_errors() {
    let inner = CompactBuilder::empty_struct();
    let bytes = CompactBuilder::new().struct_field(2, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_column_order(&mut cur),
        Err(FormatError::UnknownStructField {
            struct_name: "ColumnOrder",
            field_id: 2,
        })
    ));
}

// ---- FileMetaData ----------------------------------------------------------

#[test]
fn file_metadata_required_fields_only() {
    let root = root_schema_element_bytes(2);
    let col1 = primitive_schema_element_bytes(b"l_orderkey", 2);
    let col2 = primitive_schema_element_bytes(b"l_partkey", 1);
    let rg = minimal_row_group_bytes(60_000);

    let bytes = CompactBuilder::new()
        .i32_field(1, 2) // version
        .list_struct_field(2, &[root, col1, col2])
        .i64_field(3, 60_000)
        .list_struct_field(4, &[rg])
        .stop();

    let mut cur = Cursor::new(&bytes);
    let md = read_file_metadata(&mut cur).unwrap();
    assert_eq!(md.version, 2);
    assert_eq!(md.schema.len(), 3);
    assert_eq!(md.schema[0].name, b"schema");
    assert_eq!(md.schema[0].num_children, Some(2));
    assert_eq!(md.schema[1].name, b"l_orderkey");
    assert_eq!(md.num_rows, 60_000);
    assert_eq!(md.row_groups.len(), 1);
    assert_eq!(md.row_groups[0].num_rows, 60_000);
    assert_eq!(md.created_by, None);
    assert_eq!(md.key_value_metadata, None);
    assert_eq!(md.column_orders, None);
}

#[test]
fn file_metadata_full_payload() {
    let root = root_schema_element_bytes(1);
    let col = primitive_schema_element_bytes(b"id", 1);
    let rg = minimal_row_group_bytes(100);
    let kv = CompactBuilder::new()
        .binary(1, b"writer.version")
        .binary(2, b"ematix-parquet 0.0.1")
        .stop();
    let co = type_defined_column_order_bytes();

    let bytes = CompactBuilder::new()
        .i32_field(1, 2)
        .list_struct_field(2, &[root, col])
        .i64_field(3, 100)
        .list_struct_field(4, &[rg])
        .list_struct_field(5, &[kv])
        .binary(6, b"ematix-parquet vX.Y.Z (build abcdef)")
        .list_struct_field(7, &[co])
        .stop();

    let mut cur = Cursor::new(&bytes);
    let md = read_file_metadata(&mut cur).unwrap();
    assert_eq!(
        md.created_by,
        Some(&b"ematix-parquet vX.Y.Z (build abcdef)"[..])
    );
    let kv = md.key_value_metadata.unwrap();
    assert_eq!(kv.len(), 1);
    assert_eq!(kv[0].key, b"writer.version");
    assert_eq!(kv[0].value, Some(&b"ematix-parquet 0.0.1"[..]));
    let cos = md.column_orders.unwrap();
    assert_eq!(cos.len(), 1);
    assert_eq!(cos[0], ColumnOrder::TypeDefinedOrder);
}

#[test]
fn file_metadata_missing_required_version() {
    let root = root_schema_element_bytes(0);
    let rg = minimal_row_group_bytes(1);
    let bytes = CompactBuilder::new()
        .list_struct_field(2, &[root])
        .i64_field(3, 1)
        .list_struct_field(4, &[rg])
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_file_metadata(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "FileMetaData",
            field_id: 1,
        })
    ));
}

#[test]
fn file_metadata_missing_required_schema() {
    let rg = minimal_row_group_bytes(1);
    let bytes = CompactBuilder::new()
        .i32_field(1, 2)
        .i64_field(3, 1)
        .list_struct_field(4, &[rg])
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_file_metadata(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "FileMetaData",
            field_id: 2,
        })
    ));
}

#[test]
fn file_metadata_encryption_fields_errored_strictly() {
    // Field 8 (encryption_algorithm union) not yet supported.
    let root = root_schema_element_bytes(0);
    let rg = minimal_row_group_bytes(1);
    let enc = CompactBuilder::empty_struct();
    let bytes = CompactBuilder::new()
        .i32_field(1, 2)
        .list_struct_field(2, &[root])
        .i64_field(3, 1)
        .list_struct_field(4, &[rg])
        .struct_field(8, &enc)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_file_metadata(&mut cur),
        Err(FormatError::UnknownStructField {
            struct_name: "FileMetaData",
            field_id: 8,
        })
    ));
}

#[test]
fn file_metadata_default_state() {
    let md = FileMetaData::default();
    assert_eq!(md.version, 0);
    assert_eq!(md.num_rows, 0);
    assert!(md.schema.is_empty());
    assert!(md.row_groups.is_empty());
}
