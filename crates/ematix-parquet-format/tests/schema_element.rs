//! TDD pin for `SchemaElement`.
//!
//!   struct SchemaElement {
//!     1:  optional Type             type
//!     2:  optional i32              type_length
//!     3:  optional FieldRepetitionType repetition_type
//!     4:  required string           name
//!     5:  optional i32              num_children
//!     6:  optional ConvertedType    converted_type
//!     7:  optional i32              scale
//!     8:  optional i32              precision
//!     9:  optional i32              field_id
//!     10: optional LogicalType      logicalType
//!   }
//!
//! Parquet stores the schema as a depth-first flattened list of these,
//! using num_children to reconstruct the tree.

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_schema_element, DecimalType, IntType, LogicalType, SchemaElement, TimeUnit,
    TimestampType,
};
use ematix_parquet_format::types::{ConvertedType, FieldRepetitionType, ParquetType};

#[test]
fn schema_element_root_node_name_only() {
    // The root of a parquet schema is conventionally a group with just
    // a name + num_children.
    let bytes = CompactBuilder::new()
        .binary(4, b"schema")
        .i32_field(5, 16)
        .stop();
    let mut cur = Cursor::new(&bytes);
    let se = read_schema_element(&mut cur).unwrap();
    assert_eq!(se.name, b"schema");
    assert_eq!(se.num_children, Some(16));
    assert_eq!(se.column_type, None);
    assert_eq!(se.repetition_type, None);
    assert_eq!(se.logical_type, None);
}

#[test]
fn schema_element_primitive_int64_leaf() {
    // l_orderkey: BIGINT NOT NULL → INT64, REQUIRED
    let bytes = CompactBuilder::new()
        .enum_field(1, 2) // INT64
        .enum_field(3, 0) // REQUIRED
        .binary(4, b"l_orderkey")
        .i32_field(9, 1) // field_id = 1
        .stop();
    let mut cur = Cursor::new(&bytes);
    let se = read_schema_element(&mut cur).unwrap();
    assert_eq!(se.column_type, Some(ParquetType::Int64));
    assert_eq!(se.repetition_type, Some(FieldRepetitionType::Required));
    assert_eq!(se.name, b"l_orderkey");
    assert_eq!(se.field_id, Some(1));
}

#[test]
fn schema_element_with_converted_type_utf8_legacy_writer() {
    // Older writers used ConvertedType::UTF8 instead of LogicalType::String.
    let bytes = CompactBuilder::new()
        .enum_field(1, 6) // BYTE_ARRAY
        .enum_field(3, 1) // OPTIONAL
        .binary(4, b"l_returnflag")
        .enum_field(6, 0) // ConvertedType::UTF8
        .stop();
    let mut cur = Cursor::new(&bytes);
    let se = read_schema_element(&mut cur).unwrap();
    assert_eq!(se.column_type, Some(ParquetType::ByteArray));
    assert_eq!(se.converted_type, Some(ConvertedType::Utf8));
    assert_eq!(se.logical_type, None);
}

#[test]
fn schema_element_decimal_with_logical_type() {
    // l_extendedprice: DECIMAL(12, 2) over FIXED_LEN_BYTE_ARRAY(6).
    let decimal_inner = CompactBuilder::new()
        .i32_field(1, 2)  // scale
        .i32_field(2, 12) // precision
        .stop();
    let lt = CompactBuilder::new().struct_field(5, &decimal_inner).stop();

    let bytes = CompactBuilder::new()
        .enum_field(1, 7) // FIXED_LEN_BYTE_ARRAY
        .i32_field(2, 6)  // type_length = 6
        .enum_field(3, 0) // REQUIRED
        .binary(4, b"l_extendedprice")
        .enum_field(6, 5) // ConvertedType::DECIMAL (legacy)
        .i32_field(7, 2)  // scale
        .i32_field(8, 12) // precision
        .struct_field(10, &lt)
        .stop();

    let mut cur = Cursor::new(&bytes);
    let se = read_schema_element(&mut cur).unwrap();
    assert_eq!(se.type_length, Some(6));
    assert_eq!(se.scale, Some(2));
    assert_eq!(se.precision, Some(12));
    assert_eq!(se.converted_type, Some(ConvertedType::Decimal));
    assert_eq!(
        se.logical_type,
        Some(LogicalType::Decimal(DecimalType {
            scale: 2,
            precision: 12,
        }))
    );
}

#[test]
fn schema_element_timestamp_micros_utc() {
    // o_orderdate-like field stored as INT64 + LogicalType::Timestamp(micros, UTC=true).
    let unit_inner = CompactBuilder::empty_struct();
    let unit = CompactBuilder::new().struct_field(2, &unit_inner).stop();
    let ts_inner = CompactBuilder::new()
        .bool_field(1, true)
        .struct_field(2, &unit)
        .stop();
    let lt = CompactBuilder::new().struct_field(8, &ts_inner).stop();

    let bytes = CompactBuilder::new()
        .enum_field(1, 2) // INT64
        .enum_field(3, 1) // OPTIONAL
        .binary(4, b"o_orderdate")
        .struct_field(10, &lt)
        .stop();

    let se = read_schema_element(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(
        se.logical_type,
        Some(LogicalType::Timestamp(TimestampType {
            is_adjusted_to_utc: true,
            unit: TimeUnit::Micros,
        }))
    );
}

#[test]
fn schema_element_integer_logical_type() {
    // Writers may annotate INT32 columns with LogicalType::Integer
    // (e.g. UINT_32: bit_width=32, is_signed=false).
    let int_inner = CompactBuilder::new()
        .i8_field(1, 32)
        .bool_field(2, false)
        .stop();
    let lt = CompactBuilder::new().struct_field(10, &int_inner).stop();

    let bytes = CompactBuilder::new()
        .enum_field(1, 1) // INT32
        .enum_field(3, 0)
        .binary(4, b"counter")
        .struct_field(10, &lt)
        .stop();

    let se = read_schema_element(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(
        se.logical_type,
        Some(LogicalType::Integer(IntType {
            bit_width: 32,
            is_signed: false,
        }))
    );
}

#[test]
fn schema_element_missing_required_name() {
    let bytes = CompactBuilder::new()
        .enum_field(1, 1)
        .i32_field(5, 0)
        .stop();
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_schema_element(&mut cur),
        Err(FormatError::MissingRequiredField {
            struct_name: "SchemaElement",
            field_id: 4,
        })
    ));
}

#[test]
fn schema_element_default_state() {
    let se = SchemaElement::default();
    assert_eq!(se.name, b"");
    assert_eq!(se.column_type, None);
    assert_eq!(se.num_children, None);
}
