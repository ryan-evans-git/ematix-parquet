//! TDD pin for thrift union decoding, `TimeUnit`, and `LogicalType`.
//!
//! Thrift unions are wire-identical to structs but the reader enforces
//! that exactly one field is set. Empty payload types (StringType,
//! UUIDType, …) are empty nested structs.

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::{read_i8, Cursor};
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{
    read_logical_type, read_time_unit, DecimalType, GeographyType, GeometryType, IntType,
    LogicalType, TimeType, TimeUnit, TimestampType, VariantType,
};
use ematix_parquet_format::types::{EdgeInterpolationAlgorithm, ThriftEnum};

// ---- i8 raw-byte reader ----------------------------------------------------

#[test]
fn read_i8_positive() {
    let bytes = [42u8];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_i8(&mut cur).unwrap(), 42);
}

#[test]
fn read_i8_negative() {
    let bytes = [0xFFu8]; // -1 as i8
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_i8(&mut cur).unwrap(), -1);
}

#[test]
fn read_i8_boundary_values() {
    let mut cur = Cursor::new(&[0x80u8]); // i8::MIN
    assert_eq!(read_i8(&mut cur).unwrap(), i8::MIN);
    let mut cur = Cursor::new(&[0x7Fu8]); // i8::MAX
    assert_eq!(read_i8(&mut cur).unwrap(), i8::MAX);
}

// ---- EdgeInterpolationAlgorithm enum ---------------------------------------

#[test]
fn edge_interpolation_algorithm_variants() {
    fn zz(v: i32) -> Vec<u8> {
        let mut u = ((v << 1) ^ (v >> 31)) as u32;
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
    use EdgeInterpolationAlgorithm::*;
    for (v, expected) in [
        (0, Spherical),
        (1, Vincenty),
        (2, Thomas),
        (3, Andoyer),
        (4, Karney),
    ] {
        let bytes = zz(v);
        let mut cur = Cursor::new(&bytes);
        assert_eq!(EdgeInterpolationAlgorithm::read(&mut cur).unwrap(), expected);
    }
}

// ---- TimeUnit union --------------------------------------------------------

fn make_union(variant_id: i16) -> Vec<u8> {
    let inner = CompactBuilder::empty_struct();
    CompactBuilder::new().struct_field(variant_id, &inner).stop()
}

#[test]
fn time_unit_millis() {
    let bytes = make_union(1);
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_time_unit(&mut cur).unwrap(), TimeUnit::Millis);
}

#[test]
fn time_unit_micros() {
    let bytes = make_union(2);
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_time_unit(&mut cur).unwrap(), TimeUnit::Micros);
}

#[test]
fn time_unit_nanos() {
    let bytes = make_union(3);
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_time_unit(&mut cur).unwrap(), TimeUnit::Nanos);
}

#[test]
fn time_unit_empty_is_error() {
    // No variant set — union must reject.
    let bytes = [0x00u8];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_time_unit(&mut cur),
        Err(FormatError::EmptyUnion {
            union_name: "TimeUnit"
        })
    ));
}

#[test]
fn time_unit_unknown_variant_id_errors() {
    let bytes = make_union(99);
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_time_unit(&mut cur),
        Err(FormatError::UnknownStructField {
            struct_name: "TimeUnit",
            field_id: 99,
        })
    ));
}

// ---- LogicalType empty-marker variants -------------------------------------

#[test]
fn logical_type_string() {
    let bytes = make_union(1);
    assert_eq!(
        read_logical_type(&mut Cursor::new(&bytes)).unwrap(),
        LogicalType::String
    );
}

#[test]
fn logical_type_uuid_and_float16_and_null() {
    // Just a smoke pass on the variants that share the empty-payload
    // shape: UUID(14), Float16(15), Null=UNKNOWN(11), Json(12),
    // Bson(13), Map(2), List(3), Enum(4), Date(6).
    let cases: &[(i16, LogicalType)] = &[
        (2, LogicalType::Map),
        (3, LogicalType::List),
        (4, LogicalType::Enum),
        (6, LogicalType::Date),
        (11, LogicalType::Null),
        (12, LogicalType::Json),
        (13, LogicalType::Bson),
        (14, LogicalType::Uuid),
        (15, LogicalType::Float16),
    ];
    for (vid, expected) in cases {
        let bytes = make_union(*vid);
        let got = read_logical_type(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(got, *expected, "variant id {vid} decoded wrong");
    }
}

// ---- LogicalType with payload structs --------------------------------------

#[test]
fn logical_type_decimal() {
    let inner = CompactBuilder::new()
        .i32_field(1, 2)
        .i32_field(2, 18)
        .stop();
    let bytes = CompactBuilder::new().struct_field(5, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_logical_type(&mut cur).unwrap(),
        LogicalType::Decimal(DecimalType {
            scale: 2,
            precision: 18,
        })
    );
}

#[test]
fn logical_type_integer() {
    // IntType { bitWidth: 32 (i8), isSigned: true }
    let inner = CompactBuilder::new()
        .i8_field(1, 32)
        .bool_field(2, true)
        .stop();
    let bytes = CompactBuilder::new().struct_field(10, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_logical_type(&mut cur).unwrap(),
        LogicalType::Integer(IntType {
            bit_width: 32,
            is_signed: true,
        })
    );
}

#[test]
fn logical_type_timestamp_with_nested_union() {
    // TimestampType { isAdjustedToUTC=true, unit=MICROS }
    let unit_inner = CompactBuilder::empty_struct();
    let unit = CompactBuilder::new().struct_field(2, &unit_inner).stop();
    let inner = CompactBuilder::new()
        .bool_field(1, true)
        .struct_field(2, &unit)
        .stop();
    let bytes = CompactBuilder::new().struct_field(8, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_logical_type(&mut cur).unwrap(),
        LogicalType::Timestamp(TimestampType {
            is_adjusted_to_utc: true,
            unit: TimeUnit::Micros,
        })
    );
}

#[test]
fn logical_type_time_with_nanos_unit() {
    let unit_inner = CompactBuilder::empty_struct();
    let unit = CompactBuilder::new().struct_field(3, &unit_inner).stop();
    let inner = CompactBuilder::new()
        .bool_field(1, false)
        .struct_field(2, &unit)
        .stop();
    let bytes = CompactBuilder::new().struct_field(7, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_logical_type(&mut cur).unwrap(),
        LogicalType::Time(TimeType {
            is_adjusted_to_utc: false,
            unit: TimeUnit::Nanos,
        })
    );
}

#[test]
fn logical_type_variant_with_default_unset_spec_version() {
    // Empty VariantType — specification_version optional, absent.
    let inner = CompactBuilder::empty_struct();
    let bytes = CompactBuilder::new().struct_field(16, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_logical_type(&mut cur).unwrap(),
        LogicalType::Variant(VariantType {
            specification_version: None,
        })
    );
}

#[test]
fn logical_type_geometry_with_crs() {
    let inner = CompactBuilder::new().binary(1, b"EPSG:4326").stop();
    let bytes = CompactBuilder::new().struct_field(17, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    let got = read_logical_type(&mut cur).unwrap();
    match got {
        LogicalType::Geometry(GeometryType { crs }) => {
            assert_eq!(crs, Some(&b"EPSG:4326"[..]));
        }
        other => panic!("expected Geometry, got {other:?}"),
    }
}

#[test]
fn logical_type_geography_with_algorithm() {
    // GeographyType { crs: None, algorithm: Some(Vincenty=1) }
    let inner = CompactBuilder::new().enum_field(2, 1).stop();
    let bytes = CompactBuilder::new().struct_field(18, &inner).stop();
    let mut cur = Cursor::new(&bytes);
    let got = read_logical_type(&mut cur).unwrap();
    match got {
        LogicalType::Geography(GeographyType { crs, algorithm }) => {
            assert_eq!(crs, None);
            assert_eq!(algorithm, Some(EdgeInterpolationAlgorithm::Vincenty));
        }
        other => panic!("expected Geography, got {other:?}"),
    }
}

#[test]
fn logical_type_reserved_id_9_is_unknown() {
    let bytes = make_union(9);
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_logical_type(&mut cur),
        Err(FormatError::UnknownStructField {
            struct_name: "LogicalType",
            field_id: 9,
        })
    ));
}
