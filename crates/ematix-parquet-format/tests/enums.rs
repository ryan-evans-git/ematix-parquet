//! TDD pin for Parquet's i32-valued enums.
//!
//! In thrift compact protocol, an i32-valued enum used as a struct
//! field is encoded as a zigzag varint body following an I32 type
//! header. We expose one decoder per enum (`read_<name>`) plus a
//! generic `ThriftEnum` trait so future code can stay polymorphic
//! when walking unknown structures.
//!
//! Reference: https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::types::{
    BoundaryOrder, CompressionCodec, ConvertedType, Encoding, FieldRepetitionType, PageType,
    ParquetType, ThriftEnum,
};

/// Encode a non-negative i32 as a zigzag varint — gives us the
/// expected body bytes for any enum value without depending on
/// the very impl we're testing.
fn zz_bytes(v: i32) -> Vec<u8> {
    let mut u = ((v << 1) ^ (v >> 31)) as u32;
    let mut out = Vec::new();
    loop {
        if u < 0x80 {
            out.push(u as u8);
            return out;
        }
        out.push(((u & 0x7F) | 0x80) as u8);
        u >>= 7;
    }
}

fn assert_roundtrip<E: ThriftEnum + std::fmt::Debug + PartialEq>(value: i32, expected: E) {
    let bytes = zz_bytes(value);
    let mut cur = Cursor::new(&bytes);
    let got = E::read(&mut cur).unwrap();
    assert_eq!(got, expected, "value {value} decoded wrong");
    assert_eq!(cur.position(), bytes.len(), "did not consume full varint");
}

fn assert_rejects<E: ThriftEnum + std::fmt::Debug>(value: i32, type_name: &'static str) {
    let bytes = zz_bytes(value);
    let mut cur = Cursor::new(&bytes);
    match E::read(&mut cur) {
        Err(FormatError::InvalidEnumValue {
            type_name: tn,
            value: v,
        }) => {
            assert_eq!(tn, type_name, "wrong type_name in error");
            assert_eq!(v, value, "wrong value in error");
        }
        other => panic!("expected InvalidEnumValue for {value}, got {other:?}"),
    }
}

#[test]
fn parquet_type_all_variants() {
    assert_roundtrip(0, ParquetType::Boolean);
    assert_roundtrip(1, ParquetType::Int32);
    assert_roundtrip(2, ParquetType::Int64);
    assert_roundtrip(3, ParquetType::Int96);
    assert_roundtrip(4, ParquetType::Float);
    assert_roundtrip(5, ParquetType::Double);
    assert_roundtrip(6, ParquetType::ByteArray);
    assert_roundtrip(7, ParquetType::FixedLenByteArray);
}

#[test]
fn parquet_type_rejects_unknown() {
    assert_rejects::<ParquetType>(8, "ParquetType");
    assert_rejects::<ParquetType>(-1, "ParquetType");
    assert_rejects::<ParquetType>(99, "ParquetType");
}

#[test]
fn encoding_all_variants_incl_v2() {
    // Per spec, value 1 is intentionally not assigned (legacy
    // PLAIN_DICTIONARY moved to 2). The decoder must reject 1.
    assert_roundtrip(0, Encoding::Plain);
    assert_roundtrip(2, Encoding::PlainDictionary);
    assert_roundtrip(3, Encoding::Rle);
    assert_roundtrip(4, Encoding::BitPacked);
    assert_roundtrip(5, Encoding::DeltaBinaryPacked);
    assert_roundtrip(6, Encoding::DeltaLengthByteArray);
    assert_roundtrip(7, Encoding::DeltaByteArray);
    assert_roundtrip(8, Encoding::RleDictionary);
    assert_roundtrip(9, Encoding::ByteStreamSplit);
}

#[test]
fn encoding_rejects_unknown_and_gap_value() {
    assert_rejects::<Encoding>(1, "Encoding");
    assert_rejects::<Encoding>(10, "Encoding");
    assert_rejects::<Encoding>(-1, "Encoding");
}

#[test]
fn compression_codec_all_variants_incl_lz4_raw() {
    assert_roundtrip(0, CompressionCodec::Uncompressed);
    assert_roundtrip(1, CompressionCodec::Snappy);
    assert_roundtrip(2, CompressionCodec::Gzip);
    assert_roundtrip(3, CompressionCodec::Lzo);
    assert_roundtrip(4, CompressionCodec::Brotli);
    assert_roundtrip(5, CompressionCodec::Lz4);
    assert_roundtrip(6, CompressionCodec::Zstd);
    assert_roundtrip(7, CompressionCodec::Lz4Raw);
}

#[test]
fn compression_codec_rejects_unknown() {
    assert_rejects::<CompressionCodec>(8, "CompressionCodec");
    assert_rejects::<CompressionCodec>(-1, "CompressionCodec");
}

#[test]
fn page_type_all_variants_incl_v2() {
    assert_roundtrip(0, PageType::DataPage);
    assert_roundtrip(1, PageType::IndexPage);
    assert_roundtrip(2, PageType::DictionaryPage);
    assert_roundtrip(3, PageType::DataPageV2);
}

#[test]
fn page_type_rejects_unknown() {
    assert_rejects::<PageType>(4, "PageType");
    assert_rejects::<PageType>(-1, "PageType");
}

#[test]
fn field_repetition_type_all_variants() {
    assert_roundtrip(0, FieldRepetitionType::Required);
    assert_roundtrip(1, FieldRepetitionType::Optional);
    assert_roundtrip(2, FieldRepetitionType::Repeated);
}

#[test]
fn field_repetition_type_rejects_unknown() {
    assert_rejects::<FieldRepetitionType>(3, "FieldRepetitionType");
    assert_rejects::<FieldRepetitionType>(-1, "FieldRepetitionType");
}

#[test]
fn boundary_order_all_variants() {
    assert_roundtrip(0, BoundaryOrder::Unordered);
    assert_roundtrip(1, BoundaryOrder::Ascending);
    assert_roundtrip(2, BoundaryOrder::Descending);
}

#[test]
fn boundary_order_rejects_unknown() {
    assert_rejects::<BoundaryOrder>(3, "BoundaryOrder");
}

#[test]
fn converted_type_all_22_variants() {
    use ConvertedType::*;
    let cases = [
        (0i32, Utf8),
        (1, Map),
        (2, MapKeyValue),
        (3, List),
        (4, Enum),
        (5, Decimal),
        (6, Date),
        (7, TimeMillis),
        (8, TimeMicros),
        (9, TimestampMillis),
        (10, TimestampMicros),
        (11, Uint8),
        (12, Uint16),
        (13, Uint32),
        (14, Uint64),
        (15, Int8),
        (16, Int16),
        (17, Int32),
        (18, Int64),
        (19, Json),
        (20, Bson),
        (21, Interval),
    ];
    for (v, expected) in cases {
        assert_roundtrip(v, expected);
    }
}

#[test]
fn converted_type_rejects_unknown() {
    assert_rejects::<ConvertedType>(22, "ConvertedType");
    assert_rejects::<ConvertedType>(-1, "ConvertedType");
}
