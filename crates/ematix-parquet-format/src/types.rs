//! Parquet's i32-valued enums (decoded as zigzag varints).
//!
//! Source of truth: the official `parquet.thrift` at
//! https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift
//!
//! Every enum here is `#[repr(i32)]` so the discriminant matches the
//! thrift wire value 1:1 — no separate lookup table needed.

use crate::compact::{read_zigzag_i32, Cursor};
use crate::error::{FormatError, Result};

/// Common decode interface so callers can stay generic over which
/// enum they're reading. Each impl reads a single zigzag varint and
/// maps it through the type's `from_i32`.
pub trait ThriftEnum: Sized {
    const TYPE_NAME: &'static str;
    fn from_i32(value: i32) -> Result<Self>;
    fn read(cur: &mut Cursor<'_>) -> Result<Self> {
        let v = read_zigzag_i32(cur)?;
        Self::from_i32(v)
    }
}

macro_rules! thrift_enum {
    (
        $(#[$attr:meta])*
        pub enum $name:ident {
            $( $variant:ident = $value:literal ),* $(,)?
        }
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(i32)]
        pub enum $name {
            $( $variant = $value ),*
        }

        impl ThriftEnum for $name {
            const TYPE_NAME: &'static str = stringify!($name);
            fn from_i32(value: i32) -> Result<Self> {
                match value {
                    $( $value => Ok(Self::$variant), )*
                    _ => Err(FormatError::InvalidEnumValue {
                        type_name: stringify!($name),
                        value,
                    }),
                }
            }
        }
    };
}

thrift_enum! {
    /// Physical column type. `ParquetType` (not `Type`) to avoid the
    /// clash with Rust's `Type` AST/keyword-adjacent naming.
    pub enum ParquetType {
        Boolean = 0,
        Int32 = 1,
        Int64 = 2,
        Int96 = 3,
        Float = 4,
        Double = 5,
        ByteArray = 6,
        FixedLenByteArray = 7,
    }
}

thrift_enum! {
    /// Column-data encoding. Value 1 is intentionally unassigned in the
    /// spec (legacy PLAIN_DICTIONARY moved to 2). Includes Parquet v2
    /// encodings DELTA_* and BYTE_STREAM_SPLIT.
    pub enum Encoding {
        Plain = 0,
        PlainDictionary = 2,
        Rle = 3,
        BitPacked = 4,
        DeltaBinaryPacked = 5,
        DeltaLengthByteArray = 6,
        DeltaByteArray = 7,
        RleDictionary = 8,
        ByteStreamSplit = 9,
    }
}

thrift_enum! {
    /// Compression codec applied to a column chunk.
    pub enum CompressionCodec {
        Uncompressed = 0,
        Snappy = 1,
        Gzip = 2,
        Lzo = 3,
        Brotli = 4,
        Lz4 = 5,
        Zstd = 6,
        Lz4Raw = 7,
    }
}

thrift_enum! {
    /// Page kind in the column chunk. `DataPageV2` carries
    /// rep/def levels uncompressed and separates them from the
    /// encoded data.
    pub enum PageType {
        DataPage = 0,
        IndexPage = 1,
        DictionaryPage = 2,
        DataPageV2 = 3,
    }
}

thrift_enum! {
    /// Schema-element repetition.
    pub enum FieldRepetitionType {
        Required = 0,
        Optional = 1,
        Repeated = 2,
    }
}

thrift_enum! {
    /// Sort order of min/max stats across page boundaries.
    pub enum BoundaryOrder {
        Unordered = 0,
        Ascending = 1,
        Descending = 2,
    }
}

thrift_enum! {
    /// Legacy logical-type annotation. Newer files prefer the
    /// `LogicalType` union, but this enum is still required for
    /// backward compatibility.
    pub enum ConvertedType {
        Utf8 = 0,
        Map = 1,
        MapKeyValue = 2,
        List = 3,
        Enum = 4,
        Decimal = 5,
        Date = 6,
        TimeMillis = 7,
        TimeMicros = 8,
        TimestampMillis = 9,
        TimestampMicros = 10,
        Uint8 = 11,
        Uint16 = 12,
        Uint32 = 13,
        Uint64 = 14,
        Int8 = 15,
        Int16 = 16,
        Int32 = 17,
        Int64 = 18,
        Json = 19,
        Bson = 20,
        Interval = 21,
    }
}

