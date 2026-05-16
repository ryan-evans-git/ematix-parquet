//! Parquet metadata structs decoded from the thrift compact protocol.
//!
//! All struct readers are zero-copy: variable-length binary fields
//! borrow `&[u8]` from the cursor's underlying buffer. Callers that
//! need owned data can `.to_vec()` the slices.

use crate::compact::{
    read_binary, read_field_header, read_i8, read_list_binary, read_list_bool, read_list_i32,
    read_list_i64, read_list_struct, read_zigzag_i16, read_zigzag_i32, read_zigzag_i64, Cursor,
    FieldType,
};
use crate::error::{FormatError, Result};
use crate::types::{
    BoundaryOrder, CompressionCodec, ConvertedType, EdgeInterpolationAlgorithm, Encoding,
    FieldRepetitionType, PageType, ParquetType, ThriftEnum,
};

/// Per-page or per-column-chunk statistics, as produced by writers
/// that support the deprecated (min/max) and/or current
/// (min_value/max_value) field pairs.
///
/// All fields are optional in the spec. The two pairs are deprecated
/// vs current; `max_value`/`min_value` should be preferred when both
/// are present.
///
/// Field ids match parquet.thrift:
///   1: max         2: min
///   3: null_count  4: distinct_count
///   5: max_value   6: min_value
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Statistics<'a> {
    pub max: Option<&'a [u8]>,
    pub min: Option<&'a [u8]>,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<&'a [u8]>,
    pub min_value: Option<&'a [u8]>,
    /// Whether the stored `max_value` is the exact maximum (true) or
    /// a truncated/approximate upper bound (false). Added in parquet
    /// 2.9. Absent for older writers.
    pub is_max_value_exact: Option<bool>,
    /// Same idea for `min_value`.
    pub is_min_value_exact: Option<bool>,
}

/// File-level user-defined metadata entry. `value` is optional per spec.
/// Thrift `string` is wire-identical to `binary`; we expose raw bytes
/// and let callers UTF-8-decode at the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyValue<'a> {
    pub key: &'a [u8],
    pub value: Option<&'a [u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeader<'a> {
    pub num_values: i32,
    pub encoding: Encoding,
    pub definition_level_encoding: Encoding,
    pub repetition_level_encoding: Encoding,
    pub statistics: Option<Statistics<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPageHeader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictionaryPageHeader {
    pub num_values: i32,
    pub encoding: Encoding,
    pub is_sorted: Option<bool>,
}

/// Parquet v2 data page header. `is_compressed` has spec default
/// `true`, so an absent field 7 must decode to `true`, not `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeaderV2<'a> {
    pub num_values: i32,
    pub num_nulls: i32,
    pub num_rows: i32,
    pub encoding: Encoding,
    pub definition_levels_byte_length: i32,
    pub repetition_levels_byte_length: i32,
    pub is_compressed: bool,
    pub statistics: Option<Statistics<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageHeader<'a> {
    pub page_type: PageType,
    pub uncompressed_page_size: i32,
    pub compressed_page_size: i32,
    pub crc: Option<i32>,
    pub data_page_header: Option<DataPageHeader<'a>>,
    pub index_page_header: Option<IndexPageHeader>,
    pub dictionary_page_header: Option<DictionaryPageHeader>,
    pub data_page_header_v2: Option<DataPageHeaderV2<'a>>,
}

fn missing(struct_name: &'static str, field_id: i16) -> FormatError {
    FormatError::MissingRequiredField {
        struct_name,
        field_id,
    }
}

fn unknown(struct_name: &'static str, field_id: i16) -> FormatError {
    FormatError::UnknownStructField {
        struct_name,
        field_id,
    }
}

/// Consume an empty thrift struct (just the STOP byte). Used as the
/// payload reader for marker variants of `LogicalType` (StringType,
/// UUIDType, …) and `TimeUnit` (MilliSeconds, MicroSeconds, NanoSeconds).
fn read_empty_struct(cur: &mut Cursor<'_>, name: &'static str) -> Result<()> {
    if let Some(h) = read_field_header(cur, 0)? {
        return Err(unknown(name, h.id));
    }
    Ok(())
}

// ---- LogicalType payload structs ----------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalType {
    pub scale: i32,
    pub precision: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntType {
    pub bit_width: i8,
    pub is_signed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Millis,
    Micros,
    Nanos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampType {
    pub is_adjusted_to_utc: bool,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeType {
    pub is_adjusted_to_utc: bool,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantType {
    pub specification_version: Option<i8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryType<'a> {
    pub crs: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeographyType<'a> {
    pub crs: Option<&'a [u8]>,
    pub algorithm: Option<EdgeInterpolationAlgorithm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalType<'a> {
    String,
    Map,
    List,
    Enum,
    Decimal(DecimalType),
    Date,
    Time(TimeType),
    Timestamp(TimestampType),
    Integer(IntType),
    /// `UNKNOWN` in the spec — payload is `NullType{}`.
    Null,
    Json,
    Bson,
    Uuid,
    Float16,
    Variant(VariantType),
    Geometry(GeometryType<'a>),
    Geography(GeographyType<'a>),
}

// ---- TimeUnit union -------------------------------------------------------

pub fn read_time_unit(cur: &mut Cursor<'_>) -> Result<TimeUnit> {
    let mut chosen: Option<TimeUnit> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                read_empty_struct(cur, "MilliSeconds")?;
                chosen = Some(TimeUnit::Millis);
            }
            (2, FieldType::Struct) => {
                read_empty_struct(cur, "MicroSeconds")?;
                chosen = Some(TimeUnit::Micros);
            }
            (3, FieldType::Struct) => {
                read_empty_struct(cur, "NanoSeconds")?;
                chosen = Some(TimeUnit::Nanos);
            }
            _ => return Err(unknown("TimeUnit", h.id)),
        }
    }
    chosen.ok_or(FormatError::EmptyUnion {
        union_name: "TimeUnit",
    })
}

// ---- LogicalType union payload readers ------------------------------------

fn read_decimal_type(cur: &mut Cursor<'_>) -> Result<DecimalType> {
    let mut scale = None;
    let mut precision = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => scale = Some(read_zigzag_i32(cur)?),
            (2, FieldType::I32) => precision = Some(read_zigzag_i32(cur)?),
            _ => return Err(unknown("DecimalType", h.id)),
        }
    }
    Ok(DecimalType {
        scale: scale.ok_or_else(|| missing("DecimalType", 1))?,
        precision: precision.ok_or_else(|| missing("DecimalType", 2))?,
    })
}

fn read_int_type(cur: &mut Cursor<'_>) -> Result<IntType> {
    let mut bit_width = None;
    let mut is_signed = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Byte) => bit_width = Some(read_i8(cur)?),
            (2, FieldType::BoolTrue) => is_signed = Some(true),
            (2, FieldType::BoolFalse) => is_signed = Some(false),
            _ => return Err(unknown("IntType", h.id)),
        }
    }
    Ok(IntType {
        bit_width: bit_width.ok_or_else(|| missing("IntType", 1))?,
        is_signed: is_signed.ok_or_else(|| missing("IntType", 2))?,
    })
}

fn read_timestamp_type(cur: &mut Cursor<'_>) -> Result<TimestampType> {
    let mut is_utc = None;
    let mut unit = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::BoolTrue) => is_utc = Some(true),
            (1, FieldType::BoolFalse) => is_utc = Some(false),
            (2, FieldType::Struct) => unit = Some(read_time_unit(cur)?),
            _ => return Err(unknown("TimestampType", h.id)),
        }
    }
    Ok(TimestampType {
        is_adjusted_to_utc: is_utc.ok_or_else(|| missing("TimestampType", 1))?,
        unit: unit.ok_or_else(|| missing("TimestampType", 2))?,
    })
}

fn read_time_type(cur: &mut Cursor<'_>) -> Result<TimeType> {
    let mut is_utc = None;
    let mut unit = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::BoolTrue) => is_utc = Some(true),
            (1, FieldType::BoolFalse) => is_utc = Some(false),
            (2, FieldType::Struct) => unit = Some(read_time_unit(cur)?),
            _ => return Err(unknown("TimeType", h.id)),
        }
    }
    Ok(TimeType {
        is_adjusted_to_utc: is_utc.ok_or_else(|| missing("TimeType", 1))?,
        unit: unit.ok_or_else(|| missing("TimeType", 2))?,
    })
}

fn read_variant_type(cur: &mut Cursor<'_>) -> Result<VariantType> {
    let mut spec = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Byte) => spec = Some(read_i8(cur)?),
            _ => return Err(unknown("VariantType", h.id)),
        }
    }
    Ok(VariantType {
        specification_version: spec,
    })
}

fn read_geometry_type<'a>(cur: &mut Cursor<'a>) -> Result<GeometryType<'a>> {
    let mut crs = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => crs = Some(read_binary(cur)?),
            _ => return Err(unknown("GeometryType", h.id)),
        }
    }
    Ok(GeometryType { crs })
}

fn read_geography_type<'a>(cur: &mut Cursor<'a>) -> Result<GeographyType<'a>> {
    let mut crs = None;
    let mut algorithm = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => crs = Some(read_binary(cur)?),
            (2, FieldType::I32) => algorithm = Some(EdgeInterpolationAlgorithm::read(cur)?),
            _ => return Err(unknown("GeographyType", h.id)),
        }
    }
    Ok(GeographyType { crs, algorithm })
}

// ---- LogicalType union ----------------------------------------------------

pub fn read_logical_type<'a>(cur: &mut Cursor<'a>) -> Result<LogicalType<'a>> {
    let mut chosen: Option<LogicalType<'a>> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        let variant = match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                read_empty_struct(cur, "StringType")?;
                LogicalType::String
            }
            (2, FieldType::Struct) => {
                read_empty_struct(cur, "MapType")?;
                LogicalType::Map
            }
            (3, FieldType::Struct) => {
                read_empty_struct(cur, "ListType")?;
                LogicalType::List
            }
            (4, FieldType::Struct) => {
                read_empty_struct(cur, "EnumType")?;
                LogicalType::Enum
            }
            (5, FieldType::Struct) => LogicalType::Decimal(read_decimal_type(cur)?),
            (6, FieldType::Struct) => {
                read_empty_struct(cur, "DateType")?;
                LogicalType::Date
            }
            (7, FieldType::Struct) => LogicalType::Time(read_time_type(cur)?),
            (8, FieldType::Struct) => LogicalType::Timestamp(read_timestamp_type(cur)?),
            (10, FieldType::Struct) => LogicalType::Integer(read_int_type(cur)?),
            (11, FieldType::Struct) => {
                read_empty_struct(cur, "NullType")?;
                LogicalType::Null
            }
            (12, FieldType::Struct) => {
                read_empty_struct(cur, "JsonType")?;
                LogicalType::Json
            }
            (13, FieldType::Struct) => {
                read_empty_struct(cur, "BsonType")?;
                LogicalType::Bson
            }
            (14, FieldType::Struct) => {
                read_empty_struct(cur, "UUIDType")?;
                LogicalType::Uuid
            }
            (15, FieldType::Struct) => {
                read_empty_struct(cur, "Float16Type")?;
                LogicalType::Float16
            }
            (16, FieldType::Struct) => LogicalType::Variant(read_variant_type(cur)?),
            (17, FieldType::Struct) => LogicalType::Geometry(read_geometry_type(cur)?),
            (18, FieldType::Struct) => LogicalType::Geography(read_geography_type(cur)?),
            // id 9 is reserved in the spec; anything else is genuinely
            // unknown.
            _ => return Err(unknown("LogicalType", h.id)),
        };
        chosen = Some(variant);
    }
    chosen.ok_or(FormatError::EmptyUnion {
        union_name: "LogicalType",
    })
}

// ---- SortingColumn + RowGroup ---------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortingColumn {
    pub column_idx: i32,
    pub descending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RowGroup<'a> {
    pub columns: Vec<ColumnChunk<'a>>,
    pub total_byte_size: i64,
    pub num_rows: i64,
    pub sorting_columns: Option<Vec<SortingColumn>>,
    pub file_offset: Option<i64>,
    pub total_compressed_size: Option<i64>,
    pub ordinal: Option<i16>,
}

pub fn read_sorting_column(cur: &mut Cursor<'_>) -> Result<SortingColumn> {
    let mut column_idx = None;
    let mut descending = None;
    let mut nulls_first = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => column_idx = Some(read_zigzag_i32(cur)?),
            (2, FieldType::BoolTrue) => descending = Some(true),
            (2, FieldType::BoolFalse) => descending = Some(false),
            (3, FieldType::BoolTrue) => nulls_first = Some(true),
            (3, FieldType::BoolFalse) => nulls_first = Some(false),
            _ => return Err(unknown("SortingColumn", h.id)),
        }
    }
    Ok(SortingColumn {
        column_idx: column_idx.ok_or_else(|| missing("SortingColumn", 1))?,
        descending: descending.ok_or_else(|| missing("SortingColumn", 2))?,
        nulls_first: nulls_first.ok_or_else(|| missing("SortingColumn", 3))?,
    })
}

pub fn read_row_group<'a>(cur: &mut Cursor<'a>) -> Result<RowGroup<'a>> {
    let mut columns = None;
    let mut total_byte_size = None;
    let mut num_rows = None;
    let mut sorting_columns = None;
    let mut file_offset = None;
    let mut total_compressed_size = None;
    let mut ordinal = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::List) => columns = Some(read_list_struct(cur, read_column_chunk)?),
            (2, FieldType::I64) => total_byte_size = Some(read_zigzag_i64(cur)?),
            (3, FieldType::I64) => num_rows = Some(read_zigzag_i64(cur)?),
            (4, FieldType::List) => {
                sorting_columns = Some(read_list_struct(cur, |c| read_sorting_column(c))?);
            }
            (5, FieldType::I64) => file_offset = Some(read_zigzag_i64(cur)?),
            (6, FieldType::I64) => total_compressed_size = Some(read_zigzag_i64(cur)?),
            (7, FieldType::I16) => ordinal = Some(read_zigzag_i16(cur)?),
            _ => return Err(unknown("RowGroup", h.id)),
        }
    }
    Ok(RowGroup {
        columns: columns.ok_or_else(|| missing("RowGroup", 1))?,
        total_byte_size: total_byte_size.ok_or_else(|| missing("RowGroup", 2))?,
        num_rows: num_rows.ok_or_else(|| missing("RowGroup", 3))?,
        sorting_columns,
        file_offset,
        total_compressed_size,
        ordinal,
    })
}

// ---- SchemaElement --------------------------------------------------------

/// One node in Parquet's depth-first-flattened schema list.
///
/// Group nodes (the root, struct fields, list/map nodes) carry only
/// `name` + `num_children` + repetition. Leaf nodes carry `column_type`
/// plus optional annotations (`converted_type`, `scale`, `precision`,
/// `logical_type`).
///
/// Field name `column_type` instead of `type` to avoid shadowing the
/// Rust keyword.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SchemaElement<'a> {
    pub column_type: Option<ParquetType>,
    pub type_length: Option<i32>,
    pub repetition_type: Option<FieldRepetitionType>,
    pub name: &'a [u8],
    pub num_children: Option<i32>,
    pub converted_type: Option<ConvertedType>,
    pub scale: Option<i32>,
    pub precision: Option<i32>,
    pub field_id: Option<i32>,
    pub logical_type: Option<LogicalType<'a>>,
}

pub fn read_schema_element<'a>(cur: &mut Cursor<'a>) -> Result<SchemaElement<'a>> {
    let mut column_type = None;
    let mut type_length = None;
    let mut repetition_type = None;
    let mut name: Option<&[u8]> = None;
    let mut num_children = None;
    let mut converted_type = None;
    let mut scale = None;
    let mut precision = None;
    let mut field_id = None;
    let mut logical_type = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => column_type = Some(ParquetType::read(cur)?),
            (2, FieldType::I32) => type_length = Some(read_zigzag_i32(cur)?),
            (3, FieldType::I32) => repetition_type = Some(FieldRepetitionType::read(cur)?),
            (4, FieldType::Binary) => name = Some(read_binary(cur)?),
            (5, FieldType::I32) => num_children = Some(read_zigzag_i32(cur)?),
            (6, FieldType::I32) => converted_type = Some(ConvertedType::read(cur)?),
            (7, FieldType::I32) => scale = Some(read_zigzag_i32(cur)?),
            (8, FieldType::I32) => precision = Some(read_zigzag_i32(cur)?),
            (9, FieldType::I32) => field_id = Some(read_zigzag_i32(cur)?),
            (10, FieldType::Struct) => logical_type = Some(read_logical_type(cur)?),
            _ => return Err(unknown("SchemaElement", h.id)),
        }
    }
    Ok(SchemaElement {
        column_type,
        type_length,
        repetition_type,
        name: name.ok_or_else(|| missing("SchemaElement", 4))?,
        num_children,
        converted_type,
        scale,
        precision,
        field_id,
        logical_type,
    })
}

// ---- Page-index types -----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLocation {
    pub offset: i64,
    pub compressed_page_size: i32,
    pub first_row_index: i64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OffsetIndex {
    pub page_locations: Vec<PageLocation>,
    /// Per-page unencoded size for BYTE_ARRAY columns. Required when
    /// any column in the file is BYTE_ARRAY; absent otherwise.
    pub unencoded_byte_array_data_bytes: Option<Vec<i64>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ColumnIndex<'a> {
    /// Per-page null marker. `null_pages[i]==true` means
    /// `min_values[i]`/`max_values[i]` are not meaningful (all rows
    /// in that page were null).
    pub null_pages: Vec<bool>,
    pub min_values: Vec<&'a [u8]>,
    pub max_values: Vec<&'a [u8]>,
    pub boundary_order: BoundaryOrder,
    pub null_counts: Option<Vec<i64>>,
}

pub fn read_page_location(cur: &mut Cursor<'_>) -> Result<PageLocation> {
    let mut offset = None;
    let mut compressed = None;
    let mut first_row = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I64) => offset = Some(read_zigzag_i64(cur)?),
            (2, FieldType::I32) => compressed = Some(read_zigzag_i32(cur)?),
            (3, FieldType::I64) => first_row = Some(read_zigzag_i64(cur)?),
            _ => return Err(unknown("PageLocation", h.id)),
        }
    }
    Ok(PageLocation {
        offset: offset.ok_or_else(|| missing("PageLocation", 1))?,
        compressed_page_size: compressed.ok_or_else(|| missing("PageLocation", 2))?,
        first_row_index: first_row.ok_or_else(|| missing("PageLocation", 3))?,
    })
}

pub fn read_offset_index(cur: &mut Cursor<'_>) -> Result<OffsetIndex> {
    let mut page_locations = None;
    let mut unencoded = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::List) => {
                page_locations = Some(read_list_struct(cur, read_page_location)?);
            }
            (2, FieldType::List) => unencoded = Some(read_list_i64(cur)?),
            _ => return Err(unknown("OffsetIndex", h.id)),
        }
    }
    Ok(OffsetIndex {
        page_locations: page_locations.ok_or_else(|| missing("OffsetIndex", 1))?,
        unencoded_byte_array_data_bytes: unencoded,
    })
}

pub fn read_column_index<'a>(cur: &mut Cursor<'a>) -> Result<ColumnIndex<'a>> {
    let mut null_pages = None;
    let mut min_values = None;
    let mut max_values = None;
    let mut boundary_order = None;
    let mut null_counts = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::List) => null_pages = Some(read_list_bool(cur)?),
            (2, FieldType::List) => min_values = Some(read_list_binary(cur)?),
            (3, FieldType::List) => max_values = Some(read_list_binary(cur)?),
            (4, FieldType::I32) => boundary_order = Some(BoundaryOrder::read(cur)?),
            (5, FieldType::List) => null_counts = Some(read_list_i64(cur)?),
            _ => return Err(unknown("ColumnIndex", h.id)),
        }
    }
    Ok(ColumnIndex {
        null_pages: null_pages.ok_or_else(|| missing("ColumnIndex", 1))?,
        min_values: min_values.ok_or_else(|| missing("ColumnIndex", 2))?,
        max_values: max_values.ok_or_else(|| missing("ColumnIndex", 3))?,
        boundary_order: boundary_order.ok_or_else(|| missing("ColumnIndex", 4))?,
        null_counts,
    })
}

// ---- ColumnOrder union + FileMetaData -------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnOrder {
    /// Sort order is implied by the column's physical or logical type.
    TypeDefinedOrder,
}

pub fn read_column_order(cur: &mut Cursor<'_>) -> Result<ColumnOrder> {
    let mut chosen = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                read_empty_struct(cur, "TypeDefinedOrder")?;
                chosen = Some(ColumnOrder::TypeDefinedOrder);
            }
            _ => return Err(unknown("ColumnOrder", h.id)),
        }
    }
    chosen.ok_or(FormatError::EmptyUnion {
        union_name: "ColumnOrder",
    })
}

/// Top-level Parquet footer struct.
///
/// Fields 8/9 carry the Parquet Modular Encryption (PME) descriptor on
/// **plaintext-footer** files; on **encrypted-footer** files the whole
/// footer lives inside `FileCryptoMetaData` instead and these are
/// absent here. See `ematix-parquet-format/src/metadata.rs` PME section
/// (`read_file_crypto_metadata`) for the encrypted-footer trailer.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FileMetaData<'a> {
    pub version: i32,
    pub schema: Vec<SchemaElement<'a>>,
    pub num_rows: i64,
    pub row_groups: Vec<RowGroup<'a>>,
    pub key_value_metadata: Option<Vec<KeyValue<'a>>>,
    pub created_by: Option<&'a [u8]>,
    pub column_orders: Option<Vec<ColumnOrder>>,
    /// `EncryptionAlgorithm` union present iff the file is PME-encrypted.
    /// On plaintext-footer mode this describes how the column chunks
    /// were encrypted; on encrypted-footer mode this field is in
    /// `FileCryptoMetaData` instead.
    pub encryption_algorithm: Option<EncryptionAlgorithm<'a>>,
    /// Opaque caller-defined identifier for the key used to sign the
    /// (plaintext) footer in plaintext-footer mode. Caller decides how
    /// to map it back to key material. Absent on unencrypted files and
    /// on encrypted-footer files.
    pub footer_signing_key_metadata: Option<&'a [u8]>,
}

pub fn read_file_metadata<'a>(cur: &mut Cursor<'a>) -> Result<FileMetaData<'a>> {
    let mut version = None;
    let mut schema = None;
    let mut num_rows = None;
    let mut row_groups = None;
    let mut key_value_metadata = None;
    let mut created_by = None;
    let mut column_orders = None;
    let mut encryption_algorithm = None;
    let mut footer_signing_key_metadata = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => version = Some(read_zigzag_i32(cur)?),
            (2, FieldType::List) => schema = Some(read_list_struct(cur, read_schema_element)?),
            (3, FieldType::I64) => num_rows = Some(read_zigzag_i64(cur)?),
            (4, FieldType::List) => row_groups = Some(read_list_struct(cur, read_row_group)?),
            (5, FieldType::List) => {
                key_value_metadata = Some(read_list_struct(cur, read_key_value)?);
            }
            (6, FieldType::Binary) => created_by = Some(read_binary(cur)?),
            (7, FieldType::List) => {
                column_orders = Some(read_list_struct(cur, |c| read_column_order(c))?);
            }
            (8, FieldType::Struct) => {
                encryption_algorithm = Some(read_encryption_algorithm(cur)?);
            }
            (9, FieldType::Binary) => {
                footer_signing_key_metadata = Some(read_binary(cur)?);
            }
            _ => return Err(unknown("FileMetaData", h.id)),
        }
    }
    Ok(FileMetaData {
        version: version.ok_or_else(|| missing("FileMetaData", 1))?,
        schema: schema.ok_or_else(|| missing("FileMetaData", 2))?,
        num_rows: num_rows.ok_or_else(|| missing("FileMetaData", 3))?,
        row_groups: row_groups.ok_or_else(|| missing("FileMetaData", 4))?,
        key_value_metadata,
        created_by,
        column_orders,
        encryption_algorithm,
        footer_signing_key_metadata,
    })
}

/// Per-(page_type, encoding) page count, used for read-side stats
/// even though `encoding_stats` itself is optional on `ColumnMetaData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageEncodingStats {
    pub page_type: PageType,
    pub encoding: Encoding,
    pub count: i32,
}

/// Per-page byte/level-histogram size statistics. Optional add-on
/// to `ColumnMetaData` (field 16) and `OffsetIndex` (subset).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SizeStatistics {
    pub unencoded_byte_array_data_bytes: Option<i64>,
    pub repetition_level_histogram: Option<Vec<i64>>,
    pub definition_level_histogram: Option<Vec<i64>>,
}

/// Header preceding the bitset bytes of a Parquet bloom filter.
///
/// All four fields are required by the spec, but algorithm / hash /
/// compression are tagged unions where today only one variant each
/// is defined: `SplitBlockAlgorithm`, `XxHash`, and `Uncompressed`.
/// We surface those as flat enums for now since no other variants
/// exist in any deployed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BloomFilterHeader {
    pub num_bytes: i32,
    pub algorithm: BloomFilterAlgorithm,
    pub hash: BloomFilterHash,
    pub compression: BloomFilterCompression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomFilterAlgorithm {
    /// `BLOCK` — the only algorithm any writer in the wild uses.
    SplitBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomFilterHash {
    /// XXHash64 with seed 0.
    XxHash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomFilterCompression {
    Uncompressed,
}

/// Per-column-chunk descriptor. Field 17 (GeospatialStatistics) and
/// the ColumnCryptoMetaData union are not yet modeled and will error
/// as `UnknownStructField`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMetaData<'a> {
    pub column_type: ParquetType,
    pub encodings: Vec<Encoding>,
    pub path_in_schema: Vec<&'a [u8]>,
    pub codec: CompressionCodec,
    pub num_values: i64,
    pub total_uncompressed_size: i64,
    pub total_compressed_size: i64,
    pub key_value_metadata: Option<Vec<KeyValue<'a>>>,
    pub data_page_offset: i64,
    pub index_page_offset: Option<i64>,
    pub dictionary_page_offset: Option<i64>,
    pub statistics: Option<Statistics<'a>>,
    pub encoding_stats: Option<Vec<PageEncodingStats>>,
    pub bloom_filter_offset: Option<i64>,
    pub bloom_filter_length: Option<i32>,
    pub size_statistics: Option<SizeStatistics>,
}

/// Top-level chunk descriptor in `RowGroup.columns`.
///
/// PME fields:
/// - `crypto_metadata` (field 8) carries which key encrypts this column
///   chunk's pages — either the file's footer key
///   (`EncryptionWithFooterKey`) or a per-column key
///   (`EncryptionWithColumnKey { path_in_schema, key_metadata }`).
///   Present iff the file is encrypted *and* this column participates.
/// - `encrypted_column_metadata` (field 9) is the AES-GCM ciphertext of
///   the chunk's `ColumnMetaData`, used in plaintext-footer mode to
///   keep per-column stats / offsets confidential. When present,
///   `meta_data` is `None` and the caller decrypts these bytes into a
///   `ColumnMetaData` itself.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ColumnChunk<'a> {
    pub file_path: Option<&'a [u8]>,
    pub file_offset: i64,
    pub meta_data: Option<ColumnMetaData<'a>>,
    pub offset_index_offset: Option<i64>,
    pub offset_index_length: Option<i32>,
    pub column_index_offset: Option<i64>,
    pub column_index_length: Option<i32>,
    pub crypto_metadata: Option<ColumnCryptoMetaData<'a>>,
    pub encrypted_column_metadata: Option<&'a [u8]>,
}

pub fn read_size_statistics(cur: &mut Cursor<'_>) -> Result<SizeStatistics> {
    let mut unencoded = None;
    let mut rep_hist = None;
    let mut def_hist = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I64) => unencoded = Some(read_zigzag_i64(cur)?),
            (2, FieldType::List) => rep_hist = Some(read_list_i64(cur)?),
            (3, FieldType::List) => def_hist = Some(read_list_i64(cur)?),
            _ => return Err(unknown("SizeStatistics", h.id)),
        }
    }
    Ok(SizeStatistics {
        unencoded_byte_array_data_bytes: unencoded,
        repetition_level_histogram: rep_hist,
        definition_level_histogram: def_hist,
    })
}

/// Read a `BloomFilterHeader` from the bytes pointed at by
/// `ColumnMetaData.bloom_filter_offset`. The bitset begins
/// immediately after this header in the same byte stream.
pub fn read_bloom_filter_header(cur: &mut Cursor<'_>) -> Result<BloomFilterHeader> {
    let mut num_bytes = None;
    let mut algorithm = None;
    let mut hash = None;
    let mut compression = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => num_bytes = Some(read_zigzag_i32(cur)?),
            (2, FieldType::Struct) => {
                algorithm = Some(read_bloom_algorithm(cur)?);
            }
            (3, FieldType::Struct) => {
                hash = Some(read_bloom_hash(cur)?);
            }
            (4, FieldType::Struct) => {
                compression = Some(read_bloom_compression(cur)?);
            }
            _ => return Err(unknown("BloomFilterHeader", h.id)),
        }
    }
    Ok(BloomFilterHeader {
        num_bytes: num_bytes.ok_or_else(|| missing("BloomFilterHeader", 1))?,
        algorithm: algorithm.ok_or_else(|| missing("BloomFilterHeader", 2))?,
        hash: hash.ok_or_else(|| missing("BloomFilterHeader", 3))?,
        compression: compression.ok_or_else(|| missing("BloomFilterHeader", 4))?,
    })
}

fn read_bloom_algorithm(cur: &mut Cursor<'_>) -> Result<BloomFilterAlgorithm> {
    // Tagged union: each branch is an empty struct. Read the field
    // header (which identifies the variant), then walk the empty
    // struct (which is just FieldType::Stop).
    let mut algo: Option<BloomFilterAlgorithm> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match h.id {
            1 => {
                // SplitBlockAlgorithm — empty struct, walk to stop.
                while let Some(_inner) = read_field_header(cur, 0)? {}
                algo = Some(BloomFilterAlgorithm::SplitBlock);
            }
            _ => return Err(unknown("BloomFilterAlgorithm", h.id)),
        }
    }
    algo.ok_or_else(|| missing("BloomFilterAlgorithm", 1))
}

fn read_bloom_hash(cur: &mut Cursor<'_>) -> Result<BloomFilterHash> {
    let mut hash: Option<BloomFilterHash> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match h.id {
            1 => {
                while let Some(_inner) = read_field_header(cur, 0)? {}
                hash = Some(BloomFilterHash::XxHash);
            }
            _ => return Err(unknown("BloomFilterHash", h.id)),
        }
    }
    hash.ok_or_else(|| missing("BloomFilterHash", 1))
}

fn read_bloom_compression(cur: &mut Cursor<'_>) -> Result<BloomFilterCompression> {
    let mut comp: Option<BloomFilterCompression> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match h.id {
            1 => {
                while let Some(_inner) = read_field_header(cur, 0)? {}
                comp = Some(BloomFilterCompression::Uncompressed);
            }
            _ => return Err(unknown("BloomFilterCompression", h.id)),
        }
    }
    comp.ok_or_else(|| missing("BloomFilterCompression", 1))
}

pub fn read_page_encoding_stats(cur: &mut Cursor<'_>) -> Result<PageEncodingStats> {
    let mut page_type = None;
    let mut encoding = None;
    let mut count = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => page_type = Some(PageType::read(cur)?),
            (2, FieldType::I32) => encoding = Some(Encoding::read(cur)?),
            (3, FieldType::I32) => count = Some(read_zigzag_i32(cur)?),
            _ => return Err(unknown("PageEncodingStats", h.id)),
        }
    }
    Ok(PageEncodingStats {
        page_type: page_type.ok_or_else(|| missing("PageEncodingStats", 1))?,
        encoding: encoding.ok_or_else(|| missing("PageEncodingStats", 2))?,
        count: count.ok_or_else(|| missing("PageEncodingStats", 3))?,
    })
}

pub fn read_column_metadata<'a>(cur: &mut Cursor<'a>) -> Result<ColumnMetaData<'a>> {
    let mut column_type = None;
    let mut encodings = None;
    let mut path = None;
    let mut codec = None;
    let mut num_values = None;
    let mut total_uncompressed = None;
    let mut total_compressed = None;
    let mut kv = None;
    let mut data_page_offset = None;
    let mut index_page_offset = None;
    let mut dict_page_offset = None;
    let mut statistics = None;
    let mut encoding_stats = None;
    let mut bloom_offset = None;
    let mut bloom_length = None;
    let mut size_statistics = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => column_type = Some(ParquetType::read(cur)?),
            (2, FieldType::List) => {
                let raw = read_list_i32(cur)?;
                encodings = Some(
                    raw.into_iter()
                        .map(Encoding::from_i32)
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            (3, FieldType::List) => path = Some(read_list_binary(cur)?),
            (4, FieldType::I32) => codec = Some(CompressionCodec::read(cur)?),
            (5, FieldType::I64) => num_values = Some(read_zigzag_i64(cur)?),
            (6, FieldType::I64) => total_uncompressed = Some(read_zigzag_i64(cur)?),
            (7, FieldType::I64) => total_compressed = Some(read_zigzag_i64(cur)?),
            (8, FieldType::List) => kv = Some(read_list_struct(cur, read_key_value)?),
            (9, FieldType::I64) => data_page_offset = Some(read_zigzag_i64(cur)?),
            (10, FieldType::I64) => index_page_offset = Some(read_zigzag_i64(cur)?),
            (11, FieldType::I64) => dict_page_offset = Some(read_zigzag_i64(cur)?),
            (12, FieldType::Struct) => statistics = Some(read_statistics(cur)?),
            (13, FieldType::List) => {
                encoding_stats = Some(read_list_struct(cur, |c| read_page_encoding_stats(c))?);
            }
            (14, FieldType::I64) => bloom_offset = Some(read_zigzag_i64(cur)?),
            (15, FieldType::I32) => bloom_length = Some(read_zigzag_i32(cur)?),
            (16, FieldType::Struct) => size_statistics = Some(read_size_statistics(cur)?),
            _ => return Err(unknown("ColumnMetaData", h.id)),
        }
    }
    Ok(ColumnMetaData {
        column_type: column_type.ok_or_else(|| missing("ColumnMetaData", 1))?,
        encodings: encodings.ok_or_else(|| missing("ColumnMetaData", 2))?,
        path_in_schema: path.ok_or_else(|| missing("ColumnMetaData", 3))?,
        codec: codec.ok_or_else(|| missing("ColumnMetaData", 4))?,
        num_values: num_values.ok_or_else(|| missing("ColumnMetaData", 5))?,
        total_uncompressed_size: total_uncompressed.ok_or_else(|| missing("ColumnMetaData", 6))?,
        total_compressed_size: total_compressed.ok_or_else(|| missing("ColumnMetaData", 7))?,
        key_value_metadata: kv,
        data_page_offset: data_page_offset.ok_or_else(|| missing("ColumnMetaData", 9))?,
        index_page_offset,
        dictionary_page_offset: dict_page_offset,
        statistics,
        encoding_stats,
        bloom_filter_offset: bloom_offset,
        bloom_filter_length: bloom_length,
        size_statistics,
    })
}

pub fn read_column_chunk<'a>(cur: &mut Cursor<'a>) -> Result<ColumnChunk<'a>> {
    let mut file_path = None;
    let mut file_offset = None;
    let mut meta_data = None;
    let mut offset_index_offset = None;
    let mut offset_index_length = None;
    let mut column_index_offset = None;
    let mut column_index_length = None;
    let mut crypto_metadata = None;
    let mut encrypted_column_metadata = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => file_path = Some(read_binary(cur)?),
            (2, FieldType::I64) => file_offset = Some(read_zigzag_i64(cur)?),
            (3, FieldType::Struct) => meta_data = Some(read_column_metadata(cur)?),
            (4, FieldType::I64) => offset_index_offset = Some(read_zigzag_i64(cur)?),
            (5, FieldType::I32) => offset_index_length = Some(read_zigzag_i32(cur)?),
            (6, FieldType::I64) => column_index_offset = Some(read_zigzag_i64(cur)?),
            (7, FieldType::I32) => column_index_length = Some(read_zigzag_i32(cur)?),
            (8, FieldType::Struct) => {
                crypto_metadata = Some(read_column_crypto_metadata(cur)?);
            }
            (9, FieldType::Binary) => {
                encrypted_column_metadata = Some(read_binary(cur)?);
            }
            _ => return Err(unknown("ColumnChunk", h.id)),
        }
    }
    Ok(ColumnChunk {
        file_path,
        file_offset: file_offset.ok_or_else(|| missing("ColumnChunk", 2))?,
        meta_data,
        offset_index_offset,
        offset_index_length,
        column_index_offset,
        column_index_length,
        crypto_metadata,
        encrypted_column_metadata,
    })
}

pub fn read_key_value<'a>(cur: &mut Cursor<'a>) -> Result<KeyValue<'a>> {
    let mut key: Option<&[u8]> = None;
    let mut value: Option<&[u8]> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => key = Some(read_binary(cur)?),
            (2, FieldType::Binary) => value = Some(read_binary(cur)?),
            _ => return Err(unknown("KeyValue", h.id)),
        }
    }
    let key = key.ok_or_else(|| missing("KeyValue", 1))?;
    Ok(KeyValue { key, value })
}

fn read_data_page_header<'a>(cur: &mut Cursor<'a>) -> Result<DataPageHeader<'a>> {
    let mut num_values = None;
    let mut encoding = None;
    let mut def_enc = None;
    let mut rep_enc = None;
    let mut statistics = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => num_values = Some(read_zigzag_i32(cur)?),
            (2, FieldType::I32) => encoding = Some(Encoding::read(cur)?),
            (3, FieldType::I32) => def_enc = Some(Encoding::read(cur)?),
            (4, FieldType::I32) => rep_enc = Some(Encoding::read(cur)?),
            (5, FieldType::Struct) => statistics = Some(read_statistics(cur)?),
            _ => return Err(unknown("DataPageHeader", h.id)),
        }
    }
    Ok(DataPageHeader {
        num_values: num_values.ok_or_else(|| missing("DataPageHeader", 1))?,
        encoding: encoding.ok_or_else(|| missing("DataPageHeader", 2))?,
        definition_level_encoding: def_enc.ok_or_else(|| missing("DataPageHeader", 3))?,
        repetition_level_encoding: rep_enc.ok_or_else(|| missing("DataPageHeader", 4))?,
        statistics,
    })
}

fn read_index_page_header(cur: &mut Cursor<'_>) -> Result<IndexPageHeader> {
    // Spec marks the struct as "TODO" — no fields defined. Any field
    // that does appear is unexpected; surface it as format drift.
    if let Some(h) = read_field_header(cur, 0)? {
        return Err(unknown("IndexPageHeader", h.id));
    }
    Ok(IndexPageHeader)
}

fn read_dictionary_page_header(cur: &mut Cursor<'_>) -> Result<DictionaryPageHeader> {
    let mut num_values = None;
    let mut encoding = None;
    let mut is_sorted = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => num_values = Some(read_zigzag_i32(cur)?),
            (2, FieldType::I32) => encoding = Some(Encoding::read(cur)?),
            (3, FieldType::BoolTrue) => is_sorted = Some(true),
            (3, FieldType::BoolFalse) => is_sorted = Some(false),
            _ => return Err(unknown("DictionaryPageHeader", h.id)),
        }
    }
    Ok(DictionaryPageHeader {
        num_values: num_values.ok_or_else(|| missing("DictionaryPageHeader", 1))?,
        encoding: encoding.ok_or_else(|| missing("DictionaryPageHeader", 2))?,
        is_sorted,
    })
}

fn read_data_page_header_v2<'a>(cur: &mut Cursor<'a>) -> Result<DataPageHeaderV2<'a>> {
    let mut num_values = None;
    let mut num_nulls = None;
    let mut num_rows = None;
    let mut encoding = None;
    let mut def_len = None;
    let mut rep_len = None;
    let mut is_compressed = true; // spec default
    let mut statistics = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => num_values = Some(read_zigzag_i32(cur)?),
            (2, FieldType::I32) => num_nulls = Some(read_zigzag_i32(cur)?),
            (3, FieldType::I32) => num_rows = Some(read_zigzag_i32(cur)?),
            (4, FieldType::I32) => encoding = Some(Encoding::read(cur)?),
            (5, FieldType::I32) => def_len = Some(read_zigzag_i32(cur)?),
            (6, FieldType::I32) => rep_len = Some(read_zigzag_i32(cur)?),
            (7, FieldType::BoolTrue) => is_compressed = true,
            (7, FieldType::BoolFalse) => is_compressed = false,
            (8, FieldType::Struct) => statistics = Some(read_statistics(cur)?),
            _ => return Err(unknown("DataPageHeaderV2", h.id)),
        }
    }
    Ok(DataPageHeaderV2 {
        num_values: num_values.ok_or_else(|| missing("DataPageHeaderV2", 1))?,
        num_nulls: num_nulls.ok_or_else(|| missing("DataPageHeaderV2", 2))?,
        num_rows: num_rows.ok_or_else(|| missing("DataPageHeaderV2", 3))?,
        encoding: encoding.ok_or_else(|| missing("DataPageHeaderV2", 4))?,
        definition_levels_byte_length: def_len.ok_or_else(|| missing("DataPageHeaderV2", 5))?,
        repetition_levels_byte_length: rep_len.ok_or_else(|| missing("DataPageHeaderV2", 6))?,
        is_compressed,
        statistics,
    })
}

pub fn read_page_header<'a>(cur: &mut Cursor<'a>) -> Result<PageHeader<'a>> {
    let mut page_type = None;
    let mut uncompressed = None;
    let mut compressed = None;
    let mut crc = None;
    let mut dph = None;
    let mut iph = None;
    let mut dictph = None;
    let mut v2 = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::I32) => page_type = Some(PageType::read(cur)?),
            (2, FieldType::I32) => uncompressed = Some(read_zigzag_i32(cur)?),
            (3, FieldType::I32) => compressed = Some(read_zigzag_i32(cur)?),
            (4, FieldType::I32) => crc = Some(read_zigzag_i32(cur)?),
            (5, FieldType::Struct) => dph = Some(read_data_page_header(cur)?),
            (6, FieldType::Struct) => iph = Some(read_index_page_header(cur)?),
            (7, FieldType::Struct) => dictph = Some(read_dictionary_page_header(cur)?),
            (8, FieldType::Struct) => v2 = Some(read_data_page_header_v2(cur)?),
            _ => return Err(unknown("PageHeader", h.id)),
        }
    }
    Ok(PageHeader {
        page_type: page_type.ok_or_else(|| missing("PageHeader", 1))?,
        uncompressed_page_size: uncompressed.ok_or_else(|| missing("PageHeader", 2))?,
        compressed_page_size: compressed.ok_or_else(|| missing("PageHeader", 3))?,
        crc,
        data_page_header: dph,
        index_page_header: iph,
        dictionary_page_header: dictph,
        data_page_header_v2: v2,
    })
}

pub fn read_statistics<'a>(cur: &mut Cursor<'a>) -> Result<Statistics<'a>> {
    let mut stats = Statistics::default();
    let mut prev_id: i16 = 0;
    while let Some(hdr) = read_field_header(cur, prev_id)? {
        prev_id = hdr.id;
        match (hdr.id, &hdr.field_type) {
            (1, FieldType::Binary) => stats.max = Some(read_binary(cur)?),
            (2, FieldType::Binary) => stats.min = Some(read_binary(cur)?),
            (3, FieldType::I64) => stats.null_count = Some(read_zigzag_i64(cur)?),
            (4, FieldType::I64) => stats.distinct_count = Some(read_zigzag_i64(cur)?),
            (5, FieldType::Binary) => stats.max_value = Some(read_binary(cur)?),
            (6, FieldType::Binary) => stats.min_value = Some(read_binary(cur)?),
            (7, FieldType::BoolTrue) => stats.is_max_value_exact = Some(true),
            (7, FieldType::BoolFalse) => stats.is_max_value_exact = Some(false),
            (8, FieldType::BoolTrue) => stats.is_min_value_exact = Some(true),
            (8, FieldType::BoolFalse) => stats.is_min_value_exact = Some(false),
            _ => {
                return Err(FormatError::UnknownStructField {
                    struct_name: "Statistics",
                    field_id: hdr.id,
                });
            }
        }
    }
    Ok(stats)
}

// ============================================================
// Parquet Modular Encryption (PME) — Π.13a
//
// Metadata-only support: parse the encryption descriptors on the
// footer + per-column-chunk fields. Actual AES-GCM decrypt lives
// in the `ematix-parquet-crypto` crate (Π.13b) and is wired into
// `ematix-parquet-codec` under `--features encryption` (Π.13c+).
//
// Spec: https://github.com/apache/parquet-format/blob/master/Encryption.md
// IDL:  EncryptionAlgorithm union, AesGcmV1, AesGcmCtrV1,
//       FileCryptoMetaData, ColumnCryptoMetaData union,
//       EncryptionWithFooterKey, EncryptionWithColumnKey.
// ============================================================

/// AES-GCM v1 parameters carried in `EncryptionAlgorithm`.
///
/// - `aad_prefix` is optional file-identifier bytes the writer chose
///   to bind into every page's AAD. May be absent (caller-supplied
///   instead — see `supply_aad_prefix`).
/// - `aad_file_unique` is mandatory: 8-32 random bytes the writer
///   generated for this specific file, mixed into every AAD so two
///   files with identical content but different uniques produce
///   incompatible ciphertexts.
/// - `supply_aad_prefix` indicates the writer chose NOT to embed
///   `aad_prefix` in the file; the reader must obtain it from the
///   caller (usually because the prefix itself is sensitive, e.g.
///   contains a customer ID).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AesGcmV1<'a> {
    pub aad_prefix: Option<&'a [u8]>,
    pub aad_file_unique: Option<&'a [u8]>,
    pub supply_aad_prefix: Option<bool>,
}

/// AES-GCM-CTR v1 parameters. Same shape as `AesGcmV1` but indicates
/// the column-data encryption is CTR mode (without authentication tag
/// on the data; metadata is still authenticated).
///
/// v0.6.0 parses this so we can read parquet-rs's encrypted-footer
/// trailers that advertise CTR mode for the column data, but the
/// codec will reject CTR with `UnsupportedAlgorithm` until a real
/// consumer asks for it. Tracked as a v0.6.x follow-up.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AesGcmCtrV1<'a> {
    pub aad_prefix: Option<&'a [u8]>,
    pub aad_file_unique: Option<&'a [u8]>,
    pub supply_aad_prefix: Option<bool>,
}

/// Algorithm union — one of `AesGcmV1` or `AesGcmCtrV1`. The wire
/// representation is a Thrift union: exactly one of fields 1 / 2 is
/// set. Other fields are unknown/error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionAlgorithm<'a> {
    AesGcmV1(AesGcmV1<'a>),
    AesGcmCtrV1(AesGcmCtrV1<'a>),
}

pub fn read_aes_gcm_v1<'a>(cur: &mut Cursor<'a>) -> Result<AesGcmV1<'a>> {
    let mut out = AesGcmV1::default();
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => out.aad_prefix = Some(read_binary(cur)?),
            (2, FieldType::Binary) => out.aad_file_unique = Some(read_binary(cur)?),
            (3, FieldType::BoolTrue) => out.supply_aad_prefix = Some(true),
            (3, FieldType::BoolFalse) => out.supply_aad_prefix = Some(false),
            _ => return Err(unknown("AesGcmV1", h.id)),
        }
    }
    Ok(out)
}

pub fn read_aes_gcm_ctr_v1<'a>(cur: &mut Cursor<'a>) -> Result<AesGcmCtrV1<'a>> {
    let mut out = AesGcmCtrV1::default();
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Binary) => out.aad_prefix = Some(read_binary(cur)?),
            (2, FieldType::Binary) => out.aad_file_unique = Some(read_binary(cur)?),
            (3, FieldType::BoolTrue) => out.supply_aad_prefix = Some(true),
            (3, FieldType::BoolFalse) => out.supply_aad_prefix = Some(false),
            _ => return Err(unknown("AesGcmCtrV1", h.id)),
        }
    }
    Ok(out)
}

pub fn read_encryption_algorithm<'a>(cur: &mut Cursor<'a>) -> Result<EncryptionAlgorithm<'a>> {
    let mut chosen: Option<EncryptionAlgorithm<'a>> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                chosen = Some(EncryptionAlgorithm::AesGcmV1(read_aes_gcm_v1(cur)?));
            }
            (2, FieldType::Struct) => {
                chosen = Some(EncryptionAlgorithm::AesGcmCtrV1(read_aes_gcm_ctr_v1(cur)?));
            }
            _ => return Err(unknown("EncryptionAlgorithm", h.id)),
        }
    }
    chosen.ok_or(FormatError::EmptyUnion {
        union_name: "EncryptionAlgorithm",
    })
}

/// `EncryptionWithColumnKey` — per-column key path. The
/// `path_in_schema` is the dotted schema path naming the column being
/// encrypted (matches `ColumnMetaData.path_in_schema`). `key_metadata`
/// is caller-opaque bytes identifying the key.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EncryptionWithColumnKey<'a> {
    pub path_in_schema: Vec<&'a [u8]>,
    pub key_metadata: Option<&'a [u8]>,
}

/// `ColumnCryptoMetaData` union — exactly one of two variants:
/// `EncryptionWithFooterKey` (this column uses the file's footer key)
/// or `EncryptionWithColumnKey` (per-column key with optional metadata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnCryptoMetaData<'a> {
    EncryptionWithFooterKey,
    EncryptionWithColumnKey(EncryptionWithColumnKey<'a>),
}

pub fn read_encryption_with_column_key<'a>(
    cur: &mut Cursor<'a>,
) -> Result<EncryptionWithColumnKey<'a>> {
    let mut path_in_schema: Option<Vec<&'a [u8]>> = None;
    let mut key_metadata: Option<&'a [u8]> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::List) => path_in_schema = Some(read_list_binary(cur)?),
            (2, FieldType::Binary) => key_metadata = Some(read_binary(cur)?),
            _ => return Err(unknown("EncryptionWithColumnKey", h.id)),
        }
    }
    Ok(EncryptionWithColumnKey {
        path_in_schema: path_in_schema.ok_or_else(|| missing("EncryptionWithColumnKey", 1))?,
        key_metadata,
    })
}

/// `EncryptionWithFooterKey` is a marker struct (no fields) in the
/// spec — the variant tag in the parent union is the entire payload.
/// We accept it as an empty struct (STOP byte immediately) and error
/// on any field.
fn read_encryption_with_footer_key(cur: &mut Cursor<'_>) -> Result<()> {
    if let Some(h) = read_field_header(cur, 0)? {
        return Err(unknown("EncryptionWithFooterKey", h.id));
    }
    Ok(())
}

pub fn read_column_crypto_metadata<'a>(cur: &mut Cursor<'a>) -> Result<ColumnCryptoMetaData<'a>> {
    let mut chosen: Option<ColumnCryptoMetaData<'a>> = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                read_encryption_with_footer_key(cur)?;
                chosen = Some(ColumnCryptoMetaData::EncryptionWithFooterKey);
            }
            (2, FieldType::Struct) => {
                chosen = Some(ColumnCryptoMetaData::EncryptionWithColumnKey(
                    read_encryption_with_column_key(cur)?,
                ));
            }
            _ => return Err(unknown("ColumnCryptoMetaData", h.id)),
        }
    }
    chosen.ok_or(FormatError::EmptyUnion {
        union_name: "ColumnCryptoMetaData",
    })
}

/// `FileCryptoMetaData` — the **encrypted-footer** trailer that
/// replaces `FileMetaData` on disk. The reader recognises this mode by
/// the `PARE` magic (vs `PAR1`/`PAR2` for unencrypted/plaintext-footer
/// files) and decodes this struct from the bytes immediately before
/// the magic. `encryption_algorithm` describes how the actual
/// `FileMetaData` ciphertext (which follows this struct, also before
/// the magic) was encrypted; `key_metadata` identifies the key.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FileCryptoMetaData<'a> {
    pub encryption_algorithm: Option<EncryptionAlgorithm<'a>>,
    pub key_metadata: Option<&'a [u8]>,
}

pub fn read_file_crypto_metadata<'a>(cur: &mut Cursor<'a>) -> Result<FileCryptoMetaData<'a>> {
    let mut encryption_algorithm = None;
    let mut key_metadata = None;
    let mut prev = 0;
    while let Some(h) = read_field_header(cur, prev)? {
        prev = h.id;
        match (h.id, &h.field_type) {
            (1, FieldType::Struct) => {
                encryption_algorithm = Some(read_encryption_algorithm(cur)?);
            }
            (2, FieldType::Binary) => key_metadata = Some(read_binary(cur)?),
            _ => return Err(unknown("FileCryptoMetaData", h.id)),
        }
    }
    let encryption_algorithm =
        Some(encryption_algorithm.ok_or_else(|| missing("FileCryptoMetaData", 1))?);
    Ok(FileCryptoMetaData {
        encryption_algorithm,
        key_metadata,
    })
}
