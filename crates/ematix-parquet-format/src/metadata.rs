//! Parquet metadata structs decoded from the thrift compact protocol.
//!
//! All struct readers are zero-copy: variable-length binary fields
//! borrow `&[u8]` from the cursor's underlying buffer. Callers that
//! need owned data can `.to_vec()` the slices.

use crate::compact::{
    read_binary, read_field_header, read_zigzag_i32, read_zigzag_i64, Cursor, FieldType,
};
use crate::error::{FormatError, Result};
use crate::types::{Encoding, PageType, ThriftEnum};

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
    FormatError::MissingRequiredField { struct_name, field_id }
}

fn unknown(struct_name: &'static str, field_id: i16) -> FormatError {
    FormatError::UnknownStructField { struct_name, field_id }
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
    // Spec marks the struct as "TODO" — no fields defined. Walk to STOP,
    // erroring on any field that does appear so we notice format drift.
    while let Some(h) = read_field_header(cur, 0)? {
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
