//! Parquet metadata structs decoded from the thrift compact protocol.
//!
//! All struct readers are zero-copy: variable-length binary fields
//! borrow `&[u8]` from the cursor's underlying buffer. Callers that
//! need owned data can `.to_vec()` the slices.

use crate::compact::{
    read_binary, read_field_header, read_zigzag_i64, Cursor, FieldType,
};
use crate::error::{FormatError, Result};

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
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Statistics<'a> {
    pub max: Option<&'a [u8]>,
    pub min: Option<&'a [u8]>,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<&'a [u8]>,
    pub min_value: Option<&'a [u8]>,
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
