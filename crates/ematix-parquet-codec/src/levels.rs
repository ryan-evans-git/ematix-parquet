//! Definition + repetition level decoding for v1 data pages.
//!
//! Parquet records nullability and nesting at the page-body level via
//! two parallel RLE/bit-packed streams that precede the value bytes:
//!
//!   v1 page body = [rep_levels][def_levels][values]
//!
//! Each level stream — when present — is prefixed by a 4-byte LE
//! `u32` byte length and encoded with the
//! [`crate::rle::decode_rle_bit_packed`] primitive at
//! `bit_width = ceil(log2(max_level + 1))`. The stream is OMITTED
//! entirely (no length prefix, no bytes) when its max level is 0.
//!
//! For REQUIRED non-nested columns (max_def_level = 0, max_rep_level = 0)
//! both streams are absent and the page body IS the value bytes —
//! which is what lineitem (TPC-H) hits in every column.
//!
//! For nullable scalar columns (max_def_level = 1, max_rep_level = 0):
//!   - rep stream omitted
//!   - def stream present, bit_width = 1
//!     a row's value is present iff def_level == 1
//!
//! v2 data pages encode the levels uncompressed in fixed-length
//! regions ahead of the (possibly compressed) values, with the lengths
//! stored on the page header itself. Not handled here yet.

use ematix_parquet_format::types::FieldRepetitionType;
use ematix_parquet_format::metadata::SchemaElement;

use crate::error::{CodecError, Result};
use crate::rle::decode_rle_bit_packed;

/// Bit width needed to carry `0..=max_level` values.
///
///   bit_width_for(0) = 0          (level always 0; nothing on the wire)
///   bit_width_for(1) = 1          (typical nullable scalar)
///   bit_width_for(2..=3) = 2
///   bit_width_for(4..=7) = 3
///   ...
///
/// Equivalent to `ceil(log2(max_level + 1))` for `max_level >= 1`.
pub fn bit_width_for(max_level: u16) -> u8 {
    if max_level == 0 {
        0
    } else {
        (32 - (max_level as u32).leading_zeros()) as u8
    }
}

/// Decode a level stream out of the start of a v1 data-page body.
///
/// Returns `(levels, bytes_consumed)`. When `bit_width == 0` the spec
/// says the stream is omitted on the wire; we synthesize `num_values`
/// zeros and return `bytes_consumed = 0`.
///
/// Otherwise: reads a 4-byte LE length prefix, then decodes the
/// RLE/bit-packed stream of that length at the given bit width.
pub fn decode_levels(body: &[u8], bit_width: u8, num_values: usize) -> Result<(Vec<u16>, usize)> {
    if bit_width == 0 {
        return Ok((vec![0u16; num_values], 0));
    }
    if body.len() < 4 {
        return Err(CodecError::Decompress(format!(
            "level stream: need 4-byte length prefix, have {}",
            body.len()
        )));
    }
    let len = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
    let end = 4 + len;
    if body.len() < end {
        return Err(CodecError::Decompress(format!(
            "level stream: prefix says {len} body bytes, have {}",
            body.len() - 4
        )));
    }
    let level_bytes = &body[4..end];
    let levels: Vec<u16> = decode_rle_bit_packed(level_bytes, bit_width, num_values)?
        .into_iter()
        .map(|v| v as u16)
        .collect();
    Ok((levels, end))
}

/// Slice the rep + def level streams off the front of a v1 data-page
/// body and return them along with the remaining values-section
/// bytes. `max_rep_level`/`max_def_level` come from the column-chunk
/// schema (computed via `bit_width_for`).
///
/// Order on the wire is `rep` then `def` (rep levels come first).
pub fn parse_v1_data_page_body(
    body: &[u8],
    max_rep_level: u16,
    max_def_level: u16,
    num_values: usize,
) -> Result<(Vec<u16>, Vec<u16>, &[u8])> {
    let rep_bw = bit_width_for(max_rep_level);
    let def_bw = bit_width_for(max_def_level);

    let mut off = 0;
    let (rep, rep_consumed) = decode_levels(&body[off..], rep_bw, num_values)?;
    off += rep_consumed;
    let (def, def_consumed) = decode_levels(&body[off..], def_bw, num_values)?;
    off += def_consumed;

    Ok((rep, def, &body[off..]))
}

/// Q06.c2 (2026-05-24) fast path: return the byte offset where the
/// values section begins in a v1 page body, without materializing the
/// rep + def level vectors. This is what callers that just want the
/// values slice (e.g. `data_page_view` feeding the
/// dictionary-index / plain-value decoders) actually need — for a
/// REQUIRED-elsewhere column with `max_def_level == 1` and 1M values,
/// the levels stream is a few RLE-compact bytes that expands to a
/// 2 MB `Vec<u16>` of all-1s under the materializing path. Skipping
/// saves that allocation per page.
///
/// Returns the offset past both level prefixes. For REQUIRED non-
/// nested columns (`max_rep_level == 0 && max_def_level == 0`) this
/// is always 0.
pub fn skip_v1_level_prefixes(
    body: &[u8],
    max_rep_level: u16,
    max_def_level: u16,
) -> Result<usize> {
    let mut off = 0;
    for bw in [bit_width_for(max_rep_level), bit_width_for(max_def_level)] {
        if bw == 0 {
            continue;
        }
        if body.len() < off + 4 {
            return Err(CodecError::Decompress(format!(
                "level stream: need 4-byte length prefix at offset {off}, body.len()={}",
                body.len()
            )));
        }
        let len = u32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as usize;
        let end = off + 4 + len;
        if body.len() < end {
            return Err(CodecError::Decompress(format!(
                "level stream: prefix says {len} body bytes at offset {off}, body.len()={}",
                body.len()
            )));
        }
        off = end;
    }
    Ok(off)
}

/// Compute `(max_rep_level, max_def_level)` for the leaf at
/// `leaf_index` within a parquet schema vector. The schema is encoded
/// depth-first; element 0 is the root group (whose levels contribute
/// nothing), and `num_children` describes each group's subtree size.
///
/// Per the parquet spec:
/// - `Required` field: levels unchanged.
/// - `Optional` field: `max_def_level += 1`.
/// - `Repeated` field: `max_rep_level += 1` AND `max_def_level += 1`.
///
/// Returns an error if `leaf_index` is past the number of leaves.
pub fn compute_max_levels(
    schema: &[SchemaElement<'_>],
    leaf_index: usize,
) -> Result<(u16, u16)> {
    if schema.is_empty() {
        return Err(CodecError::InvalidInput("empty schema".into()));
    }
    // Stack frame: (max_rep, max_def, remaining_children_at_this_level).
    // Root frame seeded with (0, 0, root.num_children).
    let root_children = schema[0].num_children.unwrap_or(0);
    if root_children <= 0 {
        return Err(CodecError::InvalidInput(
            "schema root has no children".into(),
        ));
    }
    let mut stack: Vec<(u16, u16, i32)> = Vec::with_capacity(8);
    stack.push((0, 0, root_children));
    let mut leaf_count: usize = 0;

    for el in &schema[1..] {
        let (parent_rep, parent_def, _) = *stack
            .last()
            .ok_or_else(|| CodecError::InvalidInput("schema tree underflow".into()))?;
        let (this_rep, this_def) = match el.repetition_type {
            Some(FieldRepetitionType::Required) => (parent_rep, parent_def),
            Some(FieldRepetitionType::Optional) => (parent_rep, parent_def + 1),
            Some(FieldRepetitionType::Repeated) => (parent_rep + 1, parent_def + 1),
            None => (parent_rep, parent_def),
        };
        let n_children = el.num_children.unwrap_or(0);
        if n_children == 0 {
            if leaf_count == leaf_index {
                return Ok((this_rep, this_def));
            }
            leaf_count += 1;
            // Account for this leaf in its parent's remaining child count
            // and unwind any parents whose subtree just finished.
            if let Some(top) = stack.last_mut() {
                top.2 -= 1;
            }
            while let Some(&(_, _, remaining)) = stack.last() {
                if remaining <= 0 && stack.len() > 1 {
                    stack.pop();
                    if let Some(parent) = stack.last_mut() {
                        parent.2 -= 1;
                    }
                } else {
                    break;
                }
            }
        } else {
            // Group node — push its frame and descend.
            stack.push((this_rep, this_def, n_children));
        }
    }
    Err(CodecError::InvalidInput(format!(
        "leaf index {leaf_index} out of range (schema has {leaf_count} leaves)"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ematix_parquet_format::metadata::SchemaElement;
    use ematix_parquet_format::types::{FieldRepetitionType, ParquetType};

    fn root(n_children: i32) -> SchemaElement<'static> {
        SchemaElement {
            column_type: None,
            type_length: None,
            repetition_type: None,
            name: b"root",
            num_children: Some(n_children),
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        }
    }

    fn leaf(rep: FieldRepetitionType) -> SchemaElement<'static> {
        SchemaElement {
            column_type: Some(ParquetType::Int32),
            type_length: None,
            repetition_type: Some(rep),
            name: b"leaf",
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        }
    }

    #[test]
    fn flat_required_column_has_zero_levels() {
        let schema = vec![root(1), leaf(FieldRepetitionType::Required)];
        let (rep, def) = compute_max_levels(&schema, 0).unwrap();
        assert_eq!(rep, 0);
        assert_eq!(def, 0);
    }

    #[test]
    fn flat_optional_column_has_def_level_one() {
        let schema = vec![root(1), leaf(FieldRepetitionType::Optional)];
        let (rep, def) = compute_max_levels(&schema, 0).unwrap();
        assert_eq!(rep, 0);
        assert_eq!(def, 1);
    }

    #[test]
    fn flat_repeated_column_has_both_levels_one() {
        let schema = vec![root(1), leaf(FieldRepetitionType::Repeated)];
        let (rep, def) = compute_max_levels(&schema, 0).unwrap();
        assert_eq!(rep, 1);
        assert_eq!(def, 1);
    }

    #[test]
    fn tpch_lineitem_flat_each_column_independent() {
        // 16-column flat schema: alternating Required and Optional.
        let mut schema = vec![root(16)];
        for i in 0..16 {
            schema.push(leaf(if i % 2 == 0 {
                FieldRepetitionType::Required
            } else {
                FieldRepetitionType::Optional
            }));
        }
        for i in 0..16 {
            let (rep, def) = compute_max_levels(&schema, i).unwrap();
            assert_eq!(rep, 0, "leaf {i} rep");
            assert_eq!(def, if i % 2 == 0 { 0 } else { 1 }, "leaf {i} def");
        }
    }

    #[test]
    fn leaf_index_out_of_range_errors() {
        let schema = vec![root(1), leaf(FieldRepetitionType::Required)];
        assert!(compute_max_levels(&schema, 1).is_err());
    }
}
