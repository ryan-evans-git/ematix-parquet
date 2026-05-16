//! TDD pin for the compact-protocol struct walker.
//!
//! Spec recap:
//!   field header byte = (delta_id << 4) | type_nibble
//!   - delta_id ∈ [1..15] → next field id = prev_id + delta_id
//!   - delta_id == 0      → long form, explicit zigzag i16 field id follows
//!   - byte == 0x00       → STOP
//!     embedded booleans: type_nibble = 1 (true) or 2 (false); no body bytes
//!
//! Caller threads `prev_id` through `read_field_header` and decodes the
//! body based on `FieldType`. We test the header reader plus a thin
//! state holder.

use ematix_parquet_format::compact::{
    read_field_header, read_zigzag_i32, Cursor, FieldHeader, FieldType,
};
use ematix_parquet_format::error::FormatError;

#[test]
fn empty_struct_is_just_stop() {
    let bytes = [0x00u8];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_field_header(&mut cur, 0).unwrap(), None);
    assert_eq!(cur.position(), 1);
}

#[test]
fn single_i32_field_delta_form() {
    // field id=1, type I32, value=5 (zigzag 0x0A)
    //   header byte: (1<<4)|5 = 0x15
    //   body:        0x0A
    //   stop:        0x00
    let bytes = [0x15, 0x0A, 0x00];
    let mut cur = Cursor::new(&bytes);

    let h = read_field_header(&mut cur, 0).unwrap().unwrap();
    assert_eq!(
        h,
        FieldHeader {
            id: 1,
            field_type: FieldType::I32,
        }
    );
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 5);

    assert_eq!(read_field_header(&mut cur, h.id).unwrap(), None);
    assert_eq!(cur.position(), 3);
}

#[test]
fn embedded_bool_true_has_no_body() {
    // field id=1, type BoolTrue (embedded)
    //   header byte: (1<<4)|1 = 0x11
    //   stop:        0x00
    let bytes = [0x11, 0x00];
    let mut cur = Cursor::new(&bytes);

    let h = read_field_header(&mut cur, 0).unwrap().unwrap();
    assert_eq!(
        h,
        FieldHeader {
            id: 1,
            field_type: FieldType::BoolTrue,
        }
    );
    // Position must be just past the header (no body consumed).
    assert_eq!(cur.position(), 1);

    assert_eq!(read_field_header(&mut cur, h.id).unwrap(), None);
    assert_eq!(cur.position(), 2);
}

#[test]
fn embedded_bool_false_has_no_body() {
    // field id=1, type BoolFalse (embedded)
    let bytes = [0x12, 0x00];
    let mut cur = Cursor::new(&bytes);

    let h = read_field_header(&mut cur, 0).unwrap().unwrap();
    assert_eq!(
        h,
        FieldHeader {
            id: 1,
            field_type: FieldType::BoolFalse,
        }
    );
    assert_eq!(cur.position(), 1);
}

#[test]
fn two_fields_with_delta_encoding() {
    // field id=1 (delta=1), I32 value=1 (zigzag 0x02)
    // field id=4 (delta=3), I32 value=2 (zigzag 0x04)
    let bytes = [0x15, 0x02, 0x35, 0x04, 0x00];
    let mut cur = Cursor::new(&bytes);

    let h1 = read_field_header(&mut cur, 0).unwrap().unwrap();
    assert_eq!(h1.id, 1);
    assert_eq!(h1.field_type, FieldType::I32);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 1);

    let h2 = read_field_header(&mut cur, h1.id).unwrap().unwrap();
    assert_eq!(h2.id, 4);
    assert_eq!(h2.field_type, FieldType::I32);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 2);

    assert_eq!(read_field_header(&mut cur, h2.id).unwrap(), None);
}

#[test]
fn long_form_field_id() {
    // field id=20 (delta=20 > 15, must use long form), I32 value=5
    //   header byte: (0<<4)|5 = 0x05
    //   explicit id (zigzag i16): zigzag(20) = 40 = 0x28
    //   body:                     0x0A
    //   stop:                     0x00
    let bytes = [0x05, 0x28, 0x0A, 0x00];
    let mut cur = Cursor::new(&bytes);

    let h = read_field_header(&mut cur, 0).unwrap().unwrap();
    assert_eq!(h.id, 20);
    assert_eq!(h.field_type, FieldType::I32);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 5);
    assert_eq!(read_field_header(&mut cur, h.id).unwrap(), None);
}

#[test]
fn long_form_does_not_depend_on_prev_id() {
    // Long form sets id directly (not relative). prev_id=99 should be
    // ignored.
    let bytes = [0x05, 0x28, 0x0A, 0x00];
    let mut cur = Cursor::new(&bytes);
    let h = read_field_header(&mut cur, 99).unwrap().unwrap();
    assert_eq!(h.id, 20, "long-form id must be absolute, not prev_id + 20");
}

#[test]
fn invalid_field_type_nibble_is_rejected() {
    // Type nibble 13 (0xD) is not a valid thrift compact type.
    let bytes = [0x1D, 0x00];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_field_header(&mut cur, 0),
        Err(FormatError::InvalidFieldType(0xD))
    ));
}

#[test]
fn all_valid_type_nibbles_round_trip() {
    // For each of the 12 valid type codes, a header with delta=1 should
    // decode to id=1 + the matching FieldType. We don't decode bodies
    // here — just verify the dispatch table.
    let cases: &[(u8, FieldType)] = &[
        (1, FieldType::BoolTrue),
        (2, FieldType::BoolFalse),
        (3, FieldType::Byte),
        (4, FieldType::I16),
        (5, FieldType::I32),
        (6, FieldType::I64),
        (7, FieldType::Double),
        (8, FieldType::Binary),
        (9, FieldType::List),
        (10, FieldType::Set),
        (11, FieldType::Map),
        (12, FieldType::Struct),
    ];
    for &(nibble, ref expected) in cases {
        let header = (1 << 4) | nibble;
        let bytes = [header];
        let mut cur = Cursor::new(&bytes);
        let h = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!(h.id, 1, "id wrong for type nibble {nibble}");
        assert_eq!(&h.field_type, expected, "type wrong for nibble {nibble}");
    }
}
