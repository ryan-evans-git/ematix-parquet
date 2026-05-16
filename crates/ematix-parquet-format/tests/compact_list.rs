//! TDD pin for the thrift compact list header reader.
//!
//! Spec:
//!   list_header = (size_or_F << 4) | element_type
//!   - If size < 15: high nibble carries it directly, no follow-up.
//!   - If size >= 15: high nibble = 0xF, then `uvarint(size)` follows.
//!   element_type uses the same FieldType nibbles as struct fields
//!   (1=BoolTrue, 5=I32, 8=Binary, 12=Struct, etc.).
//!
//! Elements have no per-element headers — the caller decodes `count`
//! bodies of the matching element type from the cursor.

use ematix_parquet_format::compact::{read_list_header, Cursor, FieldType};
use ematix_parquet_format::error::FormatError;

#[test]
fn empty_list_of_i32() {
    // count=0, type=I32(5) → header byte (0<<4)|5 = 0x05
    let bytes = [0x05u8];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!(count, 0);
    assert_eq!(et, FieldType::I32);
    assert_eq!(cur.position(), 1);
}

#[test]
fn inline_count_3_of_binary() {
    // count=3, type=Binary(8) → header (3<<4)|8 = 0x38
    let bytes = [0x38u8];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!(count, 3);
    assert_eq!(et, FieldType::Binary);
    assert_eq!(cur.position(), 1);
}

#[test]
fn max_inline_count_14() {
    // count=14, type=Struct(12) → header (14<<4)|12 = 0xEC
    let bytes = [0xECu8];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!(count, 14);
    assert_eq!(et, FieldType::Struct);
}

#[test]
fn count_15_uses_long_form() {
    // count=15 cannot fit in 4 bits → high nibble = 0xF, then uvarint(15).
    //   header byte: (0xF<<4)|5 = 0xF5
    //   uvarint(15): 0x0F
    let bytes = [0xF5u8, 0x0F];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!(count, 15);
    assert_eq!(et, FieldType::I32);
    assert_eq!(cur.position(), 2);
}

#[test]
fn long_form_count_1000() {
    // count=1000, type=I32(5)
    //   header: 0xF5
    //   uvarint(1000): 1000 = 0b1111101000 → low 7 = 0x68, high = 0x07
    //     byte0 = 0x68 | 0x80 = 0xE8 (cont)
    //     byte1 = 0x07 (terminal)
    let bytes = [0xF5u8, 0xE8, 0x07];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!(count, 1000);
    assert_eq!(et, FieldType::I32);
    assert_eq!(cur.position(), 3);
}

#[test]
fn invalid_element_type_nibble_rejected() {
    // High nibble valid (count=1), low nibble 13 invalid.
    let bytes = [0x1Du8];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_list_header(&mut cur),
        Err(FormatError::InvalidFieldType(0xD))
    ));
}

#[test]
fn end_to_end_decode_list_of_3_i32() {
    // Header: count=3, type=I32 → 0x35
    // Bodies: zigzag(1)=0x02, zigzag(2)=0x04, zigzag(3)=0x06
    let bytes = [0x35u8, 0x02, 0x04, 0x06];
    let mut cur = Cursor::new(&bytes);
    let (count, et) = read_list_header(&mut cur).unwrap();
    assert_eq!((count, et), (3, FieldType::I32));
    use ematix_parquet_format::compact::read_zigzag_i32;
    let values: Vec<i32> = (0..count)
        .map(|_| read_zigzag_i32(&mut cur).unwrap())
        .collect();
    assert_eq!(values, vec![1, 2, 3]);
}
