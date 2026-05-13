//! Thrift compact-protocol primitive writers.
//!
//! Mirror of `compact` (the reader side). The encoder writes into a
//! `Vec<u8>` rather than a borrowed slice, so callers can build up
//! metadata blocks before stamping them into a Parquet file.
//!
//! Wire-spec reference: see `compact.rs`. Every function here has
//! an inverse function over there; round-trip is the test contract.

use crate::compact::FieldType;

/// Owned write-buffer for compact-protocol output. Cheap to construct
/// from an empty `Vec`; callers can also `with_capacity(...)` if they
/// know the final size ahead of time.
#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn write_u8(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub fn write_i8(&mut self, b: i8) {
        self.buf.push(b as u8);
    }

    /// Unsigned LEB128 varint. 7 bits of payload per byte, MSB set on
    /// all but the last byte.
    pub fn write_uvarint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push(((v & 0x7f) as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    /// Zigzag-encoded signed 32, written as a uvarint.
    pub fn write_zigzag_i32(&mut self, v: i32) {
        let zz = ((v << 1) ^ (v >> 31)) as u32;
        self.write_uvarint(zz as u64);
    }

    /// Zigzag-encoded signed 64.
    pub fn write_zigzag_i64(&mut self, v: i64) {
        let zz = ((v << 1) ^ (v >> 63)) as u64;
        self.write_uvarint(zz);
    }

    /// Zigzag-encoded signed 16. Wire-identical to a 32-bit zigzag.
    pub fn write_zigzag_i16(&mut self, v: i16) {
        self.write_zigzag_i32(v as i32);
    }

    /// IEEE-754 double, little-endian (8 bytes).
    pub fn write_double(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Length-prefixed byte slice. Length is a uvarint.
    pub fn write_binary(&mut self, bytes: &[u8]) {
        self.write_uvarint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    /// Struct field header.
    ///
    /// Wire form: one byte `(delta << 4) | type`. If the new id is
    /// strictly greater than `prev_id` by 1..=15, packs into the
    /// short form. Otherwise falls back to long form: byte
    /// `(0 << 4) | type`, then a zigzag-i16 with the absolute id.
    ///
    /// `prev_id` is the most recently written field id within the
    /// current struct (0 for the first field). The caller is
    /// responsible for threading it.
    pub fn write_field_header(
        &mut self,
        id: i16,
        field_type: FieldType,
        prev_id: i16,
    ) {
        let type_code = field_type as u8;
        let diff = id as i32 - prev_id as i32;
        if diff > 0 && diff <= 15 {
            self.buf.push(((diff as u8) << 4) | type_code);
        } else {
            // Long form: high nibble = 0, low nibble = type, then
            // explicit zigzag i16 with the absolute id.
            self.buf.push(type_code);
            self.write_zigzag_i16(id);
        }
    }

    /// STOP byte (terminates a struct).
    pub fn write_field_stop(&mut self) {
        self.buf.push(0x00);
    }

    /// List header. `count` ≤ 14 packs into the high nibble; otherwise
    /// the high nibble is `0xF` and the count follows as a uvarint.
    pub fn write_list_header(&mut self, count: usize, element_type: FieldType) {
        let type_code = element_type as u8;
        if count < 15 {
            self.buf.push(((count as u8) << 4) | type_code);
        } else {
            self.buf.push(0xF0 | type_code);
            self.write_uvarint(count as u64);
        }
    }

    /// Convenience: list of i32. Each element is zigzag-encoded.
    pub fn write_list_i32(&mut self, values: &[i32]) {
        self.write_list_header(values.len(), FieldType::I32);
        for &v in values {
            self.write_zigzag_i32(v);
        }
    }

    /// Convenience: list of i64.
    pub fn write_list_i64(&mut self, values: &[i64]) {
        self.write_list_header(values.len(), FieldType::I64);
        for &v in values {
            self.write_zigzag_i64(v);
        }
    }

    /// Convenience: list of bool. Each element is encoded as the
    /// `BoolTrue` / `BoolFalse` field type code in a single byte.
    pub fn write_list_bool(&mut self, values: &[bool]) {
        self.write_list_header(values.len(), FieldType::BoolTrue);
        for &v in values {
            self.buf
                .push(if v { FieldType::BoolTrue as u8 } else { FieldType::BoolFalse as u8 });
        }
    }

    /// Convenience: list of length-prefixed byte slices.
    pub fn write_list_binary(&mut self, items: &[&[u8]]) {
        self.write_list_header(items.len(), FieldType::Binary);
        for b in items {
            self.write_binary(b);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Round-trip tests: encode with `Writer`, decode with the existing
    //! `compact` reader, assert equality. The contract is "writer and
    //! reader are inverses for every primitive the protocol defines."

    use super::*;
    use crate::compact::{
        read_field_header, read_list_header, read_uvarint, read_zigzag_i16, read_zigzag_i32,
        read_zigzag_i64, Cursor,
    };

    fn rt_uvarint(v: u64) {
        let mut w = Writer::new();
        w.write_uvarint(v);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = read_uvarint(&mut cur).unwrap();
        assert_eq!(decoded, v, "uvarint round-trip");
        assert_eq!(cur.remaining(), 0);
    }

    #[test]
    fn uvarint_roundtrip_corner_values() {
        for v in [0u64, 1, 127, 128, 255, 256, 16_383, 16_384, u32::MAX as u64, u64::MAX] {
            rt_uvarint(v);
        }
    }

    fn rt_zz32(v: i32) {
        let mut w = Writer::new();
        w.write_zigzag_i32(v);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        assert_eq!(read_zigzag_i32(&mut cur).unwrap(), v);
    }

    #[test]
    fn zigzag_i32_roundtrip() {
        for v in [0i32, 1, -1, 127, -128, i32::MAX, i32::MIN, 12345, -67890] {
            rt_zz32(v);
        }
    }

    fn rt_zz64(v: i64) {
        let mut w = Writer::new();
        w.write_zigzag_i64(v);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        assert_eq!(read_zigzag_i64(&mut cur).unwrap(), v);
    }

    #[test]
    fn zigzag_i64_roundtrip() {
        for v in [0i64, 1, -1, i64::MAX, i64::MIN, 1_000_000_000_000] {
            rt_zz64(v);
        }
    }

    #[test]
    fn zigzag_i16_roundtrip() {
        let mut w = Writer::new();
        for v in [0i16, 1, -1, i16::MAX, i16::MIN, 12345, -12345] {
            w.write_zigzag_i16(v);
        }
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        for v in [0i16, 1, -1, i16::MAX, i16::MIN, 12345, -12345] {
            assert_eq!(read_zigzag_i16(&mut cur).unwrap(), v);
        }
    }

    #[test]
    fn field_header_delta_short_form() {
        // ids 1, 2, 5 with prev_id chain → all fit in 4-bit delta.
        let mut w = Writer::new();
        w.write_field_header(1, FieldType::I32, 0);
        w.write_field_header(2, FieldType::I64, 1);
        w.write_field_header(5, FieldType::Binary, 2);
        let bytes = w.into_bytes();

        let mut cur = Cursor::new(&bytes);
        let h1 = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!(h1.id, 1);
        assert_eq!(h1.field_type, FieldType::I32);
        let h2 = read_field_header(&mut cur, h1.id).unwrap().unwrap();
        assert_eq!(h2.id, 2);
        assert_eq!(h2.field_type, FieldType::I64);
        let h3 = read_field_header(&mut cur, h2.id).unwrap().unwrap();
        assert_eq!(h3.id, 5);
        assert_eq!(h3.field_type, FieldType::Binary);
    }

    #[test]
    fn field_header_long_form_when_jump_too_big() {
        // id jump 0 → 100 — delta > 15, must fall back to long form.
        let mut w = Writer::new();
        w.write_field_header(100, FieldType::Struct, 0);
        let bytes = w.into_bytes();
        // First byte should have high nibble 0 (long form indicator).
        assert_eq!(bytes[0] & 0xF0, 0);
        let mut cur = Cursor::new(&bytes);
        let h = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!(h.id, 100);
        assert_eq!(h.field_type, FieldType::Struct);
    }

    #[test]
    fn field_header_long_form_when_negative_id() {
        // Negative id (rare but legal): forces long form because the
        // delta can't be encoded as a 4-bit positive nibble.
        let mut w = Writer::new();
        w.write_field_header(-3, FieldType::Byte, 0);
        let bytes = w.into_bytes();
        assert_eq!(bytes[0] & 0xF0, 0);
        let mut cur = Cursor::new(&bytes);
        let h = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!(h.id, -3);
    }

    #[test]
    fn field_stop_terminates() {
        let mut w = Writer::new();
        w.write_field_header(1, FieldType::I32, 0);
        w.write_zigzag_i32(42);
        w.write_field_stop();
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let h = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!(h.id, 1);
        let v = read_zigzag_i32(&mut cur).unwrap();
        assert_eq!(v, 42);
        // Next field header should hit STOP and return None.
        assert!(read_field_header(&mut cur, h.id).unwrap().is_none());
    }

    #[test]
    fn list_header_short_form() {
        let mut w = Writer::new();
        w.write_list_header(7, FieldType::I64);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let (n, t) = read_list_header(&mut cur).unwrap();
        assert_eq!(n, 7);
        assert_eq!(t, FieldType::I64);
    }

    #[test]
    fn list_header_long_form_when_count_large() {
        let mut w = Writer::new();
        w.write_list_header(1000, FieldType::Binary);
        let bytes = w.into_bytes();
        assert_eq!(bytes[0] & 0xF0, 0xF0);
        let mut cur = Cursor::new(&bytes);
        let (n, t) = read_list_header(&mut cur).unwrap();
        assert_eq!(n, 1000);
        assert_eq!(t, FieldType::Binary);
    }

    #[test]
    fn list_i32_roundtrip() {
        let values: Vec<i32> = vec![0, 1, -1, 42, -42, i32::MAX, i32::MIN];
        let mut w = Writer::new();
        w.write_list_i32(&values);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = crate::compact::read_list_i32(&mut cur).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn list_i64_roundtrip() {
        let values: Vec<i64> = vec![0, 1, -1, i64::MAX, i64::MIN, 1234567890123];
        let mut w = Writer::new();
        w.write_list_i64(&values);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = crate::compact::read_list_i64(&mut cur).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn list_bool_roundtrip() {
        let values: Vec<bool> = vec![true, false, true, true, false];
        let mut w = Writer::new();
        w.write_list_bool(&values);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = crate::compact::read_list_bool(&mut cur).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn list_binary_roundtrip() {
        let strings: Vec<&[u8]> = vec![b"hello", b"", b"world!", b"\x00\x01\x02"];
        let mut w = Writer::new();
        w.write_list_binary(&strings);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = crate::compact::read_list_binary(&mut cur).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn binary_roundtrip() {
        let payload = b"the quick brown fox";
        let mut w = Writer::new();
        w.write_binary(payload);
        let bytes = w.into_bytes();
        let mut cur = Cursor::new(&bytes);
        let decoded = crate::compact::read_binary(&mut cur).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn double_roundtrip() {
        // Doubles are written as 8 raw LE bytes (no zigzag, no varint).
        // There's no `read_double` in the reader yet, so check by hand.
        let mut w = Writer::new();
        w.write_double(std::f64::consts::PI);
        let bytes = w.into_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(
            f64::from_le_bytes(bytes[..8].try_into().unwrap()),
            std::f64::consts::PI
        );
    }

    #[test]
    fn small_struct_roundtrip() {
        // {1: i32 = 42, 3: binary = "hi"}
        let mut w = Writer::new();
        w.write_field_header(1, FieldType::I32, 0);
        w.write_zigzag_i32(42);
        w.write_field_header(3, FieldType::Binary, 1);
        w.write_binary(b"hi");
        w.write_field_stop();
        let bytes = w.into_bytes();

        let mut cur = Cursor::new(&bytes);
        let h1 = read_field_header(&mut cur, 0).unwrap().unwrap();
        assert_eq!((h1.id, h1.field_type), (1, FieldType::I32));
        assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 42);

        let h2 = read_field_header(&mut cur, h1.id).unwrap().unwrap();
        assert_eq!((h2.id, h2.field_type), (3, FieldType::Binary));
        assert_eq!(crate::compact::read_binary(&mut cur).unwrap(), b"hi");

        assert!(read_field_header(&mut cur, h2.id).unwrap().is_none());
    }
}
