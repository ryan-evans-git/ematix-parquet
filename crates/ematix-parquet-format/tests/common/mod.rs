//! Shared test fixtures.
//!
//! `CompactBuilder` synthesizes thrift-compact wire bytes for use as
//! decoder inputs. The writer is the mirror image of the readers in
//! `ematix_parquet_format::compact` — tests should not depend on the
//! readers to verify their own inputs.
//!
//! Each test file that uses this should `#[path = "common/mod.rs"]
//! mod common;` at the top to pull it in. (Integration tests in
//! Rust are independent crates, so the standard `mod common;`
//! convention via `tests/common/mod.rs` is the way to share code.)

#![allow(dead_code)] // not every test file uses every helper

pub struct CompactBuilder {
    buf: Vec<u8>,
    prev_id: i16,
}

impl CompactBuilder {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            prev_id: 0,
        }
    }

    fn header(&mut self, id: i16, type_nibble: u8) {
        let delta = id - self.prev_id;
        if delta >= 1 && delta <= 15 {
            self.buf.push(((delta as u8) << 4) | type_nibble);
        } else {
            self.buf.push(type_nibble);
            // long-form id: zigzag i16
            let mut u = (((id as i32) << 1) ^ ((id as i32) >> 31)) as u32;
            loop {
                if u < 0x80 {
                    self.buf.push(u as u8);
                    break;
                }
                self.buf.push(((u & 0x7F) | 0x80) as u8);
                u >>= 7;
            }
        }
        self.prev_id = id;
    }

    fn write_uvarint(&mut self, mut v: u64) {
        loop {
            if v < 0x80 {
                self.buf.push(v as u8);
                return;
            }
            self.buf.push(((v & 0x7F) | 0x80) as u8);
            v >>= 7;
        }
    }

    pub fn binary(&mut self, id: i16, value: &[u8]) -> &mut Self {
        self.header(id, 8);
        self.write_uvarint(value.len() as u64);
        self.buf.extend_from_slice(value);
        self
    }

    pub fn i16_field(&mut self, id: i16, value: i16) -> &mut Self {
        self.header(id, 4);
        let v = value as i32;
        let u = ((v << 1) ^ (v >> 31)) as u32 as u64;
        self.write_uvarint(u);
        self
    }

    pub fn i32_field(&mut self, id: i16, value: i32) -> &mut Self {
        self.header(id, 5);
        let u = ((value << 1) ^ (value >> 31)) as u32 as u64;
        self.write_uvarint(u);
        self
    }

    pub fn i64_field(&mut self, id: i16, value: i64) -> &mut Self {
        self.header(id, 6);
        let u = ((value << 1) ^ (value >> 63)) as u64;
        self.write_uvarint(u);
        self
    }

    /// Embedded boolean: type nibble carries the value.
    pub fn bool_field(&mut self, id: i16, value: bool) -> &mut Self {
        let nibble = if value { 1u8 } else { 2u8 };
        self.header(id, nibble);
        self
    }

    /// I8 / BYTE field: type nibble 3, single raw byte body.
    pub fn i8_field(&mut self, id: i16, value: i8) -> &mut Self {
        self.header(id, 3);
        self.buf.push(value as u8);
        self
    }

    /// Emit a fully-formed empty nested struct (just a STOP byte).
    /// Convenient for building union variants whose payload is `{}`.
    pub fn empty_struct() -> Vec<u8> {
        vec![0x00]
    }

    /// Push a nested struct: field header (type=12) followed by the
    /// nested struct's bytes (which must already include the inner
    /// STOP byte).
    pub fn struct_field(&mut self, id: i16, nested_bytes: &[u8]) -> &mut Self {
        self.header(id, 12);
        self.buf.extend_from_slice(nested_bytes);
        self
    }

    /// Push an i32-valued enum field (encoding, page_type, etc.).
    /// Wire form is identical to i32_field; this alias just makes the
    /// intent legible.
    pub fn enum_field(&mut self, id: i16, value: i32) -> &mut Self {
        self.i32_field(id, value)
    }

    fn list_header(&mut self, count: usize, type_nibble: u8) {
        if count < 15 {
            self.buf.push(((count as u8) << 4) | type_nibble);
        } else {
            self.buf.push(0xF0 | type_nibble);
            self.write_uvarint(count as u64);
        }
    }

    /// Field that is a list<i32> (and structurally list<Encoding>,
    /// list<PageType>, etc. — same i32 wire form).
    pub fn list_i32_field(&mut self, id: i16, values: &[i32]) -> &mut Self {
        self.header(id, 9); // type=List
        self.list_header(values.len(), 5); // elem type=I32
        for &v in values {
            let u = ((v << 1) ^ (v >> 31)) as u32 as u64;
            self.write_uvarint(u);
        }
        self
    }

    /// Field that is a list<string|binary>.
    pub fn list_binary_field(&mut self, id: i16, values: &[&[u8]]) -> &mut Self {
        self.header(id, 9);
        self.list_header(values.len(), 8); // elem type=Binary
        for v in values {
            self.write_uvarint(v.len() as u64);
            self.buf.extend_from_slice(v);
        }
        self
    }

    /// Field that is a list<struct>. Each element's bytes must already
    /// include its own STOP terminator.
    pub fn list_struct_field(&mut self, id: i16, elements: &[Vec<u8>]) -> &mut Self {
        self.header(id, 9);
        self.list_header(elements.len(), 12); // elem type=Struct
        for e in elements {
            self.buf.extend_from_slice(e);
        }
        self
    }

    /// Field that is a list<bool>. Elem type nibble = 2 (canonical
    /// "bool" code for list elements per thrift compact spec); each
    /// element body is one byte: 0x01 for true, 0x02 for false.
    pub fn list_bool_field(&mut self, id: i16, values: &[bool]) -> &mut Self {
        self.header(id, 9);
        self.list_header(values.len(), 2);
        for &v in values {
            self.buf.push(if v { 0x01 } else { 0x02 });
        }
        self
    }

    /// Field that is a list<i64>.
    pub fn list_i64_field(&mut self, id: i16, values: &[i64]) -> &mut Self {
        self.header(id, 9);
        self.list_header(values.len(), 6);
        for &v in values {
            let u = ((v << 1) ^ (v >> 63)) as u64;
            self.write_uvarint(u);
        }
        self
    }

    pub fn stop(&mut self) -> Vec<u8> {
        self.buf.push(0x00);
        std::mem::take(&mut self.buf)
    }
}
