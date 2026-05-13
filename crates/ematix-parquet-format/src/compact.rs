//! Thrift compact-protocol primitive readers.
//!
//! Spec: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
//!
//! Apache Parquet's metadata (`FileMetaData`, page headers, indexes) is
//! serialized with this protocol. All readers here operate on a `&[u8]`
//! and advance a `Cursor`. Higher layers own the actual file I/O.

use crate::error::{FormatError, Result};

/// Borrowed cursor over a byte slice. Cheap to copy.
#[derive(Debug, Clone, Copy)]
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.buf.len() {
            return Err(FormatError::UnexpectedEof {
                needed: 1,
                remaining: 0,
            });
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Take a zero-copy slice of `n` bytes and advance past them.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(FormatError::UnexpectedEof {
                needed: n,
                remaining: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

/// Read an unsigned LEB128 varint. Caller chooses the result width.
///
/// Thrift compact stores integers as either plain varints (lengths,
/// sizes) or zigzag varints (signed scalars). Both share this base.
///
/// Bails after 10 continuation bytes — enough headroom for any i64
/// (ceil(64 / 7) = 10).
pub fn read_uvarint(cur: &mut Cursor<'_>) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for _ in 0..10 {
        let byte = cur.read_u8()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
    Err(FormatError::VarintOverflow)
}

/// Thrift signed-32 in compact protocol is zigzag-encoded:
/// encoded = (value << 1) ^ (value >> 31); decoded = (u >> 1) ^ -(u & 1).
pub fn read_zigzag_i32(cur: &mut Cursor<'_>) -> Result<i32> {
    let u = read_uvarint(cur)? as u32;
    Ok(((u >> 1) as i32) ^ -((u & 1) as i32))
}

/// Same idea, 64-bit width.
pub fn read_zigzag_i64(cur: &mut Cursor<'_>) -> Result<i64> {
    let u = read_uvarint(cur)?;
    Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
}

/// Thrift compact struct field type codes (low nibble of field header).
///
/// The set is fixed by the protocol spec. Code 0 is STOP (terminator),
/// represented separately as `None` from `read_field_header`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum FieldType {
    BoolTrue = 1,
    BoolFalse = 2,
    Byte = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    Double = 7,
    Binary = 8,
    List = 9,
    Set = 10,
    Map = 11,
    Struct = 12,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldHeader {
    pub id: i16,
    pub field_type: FieldType,
}

/// Read one field header from a struct stream.
///
/// `prev_id` is the id of the previously decoded field (0 for the
/// first call). Returns `None` when the STOP byte (0x00) is reached.
///
/// For embedded booleans (`BoolTrue`/`BoolFalse`) the value is in the
/// header itself — callers must not read a body.
pub fn read_field_header(cur: &mut Cursor<'_>, prev_id: i16) -> Result<Option<FieldHeader>> {
    let header = cur.read_u8()?;
    if header == 0x00 {
        return Ok(None);
    }
    let type_nibble = header & 0x0F;
    let delta = (header & 0xF0) >> 4;
    let field_type = decode_field_type(type_nibble)?;
    let id = if delta == 0 {
        // Long form: explicit zigzag i16 follows.
        let z = read_zigzag_i32(cur)?;
        z as i16
    } else {
        prev_id + delta as i16
    };
    Ok(Some(FieldHeader { id, field_type }))
}

/// Read a length-prefixed byte string from the cursor.
///
/// Compact protocol encodes Binary (and String) as `uvarint(len)` then
/// `len` raw bytes. Zero-copy: the returned slice borrows from the
/// cursor's buffer.
pub fn read_binary<'a>(cur: &mut Cursor<'a>) -> Result<&'a [u8]> {
    let len = read_uvarint(cur)? as usize;
    cur.take(len)
}

fn decode_field_type(nibble: u8) -> Result<FieldType> {
    Ok(match nibble {
        1 => FieldType::BoolTrue,
        2 => FieldType::BoolFalse,
        3 => FieldType::Byte,
        4 => FieldType::I16,
        5 => FieldType::I32,
        6 => FieldType::I64,
        7 => FieldType::Double,
        8 => FieldType::Binary,
        9 => FieldType::List,
        10 => FieldType::Set,
        11 => FieldType::Map,
        12 => FieldType::Struct,
        other => return Err(FormatError::InvalidFieldType(other)),
    })
}
