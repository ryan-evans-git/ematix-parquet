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
