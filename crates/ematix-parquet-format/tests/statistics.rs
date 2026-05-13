//! TDD pin for the first real Parquet metadata struct: `Statistics`.
//!
//! From parquet.thrift:
//!   struct Statistics {
//!     1: optional binary max;
//!     2: optional binary min;
//!     3: optional i64    null_count;
//!     4: optional i64    distinct_count;
//!     5: optional binary max_value;
//!     6: optional binary min_value;
//!   }
//!
//! Tests build wire bytes via a small `CompactBuilder` helper. The
//! helper itself is pinned by `builder_matches_hand_derived_bytes`
//! against a fully spelled-out byte sequence so we trust it for the
//! rest of the tests.

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_statistics, Statistics};

/// Mirror of the thrift-compact writer, just enough for tests.
struct CompactBuilder {
    buf: Vec<u8>,
    prev_id: i16,
}

impl CompactBuilder {
    fn new() -> Self {
        Self { buf: Vec::new(), prev_id: 0 }
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

    fn binary(&mut self, id: i16, value: &[u8]) -> &mut Self {
        self.header(id, 8); // type=Binary
        self.write_uvarint(value.len() as u64);
        self.buf.extend_from_slice(value);
        self
    }

    fn i64_field(&mut self, id: i16, value: i64) -> &mut Self {
        self.header(id, 6); // type=I64
        let u = ((value << 1) ^ (value >> 63)) as u64;
        self.write_uvarint(u);
        self
    }

    fn stop(&mut self) -> Vec<u8> {
        self.buf.push(0x00);
        std::mem::take(&mut self.buf)
    }
}

#[test]
fn builder_matches_hand_derived_bytes() {
    // Statistics with just null_count = 5.
    //   field id 3, type I64 (nibble 6), delta = 3
    //     header byte: (3<<4)|6 = 0x36
    //     body:        zigzag(5) = 10 = 0x0A
    //   stop:          0x00
    let mut b = CompactBuilder::new();
    b.i64_field(3, 5);
    let bytes = b.stop();
    assert_eq!(bytes, vec![0x36, 0x0A, 0x00]);
}

#[test]
fn all_fields_present() {
    let mut b = CompactBuilder::new();
    let bytes = b
        .binary(1, &[0xDE, 0xAD])
        .binary(2, &[0xBE, 0xEF])
        .i64_field(3, 5)
        .i64_field(4, 100)
        .binary(5, &[0xCA, 0xFE])
        .binary(6, &[0xBA, 0xBE])
        .stop();

    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats.max, Some(&[0xDEu8, 0xAD][..]));
    assert_eq!(stats.min, Some(&[0xBEu8, 0xEF][..]));
    assert_eq!(stats.null_count, Some(5));
    assert_eq!(stats.distinct_count, Some(100));
    assert_eq!(stats.max_value, Some(&[0xCAu8, 0xFE][..]));
    assert_eq!(stats.min_value, Some(&[0xBAu8, 0xBE][..]));
    assert_eq!(cur.position(), bytes.len());
}

#[test]
fn empty_struct_yields_all_none() {
    let bytes = [0x00u8];
    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats, Statistics::default());
}

#[test]
fn only_null_count_present_other_fields_default_to_none() {
    // Field 3 only — first three deltas (1, 2) skipped.
    let mut b = CompactBuilder::new();
    let bytes = b.i64_field(3, 42).stop();

    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats.max, None);
    assert_eq!(stats.min, None);
    assert_eq!(stats.null_count, Some(42));
    assert_eq!(stats.distinct_count, None);
    assert_eq!(stats.max_value, None);
    assert_eq!(stats.min_value, None);
}

#[test]
fn binary_empty_slice_is_distinct_from_absent() {
    // max is present but zero-length.
    let mut b = CompactBuilder::new();
    let bytes = b.binary(1, &[]).stop();

    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats.max, Some(&[][..]));
    assert_eq!(stats.min, None);
}

#[test]
fn min_max_value_v2_fields_only() {
    // Newer writers populate min_value/max_value (5/6) but not the
    // deprecated min/max (1/2).
    let mut b = CompactBuilder::new();
    let bytes = b
        .binary(5, &[0x01, 0x02])
        .binary(6, &[0x03, 0x04])
        .stop();

    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats.max, None);
    assert_eq!(stats.min, None);
    assert_eq!(stats.max_value, Some(&[0x01u8, 0x02][..]));
    assert_eq!(stats.min_value, Some(&[0x03u8, 0x04][..]));
}
