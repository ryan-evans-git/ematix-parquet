//! TDD pin for `Statistics` (Parquet's per-page / per-chunk stats struct).
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

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_statistics, Statistics};

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
    let mut b = CompactBuilder::new();
    let bytes = b.binary(1, &[]).stop();

    let mut cur = Cursor::new(&bytes);
    let stats = read_statistics(&mut cur).unwrap();
    assert_eq!(stats.max, Some(&[][..]));
    assert_eq!(stats.min, None);
}

#[test]
fn min_max_value_v2_fields_only() {
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
