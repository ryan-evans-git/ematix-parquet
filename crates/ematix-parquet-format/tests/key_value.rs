//! TDD pin for `KeyValue`.
//!
//!   struct KeyValue {
//!     1: required string key;
//!     2: optional string value;
//!   }
//!
//! Thrift `string` is wire-identical to `binary` (uvarint len + bytes).

#[path = "common/mod.rs"]
mod common;

use common::CompactBuilder;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::error::FormatError;
use ematix_parquet_format::metadata::{read_key_value, KeyValue};

#[test]
fn key_and_value_both_present() {
    let mut b = CompactBuilder::new();
    let bytes = b.binary(1, b"foo").binary(2, b"bar").stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_key_value(&mut cur).unwrap(),
        KeyValue {
            key: b"foo",
            value: Some(b"bar"),
        }
    );
}

#[test]
fn key_only_value_absent() {
    let mut b = CompactBuilder::new();
    let bytes = b.binary(1, b"created_by").stop();
    let mut cur = Cursor::new(&bytes);
    assert_eq!(
        read_key_value(&mut cur).unwrap(),
        KeyValue {
            key: b"created_by",
            value: None,
        }
    );
}

#[test]
fn missing_required_key_errors() {
    // Empty struct → key field absent.
    let bytes = [0x00u8];
    let mut cur = Cursor::new(&bytes);
    match read_key_value(&mut cur) {
        Err(FormatError::MissingRequiredField {
            struct_name: "KeyValue",
            field_id: 1,
        }) => {}
        other => panic!("expected MissingRequiredField for id 1, got {other:?}"),
    }
}

#[test]
fn value_present_but_empty_is_distinct_from_absent() {
    let mut b = CompactBuilder::new();
    let bytes = b.binary(1, b"k").binary(2, &[]).stop();
    let mut cur = Cursor::new(&bytes);
    let kv = read_key_value(&mut cur).unwrap();
    assert_eq!(kv.key, b"k");
    assert_eq!(kv.value, Some(&[][..]));
}
