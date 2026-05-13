//! Unit-level coverage for PLAIN decoders — no real parquet data.
//!
//! Pins the byte-order contract (little-endian) and the error paths
//! that the lineitem oracle test doesn't exercise.

use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::plain::{decode_plain_i64, decode_plain_i64_n};

#[test]
fn decode_plain_i64_empty_buffer() {
    let out = decode_plain_i64(&[]).unwrap();
    assert!(out.is_empty());
}

#[test]
fn decode_plain_i64_single_value_zero() {
    let bytes = [0u8; 8];
    assert_eq!(decode_plain_i64(&bytes).unwrap(), vec![0i64]);
}

#[test]
fn decode_plain_i64_three_values_known_le_bytes() {
    // 1, -1, i64::MAX laid out little-endian.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1i64.to_le_bytes());
    bytes.extend_from_slice(&(-1i64).to_le_bytes());
    bytes.extend_from_slice(&i64::MAX.to_le_bytes());
    assert_eq!(
        decode_plain_i64(&bytes).unwrap(),
        vec![1i64, -1, i64::MAX]
    );
}

#[test]
fn decode_plain_i64_partial_buffer_errors() {
    // 7 bytes — not a multiple of 8.
    let bytes = [0u8; 7];
    let err = decode_plain_i64(&bytes).unwrap_err();
    match err {
        CodecError::UnalignedPlainBuffer {
            value_width,
            buffer_len,
        } => {
            assert_eq!(value_width, 8);
            assert_eq!(buffer_len, 7);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn decode_plain_i64_n_truncates_to_requested_count() {
    let mut bytes = Vec::new();
    for v in [10i64, 20, 30, 40, 50] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(decode_plain_i64_n(&bytes, 3).unwrap(), vec![10i64, 20, 30]);
}

#[test]
fn decode_plain_i64_n_underflow_errors() {
    let bytes = vec![0u8; 8]; // only 1 value
    let err = decode_plain_i64_n(&bytes, 2).unwrap_err();
    assert!(matches!(
        err,
        CodecError::UnderflowingPlainBuffer {
            requested_values: 2,
            ..
        }
    ));
}
