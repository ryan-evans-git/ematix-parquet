//! Unit-level coverage for PLAIN decoders — no real parquet data.
//!
//! Pins the byte-order contract (little-endian) and the error paths
//! that the lineitem oracle test doesn't exercise.

use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::plain::{
    decode_plain_f32, decode_plain_f64, decode_plain_f64_n, decode_plain_i32, decode_plain_i32_n,
    decode_plain_i64, decode_plain_i64_n,
};

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

// ---- Int32 ----------------------------------------------------------------

#[test]
fn decode_plain_i32_boundary_values() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&(-1i32).to_le_bytes());
    bytes.extend_from_slice(&i32::MAX.to_le_bytes());
    bytes.extend_from_slice(&i32::MIN.to_le_bytes());
    assert_eq!(
        decode_plain_i32(&bytes).unwrap(),
        vec![0i32, -1, i32::MAX, i32::MIN]
    );
}

#[test]
fn decode_plain_i32_partial_buffer_errors() {
    let bytes = [0u8; 3];
    assert!(matches!(
        decode_plain_i32(&bytes),
        Err(CodecError::UnalignedPlainBuffer {
            value_width: 4,
            buffer_len: 3,
        })
    ));
}

#[test]
fn decode_plain_i32_n_truncates() {
    let mut bytes = Vec::new();
    for v in [10i32, 20, 30, 40, 50] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(decode_plain_i32_n(&bytes, 3).unwrap(), vec![10i32, 20, 30]);
}

// ---- Float32 --------------------------------------------------------------

#[test]
fn decode_plain_f32_known_values() {
    let mut bytes = Vec::new();
    for v in [0.0f32, 1.5, -2.25, f32::INFINITY, f32::NAN] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let got = decode_plain_f32(&bytes).unwrap();
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], 1.5);
    assert_eq!(got[2], -2.25);
    assert_eq!(got[3], f32::INFINITY);
    assert!(got[4].is_nan());
}

// ---- Float64 --------------------------------------------------------------

#[test]
fn decode_plain_f64_known_values() {
    let mut bytes = Vec::new();
    for v in [0.0f64, 0.10000000000000001, -1.234e-10, f64::INFINITY] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let got = decode_plain_f64(&bytes).unwrap();
    assert_eq!(got, vec![0.0, 0.10000000000000001, -1.234e-10, f64::INFINITY]);
}

#[test]
fn decode_plain_f64_n_truncates() {
    let mut bytes = Vec::new();
    for v in [1.0f64, 2.0, 3.0, 4.0, 5.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(
        decode_plain_f64_n(&bytes, 3).unwrap(),
        vec![1.0f64, 2.0, 3.0]
    );
}
