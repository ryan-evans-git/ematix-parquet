//! Unit-level coverage for PLAIN decoders — no real parquet data.
//!
//! Pins the byte-order contract (little-endian) and the error paths
//! that the lineitem oracle test doesn't exercise.

use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::plain::{
    decode_plain_bool, decode_plain_byte_array, decode_plain_byte_array_n, decode_plain_f32,
    decode_plain_f64, decode_plain_f64_n, decode_plain_i32, decode_plain_i32_n, decode_plain_i64,
    decode_plain_i64_n,
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

// ---- ByteArray ------------------------------------------------------------

fn ba(len: u32, data: &[u8]) -> Vec<u8> {
    let mut v = len.to_le_bytes().to_vec();
    v.extend_from_slice(data);
    v
}

#[test]
fn decode_plain_byte_array_empty_input() {
    let bytes: &[u8] = &[];
    let out = decode_plain_byte_array(bytes).unwrap();
    assert!(out.is_empty());
}

#[test]
fn decode_plain_byte_array_single_empty_value() {
    // u32-LE length 0, no body bytes.
    let bytes = ba(0, &[]);
    let out = decode_plain_byte_array(&bytes).unwrap();
    assert_eq!(out, vec![&[][..]]);
}

#[test]
fn decode_plain_byte_array_multiple_values() {
    let mut bytes = ba(3, b"foo");
    bytes.extend(ba(1, b"!"));
    bytes.extend(ba(5, b"hello"));
    let out = decode_plain_byte_array(&bytes).unwrap();
    assert_eq!(out, vec![&b"foo"[..], &b"!"[..], &b"hello"[..]]);
}

#[test]
fn decode_plain_byte_array_n_stops_early() {
    let mut bytes = ba(3, b"foo");
    bytes.extend(ba(1, b"!"));
    bytes.extend(ba(5, b"trash"));
    let out = decode_plain_byte_array_n(&bytes, 2).unwrap();
    assert_eq!(out, vec![&b"foo"[..], &b"!"[..]]);
}

#[test]
fn decode_plain_byte_array_truncated_value_errors() {
    // Says len=10 but only 3 bytes of payload available.
    let bytes = ba(10, b"foo");
    let err = decode_plain_byte_array(&bytes).unwrap_err();
    // The error path here goes through Cursor::take → FormatError →
    // CodecError::Wire. Just check that it's an error of *some* kind.
    let _ = err;
}

#[test]
fn decode_plain_byte_array_truncated_length_prefix_errors() {
    // Only 2 bytes — can't even read the 4-byte length prefix.
    let bytes = [0u8, 0u8];
    let err = decode_plain_byte_array(&bytes).unwrap_err();
    let _ = err;
}

// ---- Boolean --------------------------------------------------------------

#[test]
fn decode_plain_bool_empty_request() {
    let bytes: &[u8] = &[];
    assert_eq!(decode_plain_bool(bytes, 0).unwrap(), Vec::<bool>::new());
}

#[test]
fn decode_plain_bool_eight_alternating_lsb_first() {
    // values [T,F,T,F,T,F,T,F]
    // bit packing LSB-first: bit0=val0=1, bit1=val1=0, ... bit7=val7=0
    // byte = 0b01010101 = 0x55
    let bytes = [0x55u8];
    assert_eq!(
        decode_plain_bool(&bytes, 8).unwrap(),
        vec![true, false, true, false, true, false, true, false]
    );
}

#[test]
fn decode_plain_bool_ten_values_across_two_bytes() {
    // [T,F,T,F,T,F,T,F | T,T]
    // byte 0: 0x55 (as above)
    // byte 1: bit0=val8=1, bit1=val9=1, rest padding = 0b00000011 = 0x03
    let bytes = [0x55u8, 0x03];
    assert_eq!(
        decode_plain_bool(&bytes, 10).unwrap(),
        vec![true, false, true, false, true, false, true, false, true, true]
    );
}

#[test]
fn decode_plain_bool_partial_byte_consumed_correctly() {
    // 5 values [T,T,F,T,F] → byte 0b00001011 = 0x0B (low 5 bits)
    let bytes = [0x0Bu8];
    assert_eq!(
        decode_plain_bool(&bytes, 5).unwrap(),
        vec![true, true, false, true, false]
    );
}

#[test]
fn decode_plain_bool_truncated_buffer_errors() {
    // Asking for 9 values but only 1 byte (enough for 8) supplied.
    let bytes = [0xFFu8];
    let err = decode_plain_bool(&bytes, 9).unwrap_err();
    let _ = err;
}

#[test]
fn decode_plain_bool_padding_bits_ignored() {
    // Buffer has trailing 1s in the padding region; result must be
    // exactly num_values long and not include the padding.
    let bytes = [0xFFu8]; // all 8 bits = 1
    let out = decode_plain_bool(&bytes, 3).unwrap();
    assert_eq!(out, vec![true, true, true]);
}
