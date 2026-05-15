//! BYTE_STREAM_SPLIT encoding (Encoding ID 9).
//!
//! Spec: <https://github.com/apache/parquet-format/blob/master/Encodings.md>
//!
//! Layout for N values of a fixed-width primitive type with `W`
//! bytes per value (W=4 for FLOAT, W=8 for DOUBLE):
//!
//!   bytes[0 .. N]            — byte 0 of every value
//!   bytes[N .. 2N]           — byte 1 of every value
//!   ...
//!   bytes[(W-1)*N .. W*N]    — byte W-1 of every value
//!
//! Total size: `W * N` bytes — same as PLAIN. The win is purely in
//! downstream compression: each "stream" of identical-position bytes
//! is more compressible than the interleaved PLAIN form (e.g. all
//! float exponents land together and compress well).
//!
//! Rare in the wild today, but mandatory for v1.0 spec completeness.

use crate::error::{CodecError, Result};

const F32_WIDTH: usize = 4;
const F64_WIDTH: usize = 8;

/// Decode a BYTE_STREAM_SPLIT-encoded FLOAT (32-bit) page body.
///
/// `num_values` must be supplied — unlike PLAIN where the value count
/// is `bytes.len() / W`, the same byte count could in principle hold
/// any equivalent split layout. The reader knows N from the page
/// header.
pub fn decode_byte_stream_split_f32(bytes: &[u8], num_values: usize) -> Result<Vec<f32>> {
    let expected = num_values * F32_WIDTH;
    if bytes.len() != expected {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: F32_WIDTH,
            buffer_len: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(num_values);
    for i in 0..num_values {
        out.push(f32::from_le_bytes([
            bytes[i],
            bytes[num_values + i],
            bytes[2 * num_values + i],
            bytes[3 * num_values + i],
        ]));
    }
    Ok(out)
}

/// Decode a BYTE_STREAM_SPLIT-encoded DOUBLE (64-bit) page body.
pub fn decode_byte_stream_split_f64(bytes: &[u8], num_values: usize) -> Result<Vec<f64>> {
    let expected = num_values * F64_WIDTH;
    if bytes.len() != expected {
        return Err(CodecError::UnalignedPlainBuffer {
            value_width: F64_WIDTH,
            buffer_len: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(num_values);
    for i in 0..num_values {
        out.push(f64::from_le_bytes([
            bytes[i],
            bytes[num_values + i],
            bytes[2 * num_values + i],
            bytes[3 * num_values + i],
            bytes[4 * num_values + i],
            bytes[5 * num_values + i],
            bytes[6 * num_values + i],
            bytes[7 * num_values + i],
        ]));
    }
    Ok(out)
}

/// Encode a slice of f32 as BYTE_STREAM_SPLIT.
pub fn encode_byte_stream_split_f32(values: &[f32]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n * F32_WIDTH];
    for (i, &v) in values.iter().enumerate() {
        let b = v.to_le_bytes();
        out[i] = b[0];
        out[n + i] = b[1];
        out[2 * n + i] = b[2];
        out[3 * n + i] = b[3];
    }
    out
}

/// Encode a slice of f64 as BYTE_STREAM_SPLIT.
pub fn encode_byte_stream_split_f64(values: &[f64]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n * F64_WIDTH];
    for (i, &v) in values.iter().enumerate() {
        let b = v.to_le_bytes();
        for byte_pos in 0..F64_WIDTH {
            out[byte_pos * n + i] = b[byte_pos];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_f32(values: &[f32]) {
        let encoded = encode_byte_stream_split_f32(values);
        assert_eq!(encoded.len(), values.len() * 4);
        let decoded = decode_byte_stream_split_f32(&encoded, values.len()).unwrap();
        // Bit-exact comparison via to_bits — survives NaN, ±0.
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    fn roundtrip_f64(values: &[f64]) {
        let encoded = encode_byte_stream_split_f64(values);
        assert_eq!(encoded.len(), values.len() * 8);
        let decoded = decode_byte_stream_split_f64(&encoded, values.len()).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn f32_roundtrip_simple() {
        roundtrip_f32(&[1.0, -2.0, 3.14, 0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn f32_roundtrip_many() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5 - 100.0).collect();
        roundtrip_f32(&v);
    }

    #[test]
    fn f32_roundtrip_extremes() {
        roundtrip_f32(&[
            f32::MIN,
            f32::MAX,
            f32::EPSILON,
            -f32::EPSILON,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
        ]);
    }

    #[test]
    fn f64_roundtrip_simple() {
        roundtrip_f64(&[1.0, -2.0, 3.14159265358979, 0.0, -0.0, f64::INFINITY]);
    }

    #[test]
    fn f64_roundtrip_many() {
        let v: Vec<f64> = (0..500).map(|i| (i as f64) / 7.0 - 50.0).collect();
        roundtrip_f64(&v);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(encode_byte_stream_split_f32(&[]).is_empty());
        assert!(encode_byte_stream_split_f64(&[]).is_empty());
        assert!(decode_byte_stream_split_f32(&[], 0).unwrap().is_empty());
        assert!(decode_byte_stream_split_f64(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn wrong_byte_count_rejected() {
        // 7 bytes can't be a complete f32 stream of any N.
        assert!(decode_byte_stream_split_f32(&[0u8; 7], 2).is_err());
        // 9 bytes can't be a complete f64 stream of N=2 (would need 16).
        assert!(decode_byte_stream_split_f64(&[0u8; 9], 2).is_err());
    }
}
