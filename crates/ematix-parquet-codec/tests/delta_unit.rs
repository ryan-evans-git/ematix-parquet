//! Unit + oracle tests for DELTA_BINARY_PACKED INT32 decoding.

use ematix_parquet_codec::delta::{decode_delta_i32, decode_delta_i64};

/// Hand-built bytes for the sequence [5, 8, 10, 12, 14].
///
/// Layout:
///   block_size = 128:                uvarint = [0x80, 0x01]
///   mini_blocks_per_block = 4:       uvarint = [0x04]
///   num_values = 5:                  uvarint = [0x05]
///   first_value = 5:                 zigzag  = 10 = [0x0A]
///   block 0:
///     min_delta = 2:                 zigzag  = 4  = [0x04]
///     bit_widths = [1, 0, 0, 0]:     bytes   = [0x01, 0x00, 0x00, 0x00]
///     mini-block 0 (bit_width=1, 32 values):
///       After subtracting min_delta, deltas are [1, 0, 0, 0, 0...].
///       LSB-first packing → byte 0 = 0b00000001, rest = 0.
///       Bytes: [0x01, 0x00, 0x00, 0x00]
///     mini-blocks 1..3: bit_width = 0 → no bytes emitted.
#[test]
fn hand_built_minimal_block() {
    let bytes: &[u8] = &[
        0x80, 0x01, // block_size = 128
        0x04, // mini_blocks_per_block = 4
        0x05, // num_values = 5
        0x0A, // first_value = zigzag(5)
        0x04, // min_delta = zigzag(2)
        0x01, 0x00, 0x00, 0x00, // bit_widths
        0x01, 0x00, 0x00, 0x00, // mini-block 0 packed deltas
    ];
    let decoded = decode_delta_i32(bytes).unwrap();
    assert_eq!(decoded, vec![5i32, 8, 10, 12, 14]);
}

#[test]
fn hand_built_zero_values() {
    // num_values = 0 → no first_value bytes needed... actually the
    // header still encodes them. Let's check what happens for
    // num_values = 1 (just first_value, no blocks).
    let bytes: &[u8] = &[
        0x80, 0x01, // block_size
        0x04, // mini_blocks_per_block
        0x01, // num_values = 1
        0x0A, // first_value = 5
              // No blocks needed.
    ];
    let decoded = decode_delta_i32(bytes).unwrap();
    assert_eq!(decoded, vec![5i32]);
}

// ---- Roundtrip oracle via parquet-rs's encoder -----------------------------

use parquet::data_type::{Int32Type, Int64Type};
use parquet::encodings::encoding::{DeltaBitPackEncoder, Encoder};

fn pr_encode_delta(values: &[i32]) -> Vec<u8> {
    let mut enc = DeltaBitPackEncoder::<Int32Type>::new();
    enc.put(values).unwrap();
    enc.flush_buffer().unwrap().to_vec()
}

fn pr_encode_delta_i64(values: &[i64]) -> Vec<u8> {
    let mut enc = DeltaBitPackEncoder::<Int64Type>::new();
    enc.put(values).unwrap();
    enc.flush_buffer().unwrap().to_vec()
}

fn roundtrip_check(values: &[i32]) {
    let bytes = pr_encode_delta(values);
    let decoded = decode_delta_i32(&bytes).unwrap();
    assert_eq!(decoded, values, "roundtrip mismatch for {values:?}");
}

fn roundtrip_check_i64(values: &[i64]) {
    let bytes = pr_encode_delta_i64(values);
    let decoded = decode_delta_i64(&bytes).unwrap();
    assert_eq!(decoded, values, "i64 roundtrip mismatch for {values:?}");
}

#[test]
fn roundtrip_single_value() {
    roundtrip_check(&[42]);
}

#[test]
fn roundtrip_monotonic_small() {
    roundtrip_check(&[1, 2, 3, 4, 5]);
}

#[test]
fn roundtrip_monotonic_long() {
    let v: Vec<i32> = (0..1000).collect();
    roundtrip_check(&v);
}

#[test]
fn roundtrip_decreasing_values() {
    let v: Vec<i32> = (0..500).rev().collect();
    roundtrip_check(&v);
}

#[test]
fn roundtrip_mixed_positive_negative() {
    roundtrip_check(&[-100, -50, 0, 50, 100, -200, 1000, 999, 1001]);
}

#[test]
fn roundtrip_extreme_values() {
    roundtrip_check(&[i32::MIN, 0, i32::MAX]);
}

#[test]
fn roundtrip_constant_values() {
    // All same → all deltas = 0.
    roundtrip_check(&[7i32; 256]);
}

#[test]
fn roundtrip_many_random_values() {
    // Pseudo-random sequence; no RNG dep, just an LCG.
    let mut seed: u32 = 0x12345678;
    let v: Vec<i32> = (0..2048)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed as i32
        })
        .collect();
    roundtrip_check(&v);
}

// ---- i64 roundtrips --------------------------------------------------------

#[test]
fn i64_roundtrip_single_value() {
    roundtrip_check_i64(&[42_000_000_000i64]);
}

#[test]
fn i64_roundtrip_monotonic() {
    let v: Vec<i64> = (0..1000i64).collect();
    roundtrip_check_i64(&v);
}

#[test]
fn i64_roundtrip_clustered_large_values() {
    // Values that need full i64 magnitude but small deltas — typical
    // shape for sorted timestamp or sequence columns.
    let base = 1_700_000_000_000_000_000i64; // ~ unix nanos for 2023
    let v: Vec<i64> = (0..500).map(|i| base + (i as i64) * 1_000_000).collect();
    roundtrip_check_i64(&v);
}

#[test]
fn i64_roundtrip_mixed_signs() {
    roundtrip_check_i64(&[-1_000_000i64, -100, 0, 100, 1_000_000, -5_000_000, 7]);
}

#[test]
fn i64_roundtrip_constant() {
    roundtrip_check_i64(&[12345i64; 256]);
}
