//! Unit tests for Parquet's RLE / bit-packed hybrid primitive.
//!
//! All byte vectors are hand-derived from the spec. The encoder isn't
//! implemented yet (we're decode-only for now), so these stand on
//! their own as a contract document.

use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::rle::decode_rle_bit_packed;

#[test]
fn empty_request() {
    let bytes = [];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 3, 0).unwrap(),
        Vec::<u64>::new()
    );
}

#[test]
fn bit_width_zero_yields_all_zeros() {
    // bit_width=0 → the byte stream is irrelevant; spec says every
    // value is 0.
    let bytes = [];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 0, 5).unwrap(),
        vec![0u64; 5]
    );
}

#[test]
fn rle_run_bit_width_8() {
    // RLE run of 5 copies of 42.
    //   header: uvarint(5 << 1) = uvarint(10) = 0x0A
    //   value:  42 = 0x2A, single byte LE
    let bytes = [0x0A, 0x2A];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 8, 5).unwrap(),
        vec![42u64; 5]
    );
}

#[test]
fn rle_run_bit_width_18_three_byte_value() {
    // RLE run of 3 copies of 100000.
    //   header: uvarint(3 << 1) = 0x06
    //   value:  100000 = 0x01_86_A0 → bytes [0xA0, 0x86, 0x01]
    let bytes = [0x06, 0xA0, 0x86, 0x01];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 18, 3).unwrap(),
        vec![100_000u64; 3]
    );
}

#[test]
fn bit_packed_run_bit_width_8_eight_values() {
    // 8 byte-wide values packed back-to-back.
    //   header: uvarint((8/8) << 1 | 1) = uvarint(3) = 0x03
    //   bytes:  values directly
    let bytes = [0x03, 10, 20, 30, 40, 50, 60, 70, 80];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 8, 8).unwrap(),
        vec![10u64, 20, 30, 40, 50, 60, 70, 80]
    );
}

#[test]
fn bit_packed_run_bit_width_1_alternating() {
    // 8 single-bit values [1, 0, 1, 0, 1, 0, 1, 0].
    // LSB-first packing: bit 0 = val0 = 1, bit 1 = val1 = 0, ...
    //   byte: 0b01010101 = 0x55
    let bytes = [0x03, 0x55];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 1, 8).unwrap(),
        vec![1u64, 0, 1, 0, 1, 0, 1, 0]
    );
}

#[test]
fn bit_packed_run_bit_width_3_eight_values_zero_through_seven() {
    // Values [0,1,2,3,4,5,6,7] each in 3 bits, LSB-first.
    // Derived in commit message; bytes = [0x88, 0xC6, 0xFA].
    let bytes = [0x03, 0x88, 0xC6, 0xFA];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 3, 8).unwrap(),
        vec![0u64, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn mixed_rle_then_bit_packed() {
    // 3-zero RLE run, then a bit-packed group of 8 [0..7].
    //   rle header: uvarint(3 << 1) = 0x06, value byte 0 → [0x06, 0x00]
    //   bit-packed header + values: [0x03, 0x88, 0xC6, 0xFA]
    let bytes = [0x06, 0x00, 0x03, 0x88, 0xC6, 0xFA];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 3, 11).unwrap(),
        vec![0u64, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn bit_packed_padding_dropped_when_num_values_not_multiple_of_8() {
    // Writer packed 8 values; reader only wants 5.
    //   bytes encode [0..7]; we request 5.
    let bytes = [0x03, 0x88, 0xC6, 0xFA];
    assert_eq!(
        decode_rle_bit_packed(&bytes, 3, 5).unwrap(),
        vec![0u64, 1, 2, 3, 4]
    );
}

#[test]
fn bit_width_out_of_range_errors() {
    let bytes = [0x00];
    let err = decode_rle_bit_packed(&bytes, 65, 1).unwrap_err();
    assert!(matches!(err, CodecError::BitWidthOutOfRange(65)));
}
