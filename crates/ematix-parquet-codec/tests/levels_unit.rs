//! Unit tests for the def/rep level decoder.

use ematix_parquet_codec::levels::{bit_width_for, decode_levels, parse_v1_data_page_body};

#[test]
fn bit_width_for_known_values() {
    assert_eq!(bit_width_for(0), 0);
    assert_eq!(bit_width_for(1), 1);
    assert_eq!(bit_width_for(2), 2);
    assert_eq!(bit_width_for(3), 2);
    assert_eq!(bit_width_for(4), 3);
    assert_eq!(bit_width_for(7), 3);
    assert_eq!(bit_width_for(8), 4);
    assert_eq!(bit_width_for(15), 4);
    assert_eq!(bit_width_for(16), 5);
}

#[test]
fn decode_levels_bit_width_zero_yields_zeros_and_consumes_nothing() {
    let body: &[u8] = &[];
    let (levels, consumed) = decode_levels(body, 0, 5).unwrap();
    assert_eq!(levels, vec![0u16; 5]);
    assert_eq!(consumed, 0);
}

#[test]
fn decode_levels_one_bit_all_ones_via_rle() {
    // 10 def levels, all 1 (nullable scalar column with no nulls).
    // Inner RLE stream (bit_width=1):
    //   RLE run of 10 ones: header varint(10 << 1) = 20 = 0x14
    //   value byte:        0x01 (ceil(1/8) = 1 byte)
    // Length prefix (LE u32): 2 = [0x02, 0x00, 0x00, 0x00]
    let body = [0x02u8, 0x00, 0x00, 0x00, 0x14, 0x01];
    let (levels, consumed) = decode_levels(&body, 1, 10).unwrap();
    assert_eq!(levels, vec![1u16; 10]);
    assert_eq!(consumed, 4 + 2);
}

#[test]
fn decode_levels_one_bit_alternating_via_bit_packed() {
    // 8 def levels [1,0,1,0,1,0,1,0].
    // Inner stream: bit-packed header varint((8/8)<<1|1)=3=0x03
    //   one byte: 0b01010101 = 0x55
    // Total inner: 2 bytes [0x03, 0x55]
    // Length prefix = 2.
    let body = [0x02u8, 0x00, 0x00, 0x00, 0x03, 0x55];
    let (levels, consumed) = decode_levels(&body, 1, 8).unwrap();
    assert_eq!(levels, vec![1, 0, 1, 0, 1, 0, 1, 0]);
    assert_eq!(consumed, 4 + 2);
}

#[test]
fn decode_levels_truncated_prefix_errors() {
    let body = [0x01u8, 0x00]; // only 2 bytes — can't read 4-byte prefix
    assert!(decode_levels(&body, 1, 1).is_err());
}

#[test]
fn decode_levels_truncated_body_errors() {
    // Prefix says 100 bytes but only 4 follow.
    let body = [0x64u8, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
    assert!(decode_levels(&body, 1, 10).is_err());
}

// ---- parse_v1_data_page_body ----------------------------------------------

#[test]
fn parse_v1_required_non_nested_skips_both_streams() {
    // max_def_level=0, max_rep_level=0 → no rep, no def → body IS values.
    let body = b"\xAA\xBB\xCC\xDD";
    let (rep, def, values) = parse_v1_data_page_body(body, 0, 0, 4).unwrap();
    assert_eq!(rep, vec![0u16; 4]);
    assert_eq!(def, vec![0u16; 4]);
    assert_eq!(values, body);
}

#[test]
fn parse_v1_nullable_scalar_decodes_def_only() {
    // max_def_level=1, max_rep_level=0. Body layout:
    //   [no rep section]
    //   [4-byte LE def length=2] [0x14, 0x01]   ← 10 ones
    //   [values bytes]
    let mut body: Vec<u8> = vec![0x02, 0x00, 0x00, 0x00, 0x14, 0x01];
    body.extend_from_slice(b"value-bytes-here");

    let (rep, def, values) = parse_v1_data_page_body(&body, 0, 1, 10).unwrap();
    assert_eq!(rep, vec![0u16; 10]);
    assert_eq!(def, vec![1u16; 10]);
    assert_eq!(values, b"value-bytes-here");
}

#[test]
fn parse_v1_nested_decodes_both_streams() {
    // Synthetic: max_rep_level=1, max_def_level=1, 4 values.
    // rep stream: RLE run of 4 zeros (no repetition).
    //   header varint(4<<1) = 8 = 0x08
    //   value byte 0x00 (bit_width=1, ceil(1/8)=1 byte)
    //   inner = [0x08, 0x00], prefix len = 2.
    // def stream: RLE run of 4 ones.
    //   header varint(4<<1) = 0x08, value byte 0x01
    //   inner = [0x08, 0x01], prefix len = 2.
    let mut body: Vec<u8> = vec![];
    body.extend_from_slice(&[0x02, 0x00, 0x00, 0x00, 0x08, 0x00]); // rep
    body.extend_from_slice(&[0x02, 0x00, 0x00, 0x00, 0x08, 0x01]); // def
    body.extend_from_slice(b"VVVV"); // values

    let (rep, def, values) = parse_v1_data_page_body(&body, 1, 1, 4).unwrap();
    assert_eq!(rep, vec![0u16; 4]);
    assert_eq!(def, vec![1u16; 4]);
    assert_eq!(values, b"VVVV");
}
