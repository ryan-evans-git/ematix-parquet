//! Equivalence test: `decode_rle_dictionary_into` (fused) must
//! produce identical output to `decode_rle_dictionary_indices +
//! lookup_dict` (two-pass) for every input.

use ematix_parquet_codec::dict::{
    decode_rle_dictionary_indices, decode_rle_dictionary_into, lookup_dict,
};

fn check(body: &[u8], dict: &[i32], num_values: usize) {
    // Two-pass reference.
    let indices = decode_rle_dictionary_indices(body, num_values).unwrap();
    let reference = lookup_dict(dict, &indices).unwrap();

    // Fused.
    let mut fused: Vec<i32> = Vec::new();
    decode_rle_dictionary_into(body, dict, num_values, &mut fused).unwrap();

    assert_eq!(fused, reference);
}

#[test]
fn rle_run_bit_width_3() {
    // body[0] = bit_width = 3
    // then: rle run of 5 copies of index 2 → header varint(10)=0x0A,
    //       value bytes: ceil(3/8)=1, value=2=0x02
    let body = [0x03u8, 0x0A, 0x02];
    let dict = [10i32, 20, 30, 40, 50, 60, 70, 80];
    check(&body, &dict, 5);
}

#[test]
fn bit_packed_run_bit_width_3() {
    // body[0] = 3, then bit-packed group of 8 = [0,1,2,3,4,5,6,7]
    // (same bytes as in tests/rle_unit.rs)
    let body = [0x03u8, 0x03, 0x88, 0xC6, 0xFA];
    let dict = [100i32, 200, 300, 400, 500, 600, 700, 800];
    check(&body, &dict, 8);
}

#[test]
fn mixed_rle_and_bit_packed() {
    // bit_width=3, [3-zero RLE run][bit-packed [0..7]] = 11 values total
    let body = [0x03u8, 0x06, 0x00, 0x03, 0x88, 0xC6, 0xFA];
    let dict = [
        1_000_000i32,
        2_000_000,
        3_000_000,
        4_000_000,
        5_000_000,
        6_000_000,
        7_000_000,
        8_000_000,
    ];
    check(&body, &dict, 11);
}

#[test]
fn truncated_request_stops_early() {
    // 8 values are bit-packed but we only want 5.
    let body = [0x03u8, 0x03, 0x88, 0xC6, 0xFA];
    let dict = [1i32, 2, 3, 4, 5, 6, 7, 8];
    check(&body, &dict, 5);
}

#[test]
fn bit_width_zero_yields_dict_zero() {
    // bit_width=0 → all indices are 0 → all values = dict[0]
    let body = [0x00u8];
    let dict = [42i32, 999];
    let mut out: Vec<i32> = Vec::new();
    decode_rle_dictionary_into(&body, &dict, 5, &mut out).unwrap();
    assert_eq!(out, vec![42i32; 5]);
}

#[test]
fn out_of_range_index_errors() {
    // bit_width=8, RLE of 1 with value=99, dict has 5 entries.
    let body = [0x08u8, 0x02, 0x63]; // 0x63=99
    let dict = [1i32, 2, 3, 4, 5];
    let mut out: Vec<i32> = Vec::new();
    let err = decode_rle_dictionary_into(&body, &dict, 1, &mut out).unwrap_err();
    use ematix_parquet_codec::error::CodecError;
    assert!(matches!(
        err,
        CodecError::DictIndexOutOfRange { index: 99, .. }
    ));
}

#[test]
fn appends_to_existing_vec() {
    // Verifies the function reserves additional capacity and appends
    // rather than clearing. Useful for cross-page accumulation.
    let body = [0x03u8, 0x06, 0x00]; // bit_width=3, RLE of 3 zeros
    let dict = [7i32, 8, 9];
    let mut out = vec![100i32, 200];
    decode_rle_dictionary_into(&body, &dict, 3, &mut out).unwrap();
    assert_eq!(out, vec![100i32, 200, 7, 7, 7]);
}
