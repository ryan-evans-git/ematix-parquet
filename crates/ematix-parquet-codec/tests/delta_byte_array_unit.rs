//! Unit + oracle tests for DELTA_LENGTH_BYTE_ARRAY and
//! DELTA_BYTE_ARRAY encodings.

use ematix_parquet_codec::delta::{decode_delta_byte_array, decode_delta_length_byte_array};

// ---- DELTA_LENGTH_BYTE_ARRAY (lengths via DELTA + concatenated bytes) ----

#[test]
fn delta_length_byte_array_basic() {
    // Three values: "foo", "barbar", "x"
    let values: Vec<&[u8]> = vec![b"foo", b"barbar", b"x"];
    let bytes = pr_encode_delta_length(&values);
    let decoded = decode_delta_length_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

#[test]
fn delta_length_byte_array_empty_strings_interleaved() {
    let values: Vec<&[u8]> = vec![b"a", b"", b"bb", b"", b"ccc"];
    let bytes = pr_encode_delta_length(&values);
    let decoded = decode_delta_length_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

#[test]
fn delta_length_byte_array_many_values() {
    // 500 distinct strings to drive a real DELTA-packed length stream.
    let owned: Vec<String> = (0..500).map(|i| format!("value_number_{:04}", i)).collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
    let bytes = pr_encode_delta_length(&values);
    let decoded = decode_delta_length_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

// ---- DELTA_BYTE_ARRAY (prefix_lengths + suffix_lengths + suffix bytes) ----

#[test]
fn delta_byte_array_sorted_with_common_prefixes() {
    let values: Vec<&[u8]> = vec![
        b"alpha",
        b"alphabet",
        b"alphabetical",
        b"alphabetically",
        b"beta",
        b"betacarotene",
    ];
    let bytes = pr_encode_delta_byte_array(&values);
    let decoded = decode_delta_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

#[test]
fn delta_byte_array_no_shared_prefix() {
    let values: Vec<&[u8]> = vec![b"foo", b"bar", b"quux", b"xyzzy"];
    let bytes = pr_encode_delta_byte_array(&values);
    let decoded = decode_delta_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

#[test]
fn delta_byte_array_single_value() {
    let values: Vec<&[u8]> = vec![b"only"];
    let bytes = pr_encode_delta_byte_array(&values);
    let decoded = decode_delta_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

#[test]
fn delta_byte_array_many_sorted_values() {
    // Highly redundant prefixes — the encoding's sweet spot.
    let owned: Vec<String> = (0..1000)
        .map(|i| format!("https://example.com/path/segment/{:06}", i))
        .collect();
    let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
    let bytes = pr_encode_delta_byte_array(&values);
    let decoded = decode_delta_byte_array(&bytes).unwrap();
    let decoded_refs: Vec<&[u8]> = decoded.iter().map(|v| v.as_slice()).collect();
    assert_eq!(decoded_refs, values);
}

// ---- helpers via parquet-rs encoders ---------------------------------------

use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::encodings::encoding::{DeltaByteArrayEncoder, DeltaLengthByteArrayEncoder, Encoder};

fn pr_encode_delta_length(values: &[&[u8]]) -> Vec<u8> {
    let bas: Vec<ByteArray> = values.iter().map(|v| ByteArray::from(v.to_vec())).collect();
    let mut enc = DeltaLengthByteArrayEncoder::<ByteArrayType>::new();
    enc.put(&bas).unwrap();
    enc.flush_buffer().unwrap().to_vec()
}

fn pr_encode_delta_byte_array(values: &[&[u8]]) -> Vec<u8> {
    let bas: Vec<ByteArray> = values.iter().map(|v| ByteArray::from(v.to_vec())).collect();
    let mut enc = DeltaByteArrayEncoder::<ByteArrayType>::new();
    enc.put(&bas).unwrap();
    enc.flush_buffer().unwrap().to_vec()
}
