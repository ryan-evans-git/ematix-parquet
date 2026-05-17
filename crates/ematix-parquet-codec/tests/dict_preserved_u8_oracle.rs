//! u8 dict-indices reader oracle.
//!
//! Round-trip the `read_column_byte_array_dict_preserved_u8`
//! reader against the codec's own byte_array dict writer, plus
//! confirm the failure modes (dict > 256, PLAIN-only chunk).
//!
//! Also exercises the lower-level `decode_rle_dictionary_indices_u8`
//! against its u32 sibling to prove byte-equivalence + correct
//! `BitWidthOutOfRange` propagation.

use ematix_parquet_codec::dict::{
    decode_rle_dictionary_indices, decode_rle_dictionary_indices_u8,
    decode_rle_dictionary_indices_u8_into,
};
use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::read::{
    read_column_byte_array, read_column_byte_array_dict_preserved,
    read_column_byte_array_dict_preserved_u8,
};
use ematix_parquet_codec::rle::encode_rle_bit_packed_single_run;
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_table_to_path, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use tempfile::NamedTempFile;

fn build_body(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut body = vec![bit_width];
    body.extend(encode_rle_bit_packed_single_run(indices, bit_width));
    body
}

#[test]
fn decode_u8_matches_u32_for_bw_le_8() {
    // bw=8, dict_size=256, 1024 rows round-robin.
    let indices: Vec<u32> = (0..1024).map(|i| i % 256).collect();
    let body = build_body(&indices, 8);
    let wide = decode_rle_dictionary_indices(&body, 1024).unwrap();
    let narrow = decode_rle_dictionary_indices_u8(&body, 1024).unwrap();
    assert_eq!(narrow.len(), wide.len());
    for (n, w) in narrow.iter().zip(wide.iter()) {
        assert_eq!(*n as u32, *w, "narrow != wide at u32-cast");
    }
}

#[test]
fn decode_u8_works_at_every_supported_width() {
    for bw in 1u8..=8 {
        let dict_size: u32 = 1 << bw; // ≤ 256
        let rows = 256;
        let indices: Vec<u32> = (0..rows).map(|i| (i as u32) % dict_size).collect();
        let body = build_body(&indices, bw);
        let got = decode_rle_dictionary_indices_u8(&body, rows).unwrap();
        assert_eq!(got.len(), rows, "row count mismatch at bw={bw}");
        for (row, &g) in got.iter().enumerate() {
            assert_eq!(
                g as u32, indices[row],
                "index mismatch at row {row} bw={bw}"
            );
        }
    }
}

#[test]
fn decode_u8_rejects_bw_gt_8() {
    // bw=9 → BitWidthOutOfRange (the u8 path's invariant).
    let indices: Vec<u32> = vec![100; 64];
    let body = build_body(&indices, 9);
    let r = decode_rle_dictionary_indices_u8(&body, 64);
    assert!(matches!(r, Err(CodecError::BitWidthOutOfRange(9))));
}

#[test]
fn decode_u8_into_appends() {
    let mut out: Vec<u8> = Vec::new();
    let indices1: Vec<u32> = (0..128).map(|i| i % 16).collect();
    let body1 = build_body(&indices1, 4);
    decode_rle_dictionary_indices_u8_into(&body1, 128, &mut out).unwrap();
    assert_eq!(out.len(), 128);

    // Second page appends.
    let indices2: Vec<u32> = (0..64).map(|i| i % 16).collect();
    let body2 = build_body(&indices2, 4);
    decode_rle_dictionary_indices_u8_into(&body2, 64, &mut out).unwrap();
    assert_eq!(out.len(), 128 + 64);
}

#[test]
fn read_column_dict_preserved_u8_round_trip() {
    // Low-cardinality byte_array column: 4 distinct values × 1024 rows.
    let dict_values: [&[u8]; 4] = [b"A", b"BB", b"CCC", b"DDDD"];
    let rows: Vec<&[u8]> = (0..1024).map(|i| dict_values[i % 4]).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_byte_array_column_dict_to_path(tmp.path(), "tag", &rows, CompressionCodec::Snappy)
        .unwrap();

    let file = ParquetFile::open(tmp.path()).unwrap();
    let col = read_column_byte_array_dict_preserved_u8(&file, 0, 0).unwrap();

    // Dict: 4 entries.
    assert_eq!(col.dict_offsets.len(), 5);
    // Reconstruct dict.
    let mut dict_strs: Vec<&[u8]> = Vec::new();
    for i in 0..4 {
        let lo = col.dict_offsets[i] as usize;
        let hi = col.dict_offsets[i + 1] as usize;
        dict_strs.push(&col.dict_bytes[lo..hi]);
    }
    // Indices reconstruct the row data through the dict.
    assert_eq!(col.indices.len(), 1024);
    for (row, &idx) in col.indices.iter().enumerate() {
        assert_eq!(dict_strs[idx as usize], rows[row], "row {row}");
    }
}

#[test]
fn read_column_dict_preserved_u8_matches_u32_variant() {
    let dict_values: [&[u8]; 8] = [
        b"a",
        b"bb",
        b"ccc",
        b"dddd",
        b"eeeee",
        b"ffffff",
        b"ggggggg",
        b"hhhhhhhh",
    ];
    let rows: Vec<&[u8]> = (0..512).map(|i| dict_values[i % 8]).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_byte_array_column_dict_to_path(tmp.path(), "v", &rows, CompressionCodec::Snappy).unwrap();
    let file = ParquetFile::open(tmp.path()).unwrap();

    let wide = read_column_byte_array_dict_preserved(&file, 0, 0).unwrap();
    let narrow = read_column_byte_array_dict_preserved_u8(&file, 0, 0).unwrap();

    assert_eq!(wide.dict_bytes, narrow.dict_bytes);
    assert_eq!(wide.dict_offsets, narrow.dict_offsets);
    assert_eq!(wide.indices.len(), narrow.indices.len());
    for (w, n) in wide.indices.iter().zip(narrow.indices.iter()) {
        assert_eq!(*n as u32, *w, "u32 vs u8 index disagreement");
    }
}

#[test]
fn read_column_dict_preserved_u8_rejects_dict_gt_256() {
    // 300 distinct values → dict_size > 256, must fail.
    let dict_values: Vec<Vec<u8>> = (0..300).map(|i| format!("v{i}").into_bytes()).collect();
    let rows: Vec<&[u8]> = (0..600).map(|i| dict_values[i % 300].as_slice()).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_byte_array_column_dict_to_path(tmp.path(), "v", &rows, CompressionCodec::Snappy).unwrap();
    let file = ParquetFile::open(tmp.path()).unwrap();

    let r = read_column_byte_array_dict_preserved_u8(&file, 0, 0);
    // Either the dict-too-big error (300 entries) OR the
    // BitWidthOutOfRange from the data-page bw > 8 — either is a
    // correct rejection. Sanity-check it's not silently producing
    // wrong indices.
    assert!(
        r.is_err(),
        "dict_size=300 must be rejected by the u8 reader"
    );
    // The u32 variant should still work fine.
    let wide = read_column_byte_array_dict_preserved(&file, 0, 0).unwrap();
    assert_eq!(wide.indices.len(), 600);
}

#[test]
fn read_column_dict_preserved_u8_rejects_plain_only_chunk() {
    // PLAIN-only byte_array column (no dict) — same failure mode as
    // the u32 variant.
    let rows: Vec<&[u8]> = (0..16).map(|_| b"x".as_slice()).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_table_to_path(
        tmp.path(),
        &[("v", ColumnData::ByteArray(&rows))],
        CompressionCodec::Uncompressed,
    )
    .unwrap();
    let file = ParquetFile::open(tmp.path()).unwrap();

    // Sanity: PLAIN reader still works.
    let plain = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(plain.len(), 16);

    // Dict-preserved-u8 must reject the no-dict chunk.
    let r = read_column_byte_array_dict_preserved_u8(&file, 0, 0);
    assert!(r.is_err());
}
