//! Oracle tests for INT96 + FIXED_LEN_BYTE_ARRAY PLAIN decoders.
//!
//! parquet-rs writes a file with the target column, then we decode
//! the page body ourselves and assert byte-identical values.

use std::sync::Arc;

use ematix_parquet_codec::plain::{decode_plain_fixed_len_byte_array, decode_plain_int96};
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::data_type::{FixedLenByteArray, Int96};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn write_int96_via_parquet_rs(path: &std::path::Path, values: &[Int96]) {
    let schema = parse_message_type("message s { REQUIRED INT96 ts; }").unwrap();
    // Force PLAIN encoding so the data page exercises decode_plain_int96
    // directly. Without this parquet-rs builds a dictionary page (the
    // values are PLAIN inside the dict, but the data page becomes
    // RleDictionary which decode_plain_int96 doesn't handle).
    let props = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, Arc::new(schema), Arc::new(props)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<parquet::data_type::Int96Type>()
        .write_batch(values, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_flba_via_parquet_rs(
    path: &std::path::Path,
    values: &[FixedLenByteArray],
    type_length: i32,
) {
    let msg = format!(
        "message s {{ REQUIRED FIXED_LEN_BYTE_ARRAY({type_length}) v; }}"
    );
    let schema = parse_message_type(&msg).unwrap();
    let props = WriterProperties::builder().build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, Arc::new(schema), Arc::new(props)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<parquet::data_type::FixedLenByteArrayType>()
        .write_batch(values, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

/// Pull the raw decompressed body of the first data page of column 0.
/// Convenience helper that avoids the high-level façade (the façade
/// doesn't yet dispatch INT96 or FLBA — that's intentional, since the
/// generic interface needs a way to surface FLBA's type_length).
fn first_data_page_body(path: &std::path::Path) -> Vec<u8> {
    use ematix_parquet_codec::compression::{
        decompress_snappy_into, decompress_zstd_into,
    };
    use ematix_parquet_format::types::{CompressionCodec, PageType};

    let file = ParquetFile::open(path).unwrap();
    let md = file.metadata().unwrap();
    let cm = md.row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let chunk = file.read_range(start, cm.total_compressed_size as u64).unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp = Vec::new();
    loop {
        let (hdr, body) = walker.next_page().unwrap().expect("page");
        if hdr.page_type != PageType::DataPage {
            continue;
        }
        let dph = hdr.data_page_header.as_ref().unwrap();
        assert!(
            matches!(dph.encoding, Encoding::Plain),
            "test expects PLAIN-encoded INT96/FLBA"
        );
        match cm.codec {
            CompressionCodec::Uncompressed => return body.to_vec(),
            CompressionCodec::Snappy => {
                decompress_snappy_into(body, &mut decomp).unwrap();
                return decomp;
            }
            CompressionCodec::Zstd => {
                decompress_zstd_into(body, &mut decomp).unwrap();
                return decomp;
            }
            other => panic!("unexpected codec in test fixture: {other:?}"),
        }
    }
}

#[test]
fn decode_plain_int96_matches_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int96.parquet");

    let values: Vec<Int96> = (0..16)
        .map(|i| {
            // Construct arbitrary 96-bit values; the decode should
            // preserve byte ordering exactly.
            let mut v = Int96::new();
            v.set_data(i as u32, (i * 7) as u32 + 11, (i * 13) as u32 + 100);
            v
        })
        .collect();
    write_int96_via_parquet_rs(&path, &values);

    let body = first_data_page_body(&path);
    let decoded = decode_plain_int96(&body).unwrap();
    assert_eq!(decoded.len(), values.len());
    for (ours, theirs) in decoded.iter().zip(values.iter()) {
        // Compare via the public data() accessor — same three u32s.
        assert_eq!(&ours.data(), theirs.data());
    }
}

#[test]
fn decode_plain_flba_matches_parquet_rs_len_4() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flba4.parquet");

    let raw: Vec<[u8; 4]> = (0..32)
        .map(|i| [i, i + 1, i + 2, i + 3])
        .collect();
    let values: Vec<FixedLenByteArray> = raw
        .iter()
        .map(|a| FixedLenByteArray::from(a.to_vec()))
        .collect();
    write_flba_via_parquet_rs(&path, &values, 4);

    let body = first_data_page_body(&path);
    let decoded = decode_plain_fixed_len_byte_array(&body, 4).unwrap();
    assert_eq!(decoded.len(), raw.len());
    for (ours, theirs) in decoded.iter().zip(raw.iter()) {
        assert_eq!(*ours, theirs.as_slice());
    }
}

#[test]
fn decode_plain_flba_matches_parquet_rs_len_16() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flba16.parquet");

    // 16-byte FLBA — the shape used for UUIDs / DECIMAL(38,_).
    let raw: Vec<[u8; 16]> = (0..8u8)
        .map(|i| {
            let mut row = [0u8; 16];
            for (j, b) in row.iter_mut().enumerate() {
                *b = i.wrapping_mul(17).wrapping_add(j as u8);
            }
            row
        })
        .collect();
    let values: Vec<FixedLenByteArray> =
        raw.iter().map(|a| FixedLenByteArray::from(a.to_vec())).collect();
    write_flba_via_parquet_rs(&path, &values, 16);

    let body = first_data_page_body(&path);
    let decoded = decode_plain_fixed_len_byte_array(&body, 16).unwrap();
    assert_eq!(decoded.len(), raw.len());
    for (ours, theirs) in decoded.iter().zip(raw.iter()) {
        assert_eq!(*ours, theirs.as_slice());
    }
}

#[test]
fn flba_rejects_buffer_not_multiple_of_type_length() {
    // 10 bytes, type_length=4 — partial last value, must error.
    let bytes = vec![0u8; 10];
    let err = decode_plain_fixed_len_byte_array(&bytes, 4).unwrap_err();
    assert!(format!("{err}").to_lowercase().contains("plain") || format!("{err}").contains("4"));
}
