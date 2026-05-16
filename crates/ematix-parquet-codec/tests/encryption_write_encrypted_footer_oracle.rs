//! Π.13f oracle: write an **encrypted-footer** Parquet file (PARE
//! magic) and confirm both parquet-rs and our own reader round-trip
//! it. Reuses the per-page encryption pipeline from Π.13e — this
//! adds the FileCryptoMetaData trailer + encrypted-FileMetaData
//! wrapping on the write side.

#![cfg(feature = "encryption")]

use std::fs::File;
use std::io::Read;

use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::encryption::decrypt::FileDecryptionProperties;

use ematix_parquet_codec::encrypted::{decrypt_footer, split_encrypted_footer_trailer};
use ematix_parquet_codec::write::write_i32_column_to_path_encrypted_footer;
use ematix_parquet_crypto::key::Key;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_file_metadata, AesGcmV1, EncryptionAlgorithm};

const FOOTER_KEY: [u8; 16] = *b"0123456789abcdef";

fn read_file(path: &std::path::Path) -> Vec<u8> {
    let mut buf = Vec::new();
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    buf
}

#[test]
fn encrypted_footer_round_trip_via_parquet_rs() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = (0..16).map(|i| i * 11 - 5).collect();
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted_footer(tmp.path(), "id", &values, &key, None).unwrap();

    // Magic check first — should be PARE, not PAR1.
    let bytes = read_file(tmp.path());
    assert_eq!(
        &bytes[bytes.len() - 4..],
        b"PARE",
        "encrypted-footer mode emits PARE magic"
    );

    // parquet-rs reads with the footer key.
    let dec_props = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
        .build()
        .unwrap();
    let reader_opts = ArrowReaderOptions::new().with_file_decryption_properties(dec_props);
    let file = File::open(tmp.path()).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, reader_opts).unwrap();
    let mut reader = builder.build().unwrap();
    let batch = reader.next().unwrap().unwrap();
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();
    let got: Vec<i32> = col.values().to_vec();
    assert_eq!(got, values, "parquet-rs decoded our encrypted-footer file");
}

#[test]
fn encrypted_footer_round_trip_via_decrypt_footer() {
    // Our own readers (decrypt_footer + read_file_metadata) round-trip
    // the trailer back to a parseable plaintext FileMetaData.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = vec![100, 200, 300];
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted_footer(tmp.path(), "id", &values, &key, None).unwrap();

    let bytes = read_file(tmp.path());
    assert_eq!(&bytes[bytes.len() - 4..], b"PARE");

    // Extract trailer.
    let n = bytes.len();
    let flen = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let trailer = &bytes[n - 8 - flen..n - 8];

    // Split + decrypt.
    let (fcm_bytes, enc_md_frame) = split_encrypted_footer_trailer(trailer).unwrap();
    let fcm =
        ematix_parquet_format::metadata::read_file_crypto_metadata(&mut Cursor::new(fcm_bytes))
            .unwrap();
    let aad_file_unique = match fcm.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        _ => unreachable!(),
    };
    let plaintext_md = decrypt_footer(enc_md_frame, &key, None, &aad_file_unique).unwrap();
    let md = read_file_metadata(&mut Cursor::new(&plaintext_md)).unwrap();

    assert_eq!(md.num_rows, values.len() as i64);
    assert!(!md.row_groups.is_empty());
    // In encrypted-footer mode, the decrypted FileMetaData does not
    // repeat encryption_algorithm — that field lives in
    // FileCryptoMetaData (the trailer) instead.
    assert!(md.encryption_algorithm.is_none());
}

#[test]
fn encrypted_footer_wrong_key_rejected_by_parquet_rs() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = vec![1, 2, 3];
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted_footer(tmp.path(), "id", &values, &key, None).unwrap();

    let wrong: [u8; 16] = *b"WRONG_FOOTER_KEY";
    let dec_props = FileDecryptionProperties::builder(wrong.to_vec())
        .build()
        .unwrap();
    let reader_opts = ArrowReaderOptions::new().with_file_decryption_properties(dec_props);
    let file = File::open(tmp.path()).unwrap();
    let r = ParquetRecordBatchReaderBuilder::try_new_with_options(file, reader_opts);
    assert!(r.is_err(), "wrong footer key must reject");
}
