//! Π.13e oracle: write an encrypted Parquet file end-to-end and
//! confirm both parquet-rs and our own reader can recover the values.
//!
//! Mode: plaintext footer (PAR1 magic) + per-column encrypted data
//! pages under the footer key. This is the minimum-viable surface
//! the v0.6.0 write path needs to be usable for real PME workloads;
//! per-column-key + encrypted-footer-mode writes follow in Π.13f.

#![cfg(feature = "encryption")]

use std::fs::File;
use std::io::Read;

use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::encryption::decrypt::FileDecryptionProperties;

use ematix_parquet_codec::encrypted::{decrypt_module, ColumnDecryptContext};
use ematix_parquet_codec::write::write_i32_column_to_path_encrypted;
use ematix_parquet_crypto::aad::ModuleType;
use ematix_parquet_crypto::key::Key;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{
    read_file_metadata, read_page_header, AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm,
};
use ematix_parquet_format::types::PageType;

const FOOTER_KEY: [u8; 16] = *b"0123456789abcdef";

fn extract_footer(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[bytes.len() - 4..], b"PAR1");
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    &bytes[n - 8 - footer_len..n - 8]
}

/// parquet-rs interop: end-to-end round-trip through their reader.
#[test]
fn we_write_encrypted_i32_parquet_rs_reads() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = (0..32).map(|i| i * 7 - 3).collect();
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted(tmp.path(), "id", &values, &key, None).unwrap();

    let dec_props = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
        .build()
        .unwrap();
    let reader_opts = ArrowReaderOptions::new().with_file_decryption_properties(dec_props);
    let file = File::open(tmp.path()).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, reader_opts).unwrap();
    let mut reader = builder.build().unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none(), "expected single batch");

    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();
    let got: Vec<i32> = col.values().to_vec();
    assert_eq!(got, values, "parquet-rs decoded the values we wrote");
}

#[test]
fn we_write_encrypted_we_read_back_via_decrypt_module() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted(tmp.path(), "id", &values, &key, None).unwrap();

    // Read the file bytes back manually.
    let mut buf = Vec::new();
    File::open(tmp.path())
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    // 1. Parse footer (plaintext-footer mode → standard PAR1).
    let footer = extract_footer(&buf);
    let md = read_file_metadata(&mut Cursor::new(footer)).unwrap();

    // 2. Footer must advertise AesGcmV1.
    let aad_file_unique = match md.encryption_algorithm.expect("encryption_algorithm") {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        other => panic!("expected AesGcmV1, got {other:?}"),
    };

    // 3. Column chunk must advertise EncryptionWithFooterKey.
    let cc = &md.row_groups[0].columns[0];
    assert!(matches!(
        cc.crypto_metadata,
        Some(ColumnCryptoMetaData::EncryptionWithFooterKey)
    ));
    let cm = cc.meta_data.as_ref().unwrap();

    // 4. Decrypt the DataPageHeader at data_page_offset.
    let ctx = ColumnDecryptContext {
        key: key.clone(),
        aad_prefix: None,
        aad_file_unique: &aad_file_unique,
        rg_ordinal: 0,
        col_ordinal: 0,
    };
    let on_disk = &buf[cm.data_page_offset as usize..];
    let (header_plaintext, consumed_header) =
        decrypt_module(on_disk, &ctx, ModuleType::DataPageHeader, Some(0))
            .expect("decrypt data page header");

    let page_header = read_page_header(&mut Cursor::new(&header_plaintext)).unwrap();
    assert_eq!(page_header.page_type, PageType::DataPage);
    assert_eq!(
        page_header.uncompressed_page_size as usize,
        values.len() * 4
    );

    // 5. Decrypt the data page body that follows.
    let body_frame = &on_disk[consumed_header..];
    let (body_plaintext, _consumed_body) =
        decrypt_module(body_frame, &ctx, ModuleType::DataPage, Some(0))
            .expect("decrypt data page body");

    // 6. Body is uncompressed PLAIN i32. Decode and compare.
    let decoded: Vec<i32> = body_plaintext
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(decoded, values, "self round-trip recovers the values");
}

#[test]
fn we_write_encrypted_wrong_key_parquet_rs_rejects() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let values: Vec<i32> = vec![1, 2, 3];
    let key = Key::Aes128(FOOTER_KEY);

    write_i32_column_to_path_encrypted(tmp.path(), "id", &values, &key, None).unwrap();

    let wrong_key: [u8; 16] = *b"WRONG_KEY_______";
    let dec_props = FileDecryptionProperties::builder(wrong_key.to_vec())
        .build()
        .unwrap();
    let reader_opts = ArrowReaderOptions::new().with_file_decryption_properties(dec_props);
    let file = File::open(tmp.path()).unwrap();
    let result = ParquetRecordBatchReaderBuilder::try_new_with_options(file, reader_opts);
    assert!(result.is_err(), "parquet-rs must reject wrong footer key");
}
