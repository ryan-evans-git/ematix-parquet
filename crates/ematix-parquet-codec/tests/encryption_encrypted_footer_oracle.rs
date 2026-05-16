//! Π.13d oracle: decrypt the **encrypted-footer** trailer that
//! parquet-rs emits when `FileEncryptionProperties` is given without
//! `with_plaintext_footer(true)`.
//!
//! Layout per spec:
//!
//! ```text
//! [ ... data ... ]
//! [ FileCryptoMetaData (Thrift) ]
//! [ encrypted FileMetaData ]   ← decrypted by `decrypt_footer`
//! [ footer_len: u32 LE ]
//! [ PARE ]
//! ```
//!
//! `footer_len` covers the FileCryptoMetaData + encrypted
//! FileMetaData together. Once we decrypt, the plaintext bytes feed
//! the existing `read_file_metadata` pipeline unchanged.

#![cfg(feature = "encryption")]

use std::fs::File;
use std::io::Read;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::encryption::encrypt::FileEncryptionProperties;
use parquet::file::properties::WriterProperties;

use ematix_parquet_codec::encrypted::{decrypt_footer, split_encrypted_footer_trailer};
use ematix_parquet_crypto::key::Key;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_file_metadata, AesGcmV1, EncryptionAlgorithm};

const FOOTER_KEY: &[u8; 16] = b"0123456789012345";

fn write_encrypted_footer_file() -> Vec<u8> {
    let enc_props = FileEncryptionProperties::builder((*FOOTER_KEY).into())
        // No `with_plaintext_footer(true)` → encrypted-footer mode
        // (the default when encryption properties are present).
        .build()
        .unwrap();
    let writer_props = WriterProperties::builder()
        .with_file_encryption_properties(enc_props)
        .build();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1i32, 2, 3, 4]))],
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let file = File::create(tmp.path()).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(writer_props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let mut file = File::open(tmp.path()).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    buf
}

fn extract_trailer(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[bytes.len() - 4..], b"PARE", "expected PARE magic");
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    &bytes[n - 8 - footer_len..n - 8]
}

#[test]
fn decrypt_footer_recovers_plaintext_file_metadata() {
    let bytes = write_encrypted_footer_file();
    let trailer = extract_trailer(&bytes);

    // Step 1: split the trailer into the cleartext FileCryptoMetaData
    // and the encrypted-FileMetaData ciphertext frame.
    let (fcm_bytes, enc_md_frame) = split_encrypted_footer_trailer(trailer).unwrap();
    assert!(!fcm_bytes.is_empty());
    assert!(!enc_md_frame.is_empty());

    // Step 2: pull the AAD info out of FileCryptoMetaData.
    use ematix_parquet_format::metadata::read_file_crypto_metadata;
    let fcm = read_file_crypto_metadata(&mut Cursor::new(fcm_bytes)).unwrap();
    let aad_file_unique = match fcm.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        other => panic!("expected AesGcmV1, got {other:?}"),
    };

    // Step 3: decrypt with the footer key. The plaintext bytes should
    // be a parseable FileMetaData.
    let key = Key::Aes128(*FOOTER_KEY);
    let plaintext_footer =
        decrypt_footer(enc_md_frame, &key, None, &aad_file_unique).expect("decrypt_footer");

    let md = read_file_metadata(&mut Cursor::new(&plaintext_footer))
        .expect("plaintext FileMetaData parses");

    assert_eq!(md.num_rows, 4, "row count round-trips");
    assert!(!md.row_groups.is_empty(), "row groups present");
    // In encrypted-footer mode the encryption_algorithm lives on
    // FileCryptoMetaData (the trailer) and is NOT repeated inside
    // the decrypted FileMetaData itself — readers must remember the
    // AAD bytes from the trailer. parquet-rs follows this
    // convention; we match it.
}

#[test]
fn decrypt_footer_wrong_key_fails() {
    let bytes = write_encrypted_footer_file();
    let trailer = extract_trailer(&bytes);
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

    let wrong_key = Key::Aes128(*b"WRONG_FOOTER_KEY");
    let err = decrypt_footer(enc_md_frame, &wrong_key, None, &aad_file_unique)
        .expect_err("wrong footer key must reject");
    let msg = format!("{err}");
    assert!(msg.contains("PME decrypt failed"), "got {msg}");
}

#[test]
fn split_trailer_recognises_file_crypto_metadata_boundary() {
    let bytes = write_encrypted_footer_file();
    let trailer = extract_trailer(&bytes);
    let (fcm_bytes, enc_md_frame) = split_encrypted_footer_trailer(trailer).unwrap();
    // Sanity: the two slices together cover the whole trailer with no
    // gap or overlap.
    assert_eq!(fcm_bytes.len() + enc_md_frame.len(), trailer.len());
    // FileCryptoMetaData ends with a Thrift STOP byte (0x00).
    assert_eq!(*fcm_bytes.last().unwrap(), 0x00);
    // Encrypted FileMetaData starts with its 4-byte size prefix.
    assert!(enc_md_frame.len() > 4 + 12 + 16, "frame has header + body");
}
