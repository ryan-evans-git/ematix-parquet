//! Π.13c oracle: our `decrypt_module` primitive recovers the
//! plaintext PageHeader Thrift bytes from a parquet-rs-written
//! encrypted-page wire frame.
//!
//! Scope is intentionally narrow: this proves the AAD construction +
//! AES-GCM decrypt + on-disk wire layout all agree with parquet-rs.
//! Full read-façade integration (`read_column_*_encrypted` returning
//! decoded values end-to-end) lands together with encrypted-footer
//! mode in Π.13d.

#![cfg(feature = "encryption")]

use std::fs::File;
use std::io::Read;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::encryption::encrypt::FileEncryptionProperties;
use parquet::file::properties::WriterProperties;

use ematix_parquet_codec::encrypted::{decrypt_module, ColumnDecryptContext};
use ematix_parquet_crypto::aad::ModuleType;
use ematix_parquet_crypto::key::Key;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{
    read_file_metadata, read_page_header, AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm,
};

const FOOTER_KEY: &[u8; 16] = b"0123456789012345";

fn write_encrypted_i32_file(plaintext_footer: bool) -> Vec<u8> {
    let enc_props = FileEncryptionProperties::builder((*FOOTER_KEY).into())
        .with_plaintext_footer(plaintext_footer)
        .build()
        .unwrap();
    let writer_props = WriterProperties::builder()
        .with_file_encryption_properties(enc_props)
        .build();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5, 6, 7, 8]))],
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

fn extract_footer(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[bytes.len() - 4..], b"PAR1");
    let n = bytes.len();
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    &bytes[n - 8 - footer_len..n - 8]
}

#[test]
fn decrypt_module_recovers_page_header_from_parquet_rs_file() {
    // 1. Have parquet-rs write a small plaintext-footer encrypted
    //    i32 file (footer key only, so the column uses
    //    EncryptionWithFooterKey).
    let bytes = write_encrypted_i32_file(/*plaintext_footer=*/ true);

    // 2. Walk the footer to get the column chunk's data_page_offset
    //    and the AAD prefix bytes we need.
    let footer = extract_footer(&bytes);
    let md = read_file_metadata(&mut Cursor::new(footer)).unwrap();
    let aad_file_unique = match md.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        other => panic!("expected AesGcmV1, got {other:?}"),
    };

    let rg = &md.row_groups[0];
    let cc = &rg.columns[0];
    assert!(
        matches!(
            cc.crypto_metadata,
            Some(ColumnCryptoMetaData::EncryptionWithFooterKey)
        ),
        "expected EncryptionWithFooterKey on the single column"
    );
    let cm = cc.meta_data.as_ref().expect("meta_data on column chunk");
    let data_page_offset = cm.data_page_offset as usize;

    // 3. The bytes at data_page_offset are an encrypted DataPageHeader
    //    module (4-byte size || nonce || ct || tag). Decrypt with our
    //    primitive using ModuleType::DictionaryPageHeader if dict-encoded
    //    or DataPageHeader otherwise. Single-row-group, single-column,
    //    single-page file → page_ordinal = 0.
    let ctx = ColumnDecryptContext {
        key: Key::Aes128(*FOOTER_KEY),
        aad_prefix: None,
        aad_file_unique: &aad_file_unique,
        rg_ordinal: 0,
        col_ordinal: 0,
    };

    // parquet-rs writes a dictionary page first for small INT32
    // batches. Read its header first. dictionary_page_offset is set
    // when present.
    let first_offset = cm
        .dictionary_page_offset
        .map(|o| o as usize)
        .unwrap_or(data_page_offset);
    let first_module = if cm.dictionary_page_offset.is_some() {
        ModuleType::DictionaryPageHeader
    } else {
        ModuleType::DataPageHeader
    };

    let on_disk = &bytes[first_offset..];
    let page_ord = if first_module == ModuleType::DataPageHeader {
        Some(0i16)
    } else {
        None
    };
    let (plaintext, _consumed) =
        decrypt_module(on_disk, &ctx, first_module, page_ord).expect("decrypt module");

    // 4. Plaintext should be a parseable PageHeader Thrift struct.
    let mut cur = Cursor::new(&plaintext);
    let header = read_page_header(&mut cur).expect("parseable PageHeader");
    // Either DictionaryPage or DataPage type — both are valid; the
    // batch is small enough that parquet-rs typically dict-encodes.
    assert!(
        header.uncompressed_page_size > 0,
        "page size should be populated"
    );
    assert!(
        header.compressed_page_size > 0,
        "compressed page size should be populated"
    );
}

#[test]
fn decrypt_module_wrong_key_returns_error() {
    let bytes = write_encrypted_i32_file(true);
    let footer = extract_footer(&bytes);
    let md = read_file_metadata(&mut Cursor::new(footer)).unwrap();
    let aad_file_unique = match md.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        _ => unreachable!(),
    };
    let cc = &md.row_groups[0].columns[0];
    let cm = cc.meta_data.as_ref().unwrap();
    let first_offset = cm
        .dictionary_page_offset
        .map(|o| o as usize)
        .unwrap_or(cm.data_page_offset as usize);
    let first_module = if cm.dictionary_page_offset.is_some() {
        ModuleType::DictionaryPageHeader
    } else {
        ModuleType::DataPageHeader
    };

    // Wrong key → AES-GCM tag check fails.
    let bad_ctx = ColumnDecryptContext {
        key: Key::Aes128(*b"WRONG_____KEY___"),
        aad_prefix: None,
        aad_file_unique: &aad_file_unique,
        rg_ordinal: 0,
        col_ordinal: 0,
    };
    let page_ord = if first_module == ModuleType::DataPageHeader {
        Some(0i16)
    } else {
        None
    };
    let err = decrypt_module(&bytes[first_offset..], &bad_ctx, first_module, page_ord)
        .expect_err("wrong key must reject");
    let msg = format!("{err}");
    assert!(msg.contains("PME decrypt failed"), "got {msg}");
}

#[test]
fn decrypt_module_wrong_aad_ordinal_returns_error() {
    let bytes = write_encrypted_i32_file(true);
    let footer = extract_footer(&bytes);
    let md = read_file_metadata(&mut Cursor::new(footer)).unwrap();
    let aad_file_unique = match md.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        _ => unreachable!(),
    };
    let cc = &md.row_groups[0].columns[0];
    let cm = cc.meta_data.as_ref().unwrap();
    let first_offset = cm
        .dictionary_page_offset
        .map(|o| o as usize)
        .unwrap_or(cm.data_page_offset as usize);
    let first_module = if cm.dictionary_page_offset.is_some() {
        ModuleType::DictionaryPageHeader
    } else {
        ModuleType::DataPageHeader
    };

    // Right key, wrong AAD (col_ordinal = 1 instead of 0).
    let bad_ctx = ColumnDecryptContext {
        key: Key::Aes128(*FOOTER_KEY),
        aad_prefix: None,
        aad_file_unique: &aad_file_unique,
        rg_ordinal: 0,
        col_ordinal: 1, // wrong!
    };
    let page_ord = if first_module == ModuleType::DataPageHeader {
        Some(0i16)
    } else {
        None
    };
    let err = decrypt_module(&bytes[first_offset..], &bad_ctx, first_module, page_ord)
        .expect_err("wrong AAD must reject");
    let msg = format!("{err}");
    assert!(msg.contains("PME decrypt failed"), "got {msg}");
}
