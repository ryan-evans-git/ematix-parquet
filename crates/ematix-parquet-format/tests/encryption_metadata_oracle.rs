//! Π.13a oracle: parse encryption descriptors from parquet-rs PME files.
//!
//! No decrypt yet — these tests only confirm we can recognise + parse
//! the Thrift extensions parquet-rs emits in:
//!   - **plaintext-footer mode**: `FileMetaData.encryption_algorithm`
//!     (field 8) + `ColumnChunk.crypto_metadata` (field 8).
//!   - **encrypted-footer mode**: the `FileCryptoMetaData` trailer that
//!     replaces `FileMetaData` on disk, behind the `PARE` magic.
//!
//! Π.13c/d wire in the actual AES-GCM decrypt and add round-trip-value
//! oracle tests on top of this metadata layer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::encryption::encrypt::FileEncryptionProperties;
use parquet::file::properties::WriterProperties;

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{
    read_file_crypto_metadata, read_file_metadata, AesGcmV1, ColumnCryptoMetaData,
    EncryptionAlgorithm,
};

const FOOTER_KEY: &[u8; 16] = b"0123456789012345";
const COLUMN_KEY_X: &[u8; 16] = b"1234567890123450";
const COLUMN_KEY_Y: &[u8; 16] = b"1234567890123451";

/// Write a small encrypted Parquet file with parquet-rs and return
/// the on-disk bytes.
fn write_encrypted_parquet(
    plaintext_footer: bool,
    with_column_keys: bool,
    aad_prefix: Option<&[u8]>,
) -> Vec<u8> {
    let mut builder = FileEncryptionProperties::builder((*FOOTER_KEY).into());
    if with_column_keys {
        builder = builder
            .with_column_key("x", (*COLUMN_KEY_X).into())
            .with_column_key("y", (*COLUMN_KEY_Y).into());
    }
    if let Some(prefix) = aad_prefix {
        builder = builder
            .with_aad_prefix(prefix.to_vec())
            .with_aad_prefix_storage(true);
    }
    let enc_props = builder
        .with_plaintext_footer(plaintext_footer)
        .build()
        .unwrap();

    let writer_props = WriterProperties::builder()
        .with_file_encryption_properties(enc_props)
        .build();

    // Three columns named id / x / y so the per-column-key path lights
    // up cleanly.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0i32, 1, 2, 3])),
            Arc::new(Int32Array::from(vec![10i32, 20, 30, 40])),
            Arc::new(Int32Array::from(vec![100i32, 200, 300, 400])),
        ],
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

/// Slice out the `FileMetaData` Thrift bytes from the tail of a
/// **plaintext-footer** Parquet file (magic = "PAR1"):
///
///   [... data ...] [FileMetaData (thrift)] [footer_len: u32 LE] [PAR1]
fn extract_plaintext_footer(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[bytes.len() - 4..], b"PAR1", "expected PAR1 magic");
    let n = bytes.len();
    let len_bytes: [u8; 4] = bytes[n - 8..n - 4].try_into().unwrap();
    let footer_len = u32::from_le_bytes(len_bytes) as usize;
    &bytes[n - 8 - footer_len..n - 8]
}

/// Slice out the `FileCryptoMetaData` Thrift bytes from the tail of
/// an **encrypted-footer** Parquet file (magic = "PARE"). The
/// FileCryptoMetaData lives at the start of the footer-length region;
/// the encrypted FileMetaData ciphertext follows it and runs to the
/// length-prefix byte. We only need the FileCryptoMetaData slice; our
/// Thrift reader consumes up to STOP and stops there, so we can hand
/// it the whole trailer region without issue.
fn extract_encrypted_footer_trailer(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[bytes.len() - 4..], b"PARE", "expected PARE magic");
    let n = bytes.len();
    let len_bytes: [u8; 4] = bytes[n - 8..n - 4].try_into().unwrap();
    let footer_len = u32::from_le_bytes(len_bytes) as usize;
    &bytes[n - 8 - footer_len..n - 8]
}

#[test]
fn plaintext_footer_mode_parses_encryption_algorithm() {
    let bytes = write_encrypted_parquet(/*plaintext_footer=*/ true, false, None);

    let footer = extract_plaintext_footer(&bytes);
    let mut cur = Cursor::new(footer);
    let md = read_file_metadata(&mut cur).expect("read_file_metadata on plaintext footer");

    // Encryption algorithm present and is AesGcmV1.
    match md
        .encryption_algorithm
        .expect("encryption_algorithm should be present")
    {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => {
            assert!(
                aad_file_unique.is_some(),
                "aad_file_unique should always be emitted by parquet-rs"
            );
        }
        other => panic!("expected AesGcmV1, got {other:?}"),
    }

    // Every column chunk should carry crypto_metadata (no per-column
    // keys → footer-key variant on every column).
    let rg = &md.row_groups[0];
    assert_eq!(rg.columns.len(), 3);
    for col in &rg.columns {
        match col
            .crypto_metadata
            .as_ref()
            .expect("crypto_metadata expected")
        {
            ColumnCryptoMetaData::EncryptionWithFooterKey => {}
            other => panic!("expected EncryptionWithFooterKey, got {other:?}"),
        }
    }
}

#[test]
fn plaintext_footer_mode_with_per_column_keys() {
    let bytes = write_encrypted_parquet(/*plaintext_footer=*/ true, true, None);

    let footer = extract_plaintext_footer(&bytes);
    let mut cur = Cursor::new(footer);
    let md = read_file_metadata(&mut cur).expect("read_file_metadata");

    assert!(matches!(
        md.encryption_algorithm,
        Some(EncryptionAlgorithm::AesGcmV1(_))
    ));

    let rg = &md.row_groups[0];
    // Per parquet-rs semantics: columns without an explicit
    // `with_column_key` are written **unencrypted** (crypto_metadata =
    // None). Only `x` and `y` carry per-column keys here.
    let by_name: std::collections::HashMap<String, &_> = rg
        .columns
        .iter()
        .map(|cc| {
            let cm = cc.meta_data.as_ref().expect("meta_data on each column");
            let name = std::str::from_utf8(cm.path_in_schema[0])
                .unwrap()
                .to_string();
            (name, cc)
        })
        .collect();

    assert!(
        by_name["id"].crypto_metadata.is_none(),
        "`id` was not given a column key → expected to be left unencrypted"
    );
    match by_name["x"]
        .crypto_metadata
        .as_ref()
        .expect("x crypto_metadata")
    {
        ColumnCryptoMetaData::EncryptionWithColumnKey(k) => {
            assert_eq!(k.path_in_schema, vec![b"x".as_ref()]);
        }
        other => panic!("x: expected EncryptionWithColumnKey, got {other:?}"),
    }
    match by_name["y"]
        .crypto_metadata
        .as_ref()
        .expect("y crypto_metadata")
    {
        ColumnCryptoMetaData::EncryptionWithColumnKey(k) => {
            assert_eq!(k.path_in_schema, vec![b"y".as_ref()]);
        }
        other => panic!("y: expected EncryptionWithColumnKey, got {other:?}"),
    }
}

#[test]
fn plaintext_footer_mode_with_aad_prefix_stored() {
    let prefix = b"ematix-test-aad-prefix";
    let bytes = write_encrypted_parquet(true, false, Some(prefix));

    let footer = extract_plaintext_footer(&bytes);
    let mut cur = Cursor::new(footer);
    let md = read_file_metadata(&mut cur).expect("read_file_metadata");

    match md.encryption_algorithm.expect("encryption_algorithm") {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_prefix: stored, ..
        }) => {
            assert_eq!(
                stored,
                Some(prefix.as_ref()),
                "with_aad_prefix_storage(true) should embed prefix in the metadata"
            );
        }
        other => panic!("expected AesGcmV1, got {other:?}"),
    }
}

#[test]
fn encrypted_footer_mode_trailer_parses() {
    let bytes = write_encrypted_parquet(/*plaintext_footer=*/ false, false, None);

    let trailer = extract_encrypted_footer_trailer(&bytes);
    let mut cur = Cursor::new(trailer);
    let fcm =
        read_file_crypto_metadata(&mut cur).expect("read_file_crypto_metadata on PARE trailer");

    match fcm
        .encryption_algorithm
        .expect("encryption_algorithm required")
    {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => {
            assert!(
                aad_file_unique.is_some(),
                "encrypted-footer trailer must carry aad_file_unique"
            );
        }
        other => panic!("expected AesGcmV1, got {other:?}"),
    }
}

#[test]
fn encrypted_footer_mode_magic_is_pare_not_par1() {
    let bytes = write_encrypted_parquet(false, false, None);
    assert_eq!(
        &bytes[bytes.len() - 4..],
        b"PARE",
        "encrypted-footer mode emits PARE magic"
    );
}

#[test]
fn plaintext_footer_mode_magic_is_par1_not_pare() {
    let bytes = write_encrypted_parquet(true, false, None);
    assert_eq!(
        &bytes[bytes.len() - 4..],
        b"PAR1",
        "plaintext-footer mode keeps PAR1 magic"
    );
}

/// Silence dead-code lint: helper kept for the encrypted-footer mode
/// regression test even when only the plaintext branch is exercised.
#[allow(dead_code)]
fn _drop_seek(_: &mut File) -> std::io::Result<u64> {
    let mut f = File::open("/dev/null")?;
    f.seek(SeekFrom::End(0))
}
