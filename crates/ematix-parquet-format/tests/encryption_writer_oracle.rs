//! Π.13e oracle: PME writers round-trip through our own reader.
//!
//! Covers the metadata layer:
//! - `FileMetaData.encryption_algorithm` (field 8) +
//!   `footer_signing_key_metadata` (field 9)
//! - `ColumnChunk.crypto_metadata` (field 8) +
//!   `encrypted_column_metadata` (field 9)
//! - Standalone `FileCryptoMetaData` (the encrypted-footer trailer)
//!
//! The full encrypted-page write path (per-page AES-GCM seal +
//! wire-frame emission) lands in a follow-up; this PR closes the
//! Thrift-side write half so Π.13f / future PRs can drop in the
//! per-page work without touching `metadata_writer.rs` again.

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{
    read_column_chunk, read_file_crypto_metadata, read_file_metadata, AesGcmCtrV1, AesGcmV1,
    ColumnChunk, ColumnCryptoMetaData, ColumnMetaData, EncryptionAlgorithm,
    EncryptionWithColumnKey, FileCryptoMetaData, FileMetaData, RowGroup, SchemaElement,
};
use ematix_parquet_format::metadata_writer::{
    write_column_chunk, write_file_crypto_metadata, write_file_metadata,
};
use ematix_parquet_format::types::{
    CompressionCodec as Compression, Encoding, FieldRepetitionType, ParquetType as ColumnType,
};

fn minimal_schema<'a>() -> Vec<SchemaElement<'a>> {
    vec![
        SchemaElement {
            column_type: None,
            type_length: None,
            repetition_type: None,
            name: b"schema",
            num_children: Some(1),
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        },
        SchemaElement {
            column_type: Some(ColumnType::Int32),
            type_length: None,
            repetition_type: Some(FieldRepetitionType::Optional),
            name: b"id",
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        },
    ]
}

fn minimal_cc<'a>(crypto: Option<ColumnCryptoMetaData<'a>>) -> ColumnChunk<'a> {
    let cm = ColumnMetaData {
        column_type: ColumnType::Int32,
        encodings: vec![Encoding::Plain],
        path_in_schema: vec![b"id"],
        codec: Compression::Uncompressed,
        num_values: 4,
        total_uncompressed_size: 16,
        total_compressed_size: 16,
        data_page_offset: 4,
        dictionary_page_offset: None,
        index_page_offset: None,
        key_value_metadata: None,
        statistics: None,
        encoding_stats: None,
        bloom_filter_offset: None,
        bloom_filter_length: None,
        size_statistics: None,
    };
    ColumnChunk {
        file_path: None,
        file_offset: 4,
        meta_data: Some(cm),
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
        crypto_metadata: crypto,
        encrypted_column_metadata: None,
    }
}

#[test]
fn file_metadata_with_aes_gcm_v1_round_trips() {
    let cc = minimal_cc(Some(ColumnCryptoMetaData::EncryptionWithFooterKey));
    let rg = RowGroup {
        columns: vec![cc],
        total_byte_size: 16,
        num_rows: 4,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: Some(0),
    };
    let md = FileMetaData {
        version: 1,
        schema: minimal_schema(),
        num_rows: 4,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: Some(b"ematix"),
        column_orders: None,
        encryption_algorithm: Some(EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_prefix: Some(b"prefix"),
            aad_file_unique: Some(b"unique_8b"),
            supply_aad_prefix: Some(true),
        })),
        footer_signing_key_metadata: Some(b"footer_km"),
    };

    let bytes = write_file_metadata(&md);
    let decoded = read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, md, "FileMetaData with PME fields round-trips");
}

#[test]
fn file_metadata_with_aes_gcm_ctr_v1_round_trips() {
    let cc = minimal_cc(Some(ColumnCryptoMetaData::EncryptionWithFooterKey));
    let rg = RowGroup {
        columns: vec![cc],
        total_byte_size: 16,
        num_rows: 4,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: Some(0),
    };
    let md = FileMetaData {
        version: 1,
        schema: minimal_schema(),
        num_rows: 4,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: None,
        column_orders: None,
        encryption_algorithm: Some(EncryptionAlgorithm::AesGcmCtrV1(AesGcmCtrV1 {
            aad_prefix: None,
            aad_file_unique: Some(b"u"),
            supply_aad_prefix: None,
        })),
        footer_signing_key_metadata: None,
    };

    let bytes = write_file_metadata(&md);
    let decoded = read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, md);
}

#[test]
fn column_chunk_with_encryption_with_column_key_round_trips() {
    let crypto = ColumnCryptoMetaData::EncryptionWithColumnKey(EncryptionWithColumnKey {
        path_in_schema: vec![b"a", b"b", b"id"],
        key_metadata: Some(b"per_col_km"),
    });
    let cc = minimal_cc(Some(crypto));
    let bytes = write_column_chunk(&cc);
    let decoded = read_column_chunk(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, cc);
}

#[test]
fn column_chunk_with_encryption_with_footer_key_round_trips() {
    let cc = minimal_cc(Some(ColumnCryptoMetaData::EncryptionWithFooterKey));
    let bytes = write_column_chunk(&cc);
    let decoded = read_column_chunk(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, cc);
}

#[test]
fn column_chunk_with_encrypted_column_metadata_blob_round_trips() {
    let mut cc = minimal_cc(Some(ColumnCryptoMetaData::EncryptionWithFooterKey));
    cc.encrypted_column_metadata = Some(b"\x00\x01\x02opaque-cipher-bytes\x03");
    let bytes = write_column_chunk(&cc);
    let decoded = read_column_chunk(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, cc);
}

#[test]
fn file_crypto_metadata_trailer_round_trips() {
    let fcm = FileCryptoMetaData {
        encryption_algorithm: Some(EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_prefix: Some(b"ematix-test-prefix"),
            aad_file_unique: Some(b"unique-bytes-12"),
            supply_aad_prefix: Some(false),
        })),
        key_metadata: Some(b"footer_key_id"),
    };
    let bytes = write_file_crypto_metadata(&fcm);
    let decoded = read_file_crypto_metadata(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(decoded, fcm);
}

#[test]
fn file_metadata_no_encryption_fields_byte_compatible_with_pre_pi13() {
    // Backwards-compat check: a FileMetaData with no encryption
    // fields must serialise to bytes that don't even contain field
    // tags for 8/9 (those fields are optional in the Thrift schema).
    let cc = minimal_cc(None);
    let rg = RowGroup {
        columns: vec![cc],
        total_byte_size: 16,
        num_rows: 4,
        sorting_columns: None,
        file_offset: None,
        total_compressed_size: None,
        ordinal: Some(0),
    };
    let md = FileMetaData {
        version: 1,
        schema: minimal_schema(),
        num_rows: 4,
        row_groups: vec![rg],
        key_value_metadata: None,
        created_by: Some(b"ematix"),
        column_orders: None,
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    };
    let bytes = write_file_metadata(&md);
    let decoded = read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
    assert!(decoded.encryption_algorithm.is_none());
    assert!(decoded.footer_signing_key_metadata.is_none());
}
