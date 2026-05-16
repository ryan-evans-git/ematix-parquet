//! Π.13f example: rotate the footer key on an existing encrypted
//! Parquet file by decrypting with the old key and re-encrypting
//! with a new one.
//!
//! Run: `cargo run --release --example key_rotation -p ematix-parquet-codec --features encryption`
//!
//! Shape:
//!   1. Write a small encrypted-footer file under OLD_KEY.
//!   2. Open it, decrypt the footer + every page, decode the values.
//!   3. Re-write the file under NEW_KEY using the encrypted-footer
//!      write path.
//!   4. Confirm the new file can ONLY be opened by NEW_KEY (OLD_KEY
//!      rejected).
//!
//! In production the values would be streamed page-by-page rather
//! than fully materialised, and the rotation would touch a whole
//! dataset rather than a single file — but the per-file primitive
//! is the same.

#[cfg(not(feature = "encryption"))]
fn main() {
    eprintln!("This example requires `--features encryption`. Run:");
    eprintln!(
        "  cargo run --release --example key_rotation -p ematix-parquet-codec --features encryption"
    );
}

#[cfg(feature = "encryption")]
fn main() {
    use ematix_parquet_codec::encrypted::{decrypt_module, ColumnDecryptContext};
    use ematix_parquet_codec::write::write_i32_column_to_path_encrypted_footer;
    use ematix_parquet_crypto::aad::ModuleType;
    use ematix_parquet_crypto::key::Key;
    use ematix_parquet_format::compact::Cursor;
    use ematix_parquet_format::metadata::{
        read_file_metadata, AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm,
    };
    use ematix_parquet_format::types::PageType;
    use std::fs::File;
    use std::io::Read;

    const OLD_KEY: [u8; 16] = *b"old_key_16_bytes";
    const NEW_KEY: [u8; 16] = *b"new_key_16_bytes";
    const COLUMN_NAME: &str = "id";

    let dir = tempfile::tempdir().unwrap();
    let old_path = dir.path().join("data_v1.parquet");
    let new_path = dir.path().join("data_v2.parquet");

    let values: Vec<i32> = (0..1024).map(|i| i * 3 + 1).collect();
    println!(
        "Step 1: writing {} values to {} under OLD_KEY",
        values.len(),
        old_path.display()
    );
    write_i32_column_to_path_encrypted_footer(
        &old_path,
        COLUMN_NAME,
        &values,
        &Key::Aes128(OLD_KEY),
        None,
    )
    .unwrap();

    println!("Step 2: decrypting with OLD_KEY, decoding values");
    let bytes = {
        let mut buf = Vec::new();
        File::open(&old_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        buf
    };
    // Parse the encrypted-footer trailer for the AAD info.
    let n = bytes.len();
    let flen = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let trailer = &bytes[n - 8 - flen..n - 8];
    let (fcm_bytes, enc_md_frame) =
        ematix_parquet_codec::encrypted::split_encrypted_footer_trailer(trailer).unwrap();
    let fcm =
        ematix_parquet_format::metadata::read_file_crypto_metadata(&mut Cursor::new(fcm_bytes))
            .unwrap();
    let aad_file_unique = match fcm.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        _ => unreachable!(),
    };
    let plaintext_md = ematix_parquet_codec::encrypted::decrypt_footer(
        enc_md_frame,
        &Key::Aes128(OLD_KEY),
        None,
        &aad_file_unique,
    )
    .unwrap();
    let md = read_file_metadata(&mut Cursor::new(&plaintext_md)).unwrap();
    let cc = &md.row_groups[0].columns[0];
    assert!(matches!(
        cc.crypto_metadata,
        Some(ColumnCryptoMetaData::EncryptionWithFooterKey)
    ));
    let cm = cc.meta_data.as_ref().unwrap();

    let ctx = ColumnDecryptContext {
        key: Key::Aes128(OLD_KEY),
        aad_prefix: None,
        aad_file_unique: &aad_file_unique,
        rg_ordinal: 0,
        col_ordinal: 0,
    };
    let on_disk = &bytes[cm.data_page_offset as usize..];
    let (header_bytes, consumed_header) =
        decrypt_module(on_disk, &ctx, ModuleType::DataPageHeader, Some(0)).unwrap();
    let header =
        ematix_parquet_format::metadata::read_page_header(&mut Cursor::new(&header_bytes)).unwrap();
    assert_eq!(header.page_type, PageType::DataPage);

    let body_frame = &on_disk[consumed_header..];
    let (body_plaintext, _) =
        decrypt_module(body_frame, &ctx, ModuleType::DataPage, Some(0)).unwrap();
    let decoded: Vec<i32> = body_plaintext
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(decoded, values, "decryption with OLD_KEY recovers values");
    println!("  ✓ decoded {} values successfully", decoded.len());

    println!("Step 3: re-writing under NEW_KEY → {}", new_path.display());
    write_i32_column_to_path_encrypted_footer(
        &new_path,
        COLUMN_NAME,
        &decoded,
        &Key::Aes128(NEW_KEY),
        None,
    )
    .unwrap();

    println!("Step 4: confirming OLD_KEY no longer decrypts the rotated file");
    let bytes2 = {
        let mut buf = Vec::new();
        File::open(&new_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        buf
    };
    let n2 = bytes2.len();
    let flen2 = u32::from_le_bytes(bytes2[n2 - 8..n2 - 4].try_into().unwrap()) as usize;
    let trailer2 = &bytes2[n2 - 8 - flen2..n2 - 8];
    let (fcm2_bytes, enc_md2) =
        ematix_parquet_codec::encrypted::split_encrypted_footer_trailer(trailer2).unwrap();
    let fcm2 =
        ematix_parquet_format::metadata::read_file_crypto_metadata(&mut Cursor::new(fcm2_bytes))
            .unwrap();
    let new_aad_unique = match fcm2.encryption_algorithm.unwrap() {
        EncryptionAlgorithm::AesGcmV1(AesGcmV1 {
            aad_file_unique, ..
        }) => aad_file_unique.unwrap().to_vec(),
        _ => unreachable!(),
    };

    let attempt_old = ematix_parquet_codec::encrypted::decrypt_footer(
        enc_md2,
        &Key::Aes128(OLD_KEY),
        None,
        &new_aad_unique,
    );
    assert!(
        attempt_old.is_err(),
        "OLD_KEY must NOT decrypt the rotated file"
    );
    println!("  ✓ OLD_KEY rejected by the rotated file");

    let attempt_new = ematix_parquet_codec::encrypted::decrypt_footer(
        enc_md2,
        &Key::Aes128(NEW_KEY),
        None,
        &new_aad_unique,
    );
    assert!(attempt_new.is_ok(), "NEW_KEY must decrypt the rotated file");
    println!("  ✓ NEW_KEY successfully decrypts the rotated file");
    println!("Key rotation complete.");
}
