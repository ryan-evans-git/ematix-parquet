//! `build_module_aad` byte-layout pinning tests.
//!
//! These spell out the exact byte sequence we expect each module type
//! to produce — a regression here means we've drifted from the PME
//! spec and a Parquet PME oracle would silently fail with
//! AuthenticationFailed instead of telling us why.

use ematix_parquet_crypto::aad::{build_module_aad, ModuleType};

#[test]
fn footer_aad_is_module_byte_only() {
    let aad = build_module_aad(
        None,
        b"FU\x00\x00\x00\x00\x00\x00", // 8 bytes file_unique
        ModuleType::Footer,
        0,
        0,
        None,
    );
    // Footer is the special case: file_unique || module (0). No
    // rg/col/page suffix per spec — matches parquet-rs.
    assert_eq!(aad, b"FU\x00\x00\x00\x00\x00\x00\x00");
}

#[test]
fn data_page_has_page_ordinal_and_prefix() {
    let aad = build_module_aad(
        Some(b"PREFIX"),
        b"FILEUNIQ",
        ModuleType::DataPage, // module = 2
        7,                    // rg_ordinal LE = 07 00
        3,                    // col_ordinal LE = 03 00
        Some(42),             // page_ordinal LE = 2a 00 (i16, 2 bytes)
    );
    let expected: Vec<u8> = b"PREFIX"
        .iter()
        .chain(b"FILEUNIQ".iter())
        .chain([0x02u8, 0x07, 0x00, 0x03, 0x00, 0x2a, 0x00].iter())
        .copied()
        .collect();
    assert_eq!(aad, expected);
}

#[test]
fn column_metadata_omits_page_ordinal() {
    let aad = build_module_aad(
        None,
        b"FU",
        ModuleType::ColumnMetaData, // module = 1
        -1i16,                      // LE = FF FF
        2,                          // LE = 02 00
        None,
    );
    assert_eq!(
        aad,
        // FU || 01 || FF FF || 02 00
        b"FU\x01\xFF\xFF\x02\x00"
    );
}

#[test]
fn module_type_byte_values_match_spec() {
    // Module type enum discriminants per Encryption.md table.
    assert_eq!(ModuleType::Footer as u8, 0);
    assert_eq!(ModuleType::ColumnMetaData as u8, 1);
    assert_eq!(ModuleType::DataPage as u8, 2);
    assert_eq!(ModuleType::DictionaryPage as u8, 3);
    assert_eq!(ModuleType::DataPageHeader as u8, 4);
    assert_eq!(ModuleType::DictionaryPageHeader as u8, 5);
    assert_eq!(ModuleType::ColumnIndex as u8, 6);
    assert_eq!(ModuleType::OffsetIndex as u8, 7);
    assert_eq!(ModuleType::BloomFilterHeader as u8, 8);
    assert_eq!(ModuleType::BloomFilterBitset as u8, 9);
}

#[test]
fn has_page_ordinal_only_for_data_page_modules() {
    // Spec: only DataPage + DataPageHeader carry the page-ordinal
    // suffix. Dictionary* don't (there's only one dict page per
    // column chunk; the suffix would always be 0 → no extra signal).
    assert!(ModuleType::DataPage.has_page_ordinal());
    assert!(ModuleType::DataPageHeader.has_page_ordinal());
    assert!(!ModuleType::DictionaryPage.has_page_ordinal());
    assert!(!ModuleType::DictionaryPageHeader.has_page_ordinal());
    assert!(!ModuleType::Footer.has_page_ordinal());
    assert!(!ModuleType::ColumnMetaData.has_page_ordinal());
    assert!(!ModuleType::ColumnIndex.has_page_ordinal());
    assert!(!ModuleType::OffsetIndex.has_page_ordinal());
    assert!(!ModuleType::BloomFilterHeader.has_page_ordinal());
    assert!(!ModuleType::BloomFilterBitset.has_page_ordinal());
}

#[test]
fn aes_gcm_round_trip_uses_real_aad() {
    // Sanity: an AAD built by `build_module_aad` round-trips through
    // seal/open. Catches any AAD non-determinism between calls.
    use ematix_parquet_crypto::aead::{open, seal};
    use ematix_parquet_crypto::key::Key;

    let key = Key::Aes128(*b"the_test_key_16!");
    let nonce = [9u8; 12];
    let aad = build_module_aad(
        Some(b"prefix"),
        b"file_unique_8b",
        ModuleType::DataPage,
        0,
        1,
        Some(0i16),
    );

    let pt = b"some page bytes worth protecting";
    let ct = seal(&key, &nonce, &aad, pt).unwrap();
    let recovered = open(&key, &nonce, &aad, &ct).unwrap();
    assert_eq!(recovered, pt);
}
