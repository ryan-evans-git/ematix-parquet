//! `Key` + `StaticKeys` + `KeyRetriever` trait coverage.

use ematix_parquet_crypto::error::CryptoError;
use ematix_parquet_crypto::key::{Key, KeyRetriever, StaticKeys};

#[test]
fn key_from_bytes_dispatches_on_length() {
    assert!(matches!(
        Key::from_bytes(&[0u8; 16]).unwrap(),
        Key::Aes128(_)
    ));
    assert!(matches!(
        Key::from_bytes(&[0u8; 24]).unwrap(),
        Key::Aes192(_)
    ));
    assert!(matches!(
        Key::from_bytes(&[0u8; 32]).unwrap(),
        Key::Aes256(_)
    ));
    assert!(matches!(
        Key::from_bytes(&[0u8; 15]),
        Err(CryptoError::InvalidKeyLength { got: 15 })
    ));
}

#[test]
fn static_keys_returns_footer_and_columns() {
    let mut keys = StaticKeys::new();
    keys.set_footer(Key::Aes128(*b"fff_fff_fff_fff_"))
        .set_column([b"x"], Key::Aes128(*b"xxx_xxx_xxx_xxx_"))
        .set_column(["a", "b"], Key::Aes256([7u8; 32]));

    assert_eq!(
        keys.footer_key(None).unwrap(),
        Key::Aes128(*b"fff_fff_fff_fff_")
    );
    assert_eq!(
        keys.column_key(&[b"x"], None).unwrap(),
        Key::Aes128(*b"xxx_xxx_xxx_xxx_")
    );
    assert_eq!(
        keys.column_key(&[b"a", b"b"], None).unwrap(),
        Key::Aes256([7u8; 32])
    );
}

#[test]
fn static_keys_missing_returns_keynotfound() {
    let keys = StaticKeys::new();
    assert!(matches!(
        keys.footer_key(None),
        Err(CryptoError::KeyNotFound { .. })
    ));
    assert!(matches!(
        keys.column_key(&[b"unknown"], None),
        Err(CryptoError::KeyNotFound { .. })
    ));
}
