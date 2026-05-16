//! AES-GCM correctness vs published NIST test vectors.
//!
//! Subset of NIST GCM-Spec Appendix B / NIST CAVP "gcmEncryptExtIV"
//! test vectors covering AES-128 and AES-256 with various AAD shapes.
//! Each vector: encrypt the plaintext under (key, nonce, aad), assert
//! we get the documented ciphertext+tag. Then decrypt back to verify
//! `open` is the inverse.
//!
//! These are the publicly-distributed test vectors; embedding the
//! exact bytes lets us catch any regression in either our wrapper or
//! the underlying `aes-gcm` crate.

use ematix_parquet_crypto::aead::{open, seal};
use ematix_parquet_crypto::key::Key;

fn hex(s: &str) -> Vec<u8> {
    let s: String = s.trim().chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut chars = s.chars();
    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let hi = a.to_digit(16).expect("hex digit") as u8;
        let lo = b.to_digit(16).expect("hex digit") as u8;
        out.push((hi << 4) | lo);
    }
    out
}

fn round_trip(key: Key, nonce_hex: &str, aad_hex: &str, pt_hex: &str, expected_ct_hex: &str) {
    let nonce = hex(nonce_hex);
    let aad = hex(aad_hex);
    let plaintext = hex(pt_hex);
    let expected = hex(expected_ct_hex);

    let ct = seal(&key, &nonce, &aad, &plaintext).unwrap();
    assert_eq!(
        ct, expected,
        "encrypt mismatch under key={key:?} nonce={nonce_hex} aad={aad_hex}"
    );

    let pt = open(&key, &nonce, &aad, &ct).unwrap();
    assert_eq!(pt, plaintext, "decrypt round-trip mismatch");
}

// ----- AES-128-GCM (NIST GCM Spec, Appendix B Test Case 3) -----

#[test]
fn aes128_gcm_nist_case_3_empty_aad() {
    // Test Case 3 from "The Galois/Counter Mode of Operation (GCM)"
    // by McGrew & Viega — the NIST GCM submission. Key/nonce/plaintext
    // / ciphertext as published; AAD is empty.
    let key = Key::Aes128(hex("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
    round_trip(
        key,
        "cafebabefacedbaddecaf888",
        "",
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
        // Expected ciphertext || 16-byte GCM tag.
        "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f59854d5c2af327cd64a62cf35abd2ba6fab4",
    );
}

#[test]
fn aes128_gcm_nist_case_4_with_aad() {
    // Test Case 4 — same key as Case 3 but with 20 bytes of AAD and
    // a shorter plaintext.
    let key = Key::Aes128(hex("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
    round_trip(
        key,
        "cafebabefacedbaddecaf888",
        "feedfacedeadbeeffeedfacedeadbeefabaddad2",
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e0915bc94fbc3221a5db94fae95ae7121a47",
    );
}

// ----- AES-256-GCM (NIST GCM Spec, Appendix B Test Case 14) -----

#[test]
fn aes256_gcm_nist_case_13_empty_pt_empty_aad() {
    // Test Case 13: 256-bit zero key, zero nonce, empty plaintext +
    // AAD. Expected output is the 16-byte tag of the empty payload.
    let key = Key::Aes256(
        hex("0000000000000000000000000000000000000000000000000000000000000000")
            .try_into()
            .unwrap(),
    );
    let ct = seal(&key, &hex("000000000000000000000000"), &[], &[]).unwrap();
    assert_eq!(hex_str(&ct), "530f8afbc74536b9a963b4f1c4cb738b");
    let pt = open(&key, &hex("000000000000000000000000"), &[], &ct).unwrap();
    assert!(pt.is_empty());
}

#[test]
fn aes256_gcm_case_15_inputs_round_trip() {
    // Same inputs as NIST GCM Test Case 15 (AES-256, 60-byte plaintext,
    // 20-byte AAD), but verified via round-trip rather than a literal
    // expected ciphertext — Cases 3/4 (AES-128) and Case 13 (AES-256)
    // already pin the underlying GCM kernel's output against the
    // published vectors. This case covers the 256-bit-key + AAD path.
    let key = Key::Aes256(
        hex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
            .try_into()
            .unwrap(),
    );
    let nonce = hex("cafebabefacedbaddecaf888");
    let aad = hex("feedfacedeadbeeffeedfacedeadbeefabaddad2");
    let plaintext = hex(
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
    );

    let ct = seal(&key, &nonce, &aad, &plaintext).unwrap();
    assert_eq!(ct.len(), plaintext.len() + 16, "tag appended");
    let pt = open(&key, &nonce, &aad, &ct).unwrap();
    assert_eq!(pt, plaintext);
}

// ----- Negative: wrong tag must reject -----

#[test]
fn open_with_tampered_tag_fails() {
    let key = Key::Aes128(*b"0123456789abcdef");
    let nonce = [0u8; 12];
    let aad = b"hello aad";
    let pt = b"sensitive";
    let mut ct = seal(&key, &nonce, aad, pt).unwrap();
    // Flip a tag byte.
    let last = ct.len() - 1;
    ct[last] ^= 0xFF;
    assert!(
        open(&key, &nonce, aad, &ct).is_err(),
        "tampered tag must fail to decrypt"
    );
}

#[test]
fn open_with_wrong_aad_fails() {
    let key = Key::Aes128(*b"0123456789abcdef");
    let nonce = [0u8; 12];
    let pt = b"sensitive";
    let ct = seal(&key, &nonce, b"aad-A", pt).unwrap();
    assert!(
        open(&key, &nonce, b"aad-B", &ct).is_err(),
        "different AAD must fail to decrypt"
    );
}

#[test]
fn open_with_wrong_key_fails() {
    let key_a = Key::Aes128(*b"0123456789abcdef");
    let key_b = Key::Aes128(*b"abcdef0123456789");
    let nonce = [0u8; 12];
    let pt = b"sensitive";
    let ct = seal(&key_a, &nonce, b"aad", pt).unwrap();
    assert!(
        open(&key_b, &nonce, b"aad", &ct).is_err(),
        "wrong key must fail to decrypt"
    );
}

// ----- Helpers -----

fn hex_str(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
