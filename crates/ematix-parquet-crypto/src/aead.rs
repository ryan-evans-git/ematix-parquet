//! AES-GCM seal / open wrapping the `aes-gcm` crate.
//!
//! Surface intentionally Parquet-shaped: nonce + AAD + plaintext in,
//! ciphertext-with-appended-tag out. Branches on key width at runtime
//! so callers don't pick the AES variant per call site.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes128Gcm, Aes256Gcm, Nonce};

use crate::error::{CryptoError, Result};
use crate::key::Key;
use crate::nonce::NONCE_LEN;

/// AES-GCM tag is always 16 bytes (128 bits). `aes-gcm`'s `encrypt`
/// returns `ciphertext || tag` already concatenated.
pub const TAG_LEN: usize = 16;

/// Encrypt `plaintext` with AES-GCM under `key`, authenticating
/// `nonce` and `aad`. Returns `ciphertext || tag` (16 trailing bytes
/// are the tag).
pub fn seal(key: &Key, nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::MalformedNonce { got: nonce.len() });
    }
    let nonce = Nonce::from_slice(nonce);
    let payload = Payload {
        msg: plaintext,
        aad,
    };
    match key {
        Key::Aes128(k) => {
            let cipher = Aes128Gcm::new_from_slice(k).expect("128-bit key");
            cipher
                .encrypt(nonce, payload)
                .map_err(|_| CryptoError::AuthenticationFailed)
        }
        Key::Aes192(_) => {
            // `aes-gcm 0.10` doesn't ship a public `Aes192Gcm`; the
            // PME spec allows 128/192/256, but the practical fleet
            // (parquet-rs, parquet-mr) uses 128 or 256. Surface 192
            // explicitly until a consumer asks.
            Err(CryptoError::UnsupportedAlgorithm {
                name: "AES-192-GCM",
            })
        }
        Key::Aes256(k) => {
            let cipher = Aes256Gcm::new_from_slice(k).expect("256-bit key");
            cipher
                .encrypt(nonce, payload)
                .map_err(|_| CryptoError::AuthenticationFailed)
        }
    }
}

/// Decrypt and authenticate `ciphertext_and_tag` (last 16 bytes are
/// the GCM tag). Returns the plaintext or `AuthenticationFailed` on
/// any tampering / wrong key / wrong AAD.
pub fn open(key: &Key, nonce: &[u8], aad: &[u8], ciphertext_and_tag: &[u8]) -> Result<Vec<u8>> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::MalformedNonce { got: nonce.len() });
    }
    if ciphertext_and_tag.len() < TAG_LEN {
        return Err(CryptoError::ShortCiphertext {
            got: ciphertext_and_tag.len(),
            need: TAG_LEN,
        });
    }
    let nonce = Nonce::from_slice(nonce);
    let payload = Payload {
        msg: ciphertext_and_tag,
        aad,
    };
    match key {
        Key::Aes128(k) => {
            let cipher = Aes128Gcm::new_from_slice(k).expect("128-bit key");
            cipher
                .decrypt(nonce, payload)
                .map_err(|_| CryptoError::AuthenticationFailed)
        }
        Key::Aes192(_) => Err(CryptoError::UnsupportedAlgorithm {
            name: "AES-192-GCM",
        }),
        Key::Aes256(k) => {
            let cipher = Aes256Gcm::new_from_slice(k).expect("256-bit key");
            cipher
                .decrypt(nonce, payload)
                .map_err(|_| CryptoError::AuthenticationFailed)
        }
    }
}
