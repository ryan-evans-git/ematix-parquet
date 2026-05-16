//! Key material + retrieval trait.

use std::collections::HashMap;

use crate::error::{CryptoError, Result};

/// AES-GCM key. The bit-width determines which Aes\*Gcm variant the
/// `aead` module picks at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Aes128([u8; 16]),
    Aes192([u8; 24]),
    Aes256([u8; 32]),
}

impl Key {
    /// Decode raw key bytes into the variant matching their length.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        match bytes.len() {
            16 => {
                let mut k = [0u8; 16];
                k.copy_from_slice(bytes);
                Ok(Self::Aes128(k))
            }
            24 => {
                let mut k = [0u8; 24];
                k.copy_from_slice(bytes);
                Ok(Self::Aes192(k))
            }
            32 => {
                let mut k = [0u8; 32];
                k.copy_from_slice(bytes);
                Ok(Self::Aes256(k))
            }
            n => Err(CryptoError::InvalidKeyLength { got: n }),
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Aes128(k) => k,
            Self::Aes192(k) => k,
            Self::Aes256(k) => k,
        }
    }
}

/// Caller-supplied key retriever. The codec asks for the **footer**
/// key once per file and a per-column key zero-or-once per encrypted
/// column. Real implementations talk to KMS / Vault / local files;
/// `StaticKeys` is a test helper.
///
/// `key_metadata` is the opaque bytes the writer stored in the
/// `ColumnCryptoMetaData.key_metadata` field (or
/// `FileCryptoMetaData.key_metadata` for the footer). Callers
/// interpret it however they like — typical schemes encode a KEK id
/// + wrapped DEK.
pub trait KeyRetriever: Send + Sync {
    fn footer_key(&self, key_metadata: Option<&[u8]>) -> Result<Key>;
    fn column_key(&self, path_in_schema: &[&[u8]], key_metadata: Option<&[u8]>) -> Result<Key>;
}

/// Static in-memory key store. Test/oracle use only.
#[derive(Debug, Default)]
pub struct StaticKeys {
    footer: Option<Key>,
    columns: HashMap<Vec<Vec<u8>>, Key>,
}

impl StaticKeys {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_footer(&mut self, key: Key) -> &mut Self {
        self.footer = Some(key);
        self
    }

    pub fn set_column<I, S>(&mut self, path: I, key: Key) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let path_vec: Vec<Vec<u8>> = path.into_iter().map(|s| s.as_ref().to_vec()).collect();
        self.columns.insert(path_vec, key);
        self
    }
}

impl KeyRetriever for StaticKeys {
    fn footer_key(&self, _key_metadata: Option<&[u8]>) -> Result<Key> {
        self.footer
            .clone()
            .ok_or(CryptoError::KeyNotFound { context: "footer" })
    }

    fn column_key(&self, path_in_schema: &[&[u8]], _key_metadata: Option<&[u8]>) -> Result<Key> {
        let path_vec: Vec<Vec<u8>> = path_in_schema.iter().map(|s| s.to_vec()).collect();
        self.columns
            .get(&path_vec)
            .cloned()
            .ok_or(CryptoError::KeyNotFound { context: "column" })
    }
}
