//! Error type for the crypto crate. Mirrors the `CodecError` shape
//! used elsewhere — small enum, no thiserror dep.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// The `KeyRetriever` returned `None` for a requested key id.
    KeyNotFound { context: &'static str },
    /// AES-GCM tag mismatch — the ciphertext was tampered with,
    /// the wrong key was provided, or the AAD doesn't match what
    /// the writer used.
    AuthenticationFailed,
    /// The on-disk algorithm descriptor names something we don't
    /// implement yet. v0.6.0 supports `AES_GCM_V1`; `AES_GCM_CTR_V1`
    /// is deferred.
    UnsupportedAlgorithm { name: &'static str },
    /// Nonce was not exactly 12 bytes (GCM IV size).
    MalformedNonce { got: usize },
    /// Ciphertext was shorter than tag+nonce length frame, so the
    /// frame parser couldn't even attempt decrypt.
    ShortCiphertext { got: usize, need: usize },
    /// AES-GCM key was an unexpected width.
    InvalidKeyLength { got: usize },
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound { context } => {
                write!(f, "no key returned by KeyRetriever for {context}")
            }
            Self::AuthenticationFailed => {
                write!(f, "AES-GCM authentication failed (tag mismatch)")
            }
            Self::UnsupportedAlgorithm { name } => {
                write!(f, "unsupported encryption algorithm: {name}")
            }
            Self::MalformedNonce { got } => {
                write!(f, "AES-GCM nonce must be 12 bytes (got {got})")
            }
            Self::ShortCiphertext { got, need } => {
                write!(f, "ciphertext is {got} bytes, need at least {need}")
            }
            Self::InvalidKeyLength { got } => {
                write!(f, "AES key must be 16, 24, or 32 bytes (got {got})")
            }
        }
    }
}

impl std::error::Error for CryptoError {}

pub type Result<T> = std::result::Result<T, CryptoError>;
