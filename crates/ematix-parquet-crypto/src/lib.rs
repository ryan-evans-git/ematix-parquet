//! AES-GCM primitives, AAD construction, and key retrieval for
//! Parquet Modular Encryption.
//!
//! This crate is intentionally tiny — it owns the cryptographic
//! ground truth (AES-GCM seal/open + AAD byte layout per the PME
//! spec) and exposes a `KeyRetriever` trait the application
//! implements. The codec wires it in under
//! `--features encryption`; the rest of the workspace stays
//! crypto-free.
//!
//! ## Layering
//!
//! - `aead` — `seal` / `open` over AES-128/192/256-GCM. Wraps the
//!   `aes-gcm` crate so callers see a Parquet-shaped API.
//! - `aad` — `ModuleType` enum + `build_module_aad` byte assembly
//!   per the spec (Encryption.md → "AAD construction"). Isolates the
//!   tagging quirks so they're testable in one place.
//! - `key` — `KeyRetriever` trait + `Key` enum + `StaticKeys` test
//!   helper. Production callers implement the trait against KMS.
//! - `nonce` — `RandomNonceSource` (writer-only). 96-bit GCM nonce
//!   generation. Tests use the inline fixed-nonce helper.
//! - `error` — `CryptoError` returned by every fallible op.
//!
//! ## Example
//!
//! ```
//! use ematix_parquet_crypto::aead::{open, seal};
//! use ematix_parquet_crypto::key::{Key, StaticKeys};
//!
//! let footer_key = Key::Aes128(*b"0123456789012345");
//! let mut keys = StaticKeys::new();
//! keys.set_footer(footer_key.clone());
//!
//! let plaintext = b"the small one";
//! let aad = b"file_unique||module||0||0";
//! let nonce = [7u8; 12];
//!
//! let ct = seal(&footer_key, &nonce, aad, plaintext).unwrap();
//! let pt = open(&footer_key, &nonce, aad, &ct).unwrap();
//! assert_eq!(pt, plaintext);
//! ```

pub mod aad;
pub mod aead;
pub mod error;
pub mod key;
pub mod nonce;

pub use error::{CryptoError, Result};
