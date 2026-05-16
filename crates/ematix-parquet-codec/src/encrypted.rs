//! Encrypted-page wire decoder for Parquet Modular Encryption (PME).
//!
//! This module is `#[cfg(feature = "encryption")]`-gated; default
//! builds never see it. Π.13c only wires the **primitive** —
//! `decrypt_module` decrypts one on-disk encrypted module (page
//! header, page body, footer, etc.) given the right key + AAD.
//! Full read-façade integration (`read_column_*_encrypted`) lands
//! together with encrypted-footer mode in Π.13d.
//!
//! # On-disk module wire format
//!
//! Per `apache/parquet-format/Encryption.md` ("Encrypted modules"),
//! each encrypted module is laid out as:
//!
//! ```text
//! +---------------------+---------+----------------+---------+
//! | length (4 bytes LE) | nonce   | ciphertext     | tag     |
//! |                     | (12 B)  | (variable)     | (16 B)  |
//! +---------------------+---------+----------------+---------+
//! ```
//!
//! `length` counts the nonce + ciphertext + tag (i.e. everything
//! that follows it), so the total bytes on disk = `4 + length`.

use ematix_parquet_crypto::aad::{build_module_aad, ModuleType};
use ematix_parquet_crypto::aead::{open, TAG_LEN};
use ematix_parquet_crypto::key::Key;
use ematix_parquet_crypto::nonce::NONCE_LEN;
use ematix_parquet_crypto::CryptoError;

use crate::error::{CodecError, Result};

/// 4-byte LE size prefix on every encrypted module.
const SIZE_PREFIX_LEN: usize = 4;

/// Per-column context needed to decrypt every page in that column
/// chunk. Built once at chunk-open time from the file's encryption
/// metadata + the resolved key.
#[derive(Debug)]
pub struct ColumnDecryptContext<'a> {
    pub key: Key,
    pub aad_prefix: Option<&'a [u8]>,
    pub aad_file_unique: &'a [u8],
    pub rg_ordinal: i16,
    pub col_ordinal: i16,
}

/// Decrypt one on-disk encrypted module. Returns the plaintext
/// bytes; caller continues with the existing Thrift / decompress /
/// decode path.
///
/// `bytes` must start at the 4-byte size prefix. Returns
/// `(plaintext, total_bytes_consumed)` so the caller can advance
/// past this module.
pub fn decrypt_module(
    bytes: &[u8],
    ctx: &ColumnDecryptContext<'_>,
    module: ModuleType,
    page_ordinal: Option<i16>,
) -> Result<(Vec<u8>, usize)> {
    if bytes.len() < SIZE_PREFIX_LEN {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: bytes.len(),
            need: SIZE_PREFIX_LEN,
        }));
    }
    let size = u32::from_le_bytes(bytes[..SIZE_PREFIX_LEN].try_into().unwrap()) as usize;
    let total = SIZE_PREFIX_LEN + size;
    if bytes.len() < total {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: bytes.len(),
            need: total,
        }));
    }
    if size < NONCE_LEN + TAG_LEN {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: size,
            need: NONCE_LEN + TAG_LEN,
        }));
    }

    let nonce = &bytes[SIZE_PREFIX_LEN..SIZE_PREFIX_LEN + NONCE_LEN];
    let ct_and_tag = &bytes[SIZE_PREFIX_LEN + NONCE_LEN..total];

    let aad = build_module_aad(
        ctx.aad_prefix,
        ctx.aad_file_unique,
        module,
        ctx.rg_ordinal,
        ctx.col_ordinal,
        page_ordinal,
    );

    let plaintext = open(&ctx.key, nonce, &aad, ct_and_tag).map_err(map_crypto_err)?;
    Ok((plaintext, total))
}

fn map_crypto_err(err: CryptoError) -> CodecError {
    // Surface authentication failures with a distinct message so
    // they stand out from generic decode errors in panics + logs.
    CodecError::Decompress(format!("PME decrypt failed: {err}"))
}
