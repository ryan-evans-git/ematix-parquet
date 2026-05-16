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

/// Decrypt an **encrypted-footer** Parquet file's footer (the
/// ciphertext that lives between `FileCryptoMetaData` and the
/// 4-byte trailing length prefix). Returns the plaintext
/// `FileMetaData` Thrift bytes, ready to feed into
/// `read_file_metadata`.
///
/// The encrypted-footer trailer layout is:
///
/// ```text
/// [ FileCryptoMetaData (Thrift) ][ encrypted FileMetaData ][ len: u32 LE ][ PARE ]
///   ^-- parsed first ----------- ^-- decrypted by this fn
/// ```
///
/// Use `split_encrypted_footer_trailer` to peel off the cleartext
/// `FileCryptoMetaData` slice and the encrypted-FileMetaData slice
/// from a `bytes[..]` window covering the trailer.
pub fn decrypt_footer(
    encrypted_file_metadata: &[u8],
    key: &Key,
    aad_prefix: Option<&[u8]>,
    aad_file_unique: &[u8],
) -> Result<Vec<u8>> {
    // The Footer module's AAD is just `file_aad || module_byte`
    // (no rg/col/page). Build it via the shared helper.
    let aad = build_module_aad(aad_prefix, aad_file_unique, ModuleType::Footer, 0, 0, None);

    // The encrypted FileMetaData is wrapped in the same wire frame
    // every other encrypted module uses:
    //   [ size: u32 LE ][ nonce: 12B ][ ciphertext ][ tag: 16B ]
    if encrypted_file_metadata.len() < SIZE_PREFIX_LEN {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: encrypted_file_metadata.len(),
            need: SIZE_PREFIX_LEN,
        }));
    }
    let size = u32::from_le_bytes(
        encrypted_file_metadata[..SIZE_PREFIX_LEN]
            .try_into()
            .unwrap(),
    ) as usize;
    let total = SIZE_PREFIX_LEN + size;
    if encrypted_file_metadata.len() < total {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: encrypted_file_metadata.len(),
            need: total,
        }));
    }
    if size < NONCE_LEN + TAG_LEN {
        return Err(map_crypto_err(CryptoError::ShortCiphertext {
            got: size,
            need: NONCE_LEN + TAG_LEN,
        }));
    }

    let nonce = &encrypted_file_metadata[SIZE_PREFIX_LEN..SIZE_PREFIX_LEN + NONCE_LEN];
    let ct_and_tag = &encrypted_file_metadata[SIZE_PREFIX_LEN + NONCE_LEN..total];

    open(key, nonce, &aad, ct_and_tag).map_err(map_crypto_err)
}

/// Split the bytes between the `PARE` magic / length prefix and the
/// start of the trailer into `(file_crypto_metadata_bytes,
/// encrypted_file_metadata_bytes)`.
///
/// `trailer` must be the slice `[FileCryptoMetaData || encrypted
/// FileMetaData]` — i.e. the `footer_len` bytes immediately before
/// the trailing `[len: u32 LE][PARE]` (which is what
/// `extract_encrypted_footer_trailer` produces from the full file
/// bytes — see the oracle tests).
///
/// We find the boundary by parsing the `FileCryptoMetaData` Thrift
/// and observing where the cursor stops; everything after is the
/// encrypted `FileMetaData` ciphertext frame.
pub fn split_encrypted_footer_trailer(trailer: &[u8]) -> Result<(&[u8], &[u8])> {
    use ematix_parquet_format::compact::Cursor;
    use ematix_parquet_format::metadata::read_file_crypto_metadata;

    let mut cur = Cursor::new(trailer);
    read_file_crypto_metadata(&mut cur).map_err(|e| {
        CodecError::Decompress(format!("PME footer trailer: bad FileCryptoMetaData: {e:?}"))
    })?;
    let boundary = cur.position();
    Ok((&trailer[..boundary], &trailer[boundary..]))
}
