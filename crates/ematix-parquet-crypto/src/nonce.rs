//! 96-bit nonce generation for AES-GCM.
//!
//! Production writers use `RandomNonceSource` (CSPRNG via `getrandom`).
//! Test code uses `FixedNonceSource` for reproducible vectors.
//!
//! GCM nonces MUST NOT repeat with the same key. A 96-bit random
//! nonce gives ~2^48 messages before collision becomes likely
//! (birthday bound), which is well above any reasonable per-key
//! Parquet workload. Switch to deterministic counters only if the
//! caller can guarantee unique-per-key.

use crate::error::{CryptoError, Result};

pub const NONCE_LEN: usize = 12;

pub trait NonceSource {
    fn next(&mut self) -> Result<[u8; NONCE_LEN]>;
}

/// Generate `N` random bytes via the OS CSPRNG. Used by the codec
/// write path for `aad_file_unique` (16 bytes per spec
/// recommendation) and anywhere else a one-shot random buffer is
/// needed. Caller doesn't need to depend on `getrandom` directly.
pub fn random_bytes<const N: usize>() -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    getrandom::getrandom(&mut buf).map_err(|_| CryptoError::MalformedNonce { got: 0 })?;
    Ok(buf)
}

/// CSPRNG-backed nonce source. Default production choice.
#[derive(Debug, Default)]
pub struct RandomNonceSource;

impl NonceSource for RandomNonceSource {
    fn next(&mut self) -> Result<[u8; NONCE_LEN]> {
        let mut nonce = [0u8; NONCE_LEN];
        // `getrandom` only fails if the OS rng is unavailable, which
        // would indicate something is very wrong. Surface it as a
        // crypto error rather than panic.
        getrandom::getrandom(&mut nonce).map_err(|_| CryptoError::MalformedNonce {
            got: 0, // OS rng failure — we have no nonce to report
        })?;
        Ok(nonce)
    }
}

/// Deterministic counter-based source — tests only. The counter is
/// little-endian into the last 8 bytes; first 4 bytes are caller-
/// supplied prefix.
#[derive(Debug)]
pub struct FixedNonceSource {
    prefix: [u8; 4],
    counter: u64,
}

impl FixedNonceSource {
    pub fn new(prefix: [u8; 4], start: u64) -> Self {
        Self {
            prefix,
            counter: start,
        }
    }
}

impl NonceSource for FixedNonceSource {
    fn next(&mut self) -> Result<[u8; NONCE_LEN]> {
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..4].copy_from_slice(&self.prefix);
        nonce[4..].copy_from_slice(&self.counter.to_le_bytes());
        self.counter = self.counter.wrapping_add(1);
        Ok(nonce)
    }
}
