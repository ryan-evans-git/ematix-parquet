//! Page-body decompression. Each function takes a compressed page
//! body and returns a freshly allocated `Vec<u8>` of decompressed
//! bytes. The buffers stay private to the caller — no internal
//! pool yet (perf work to come).

use crate::error::{CodecError, Result};

/// Snappy raw-format decompression. Parquet uses the framed-less
/// "raw" variant of snappy, not the framing-protocol variant.
pub fn decompress_snappy(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut dec = snap::raw::Decoder::new();
    dec.decompress_vec(compressed)
        .map_err(|e| CodecError::Decompress(format!("snappy: {e}")))
}
