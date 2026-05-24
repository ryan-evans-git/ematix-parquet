//! Error type for JSON decode + extension layout violations.

use std::fmt;

/// Anything that can go wrong reading an [`crate::EmatixDataFileExtension`]
/// off an Iceberg `data_file` entry. Kept narrow — write-side errors
/// (which live in the not-yet-written Π.21b iceberg-rust integration)
/// will have their own type.
#[derive(Debug)]
pub enum IcebergIndexError {
    /// JSON payload could not be parsed (truncated, malformed, wrong
    /// shape for a field). Includes a short context string naming
    /// the field that broke.
    Malformed(String),
    /// Extension carries a `version` string other than
    /// [`crate::EMATIX_EXTENSION_VERSION`]. Producers MUST bump the
    /// version on any breaking change; readers refuse unknown
    /// versions rather than guess.
    UnsupportedVersion(String),
}

impl fmt::Display for IcebergIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(msg) => write!(f, "malformed ematix iceberg extension: {msg}"),
            Self::UnsupportedVersion(v) => write!(
                f,
                "ematix iceberg extension version `{v}` not supported (this build expects `{}`)",
                crate::EMATIX_EXTENSION_VERSION
            ),
        }
    }
}

impl std::error::Error for IcebergIndexError {}

/// Convenience alias for fallible operations in this crate.
pub type Result<T> = std::result::Result<T, IcebergIndexError>;
