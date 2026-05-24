//! Error type for JSON decode, extension layout violations, and
//! (feature `iceberg`) async I/O against `iceberg-rust` manifests.

use std::fmt;

/// Anything that can go wrong reading an [`crate::EmatixDataFileExtension`]
/// off an Iceberg `data_file` entry, or (under the `iceberg` feature)
/// walking the table's manifests.
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
    /// An underlying `iceberg-rust` operation (manifest list / manifest
    /// load) failed. Only produced under the `iceberg` feature.
    #[cfg(feature = "iceberg")]
    Iceberg(iceberg::Error),
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
            #[cfg(feature = "iceberg")]
            Self::Iceberg(e) => write!(f, "iceberg-rust error: {e}"),
        }
    }
}

impl std::error::Error for IcebergIndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "iceberg")]
            Self::Iceberg(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "iceberg")]
impl From<iceberg::Error> for IcebergIndexError {
    fn from(e: iceberg::Error) -> Self {
        Self::Iceberg(e)
    }
}

/// Convenience alias for fallible operations in this crate.
pub type Result<T> = std::result::Result<T, IcebergIndexError>;
