//! Error type for the async crate. Wraps `object_store::Error` +
//! format-layer parse errors + the few async-specific shapes (e.g.
//! malformed Parquet trailer).

use std::fmt;

pub type Result<T> = std::result::Result<T, AsyncError>;

#[derive(Debug)]
pub enum AsyncError {
    ObjectStore(object_store::Error),
    Format(String),
    /// File is too small to contain even a footer trailer.
    TruncatedFooter {
        file_size: u64,
        declared_footer_length: u64,
    },
    /// Trailing magic isn't `PAR1`.
    NotAParquetFile {
        expected: &'static [u8; 4],
        found: [u8; 4],
        position: &'static str,
    },
    /// Range read past EOF.
    OutOfRangeRead {
        offset: u64,
        length: u64,
        file_size: u64,
    },
    /// Open / read returned an empty / short response.
    EmptyResponse,
}

impl fmt::Display for AsyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsyncError::ObjectStore(e) => write!(f, "object_store: {e}"),
            AsyncError::Format(s) => write!(f, "format: {s}"),
            AsyncError::TruncatedFooter {
                file_size,
                declared_footer_length,
            } => write!(
                f,
                "truncated footer: file_size={file_size}, declared footer_length={declared_footer_length}"
            ),
            AsyncError::NotAParquetFile {
                expected,
                found,
                position,
            } => write!(
                f,
                "not a parquet file at {position}: expected {:?}, found {:?}",
                expected, found
            ),
            AsyncError::OutOfRangeRead {
                offset,
                length,
                file_size,
            } => write!(
                f,
                "out-of-range read: offset={offset}, length={length}, file_size={file_size}"
            ),
            AsyncError::EmptyResponse => write!(f, "empty response from object store"),
        }
    }
}

impl std::error::Error for AsyncError {}

impl From<object_store::Error> for AsyncError {
    fn from(e: object_store::Error) -> Self {
        AsyncError::ObjectStore(e)
    }
}

impl From<ematix_parquet_format::error::FormatError> for AsyncError {
    fn from(e: ematix_parquet_format::error::FormatError) -> Self {
        AsyncError::Format(format!("{e}"))
    }
}
