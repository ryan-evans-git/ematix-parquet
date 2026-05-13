use ematix_parquet_format::error::FormatError;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    Format(FormatError),
    NotAParquetFile {
        expected: &'static [u8; 4],
        found: [u8; 4],
        position: &'static str,
    },
    TruncatedFooter {
        file_size: u64,
        declared_footer_length: u64,
    },
    OutOfRangeRead {
        offset: u64,
        length: u64,
        file_size: u64,
    },
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Format(e) => write!(f, "format decode error: {e}"),
            Self::NotAParquetFile {
                expected,
                found,
                position,
            } => write!(
                f,
                "missing parquet magic at {position}: expected {expected:?}, found {found:?}"
            ),
            Self::TruncatedFooter {
                file_size,
                declared_footer_length,
            } => write!(
                f,
                "footer length {declared_footer_length} exceeds file size {file_size}"
            ),
            Self::OutOfRangeRead {
                offset,
                length,
                file_size,
            } => write!(
                f,
                "read of {length} bytes at offset {offset} extends past file size {file_size}"
            ),
        }
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<FormatError> for IoError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

pub type Result<T> = std::result::Result<T, IoError>;
