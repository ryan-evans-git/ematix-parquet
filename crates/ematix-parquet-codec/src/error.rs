use ematix_parquet_format::error::FormatError;
use std::fmt;

#[derive(Debug)]
pub enum CodecError {
    /// Wire-format error from the format crate (Cursor reads, varints).
    Wire(FormatError),
    /// Snappy / Zstd / Gzip / etc. surfaced a decompression failure.
    Decompress(String),
    /// A PLAIN-encoded byte stream had a partial value at the end —
    /// the buffer length is not a multiple of the fixed value width.
    UnalignedPlainBuffer {
        value_width: usize,
        buffer_len: usize,
    },
    /// Caller asked for `n` values but the buffer holds fewer than
    /// `n * value_width` bytes.
    UnderflowingPlainBuffer {
        value_width: usize,
        buffer_len: usize,
        requested_values: usize,
    },
    /// RLE/bit-packed primitive: caller passed a `bit_width` outside
    /// the supported [0, 64] range.
    BitWidthOutOfRange(u8),
    /// A dictionary-encoded data page had a body of zero bytes, so
    /// the leading bit-width byte was missing.
    EmptyDictPageBody,
    /// A dictionary index referenced a slot beyond the dict.
    DictIndexOutOfRange { index: u32, dict_size: usize },
    /// A required input was malformed in a way the lower-level decoder
    /// errors don't already cover (e.g. row group index out of range,
    /// page-header missing, file metadata absent).
    InvalidInput(String),
    /// The façade hit a code path that isn't yet wired (e.g. an
    /// encoding or compression codec the dispatcher doesn't handle).
    /// The lower-level decoder for it may still exist; this just
    /// means the high-level entry point can't reach it yet.
    Unsupported(String),
    /// A parallel runner saw its `CancellationToken` fire before this
    /// target had been decoded. Cooperative — the runner only checks
    /// at target boundaries, so in-flight decodes complete before
    /// this surfaces.
    Cancelled,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wire(e) => write!(f, "wire-format error: {e}"),
            Self::Decompress(s) => write!(f, "decompression failed: {s}"),
            Self::UnalignedPlainBuffer {
                value_width,
                buffer_len,
            } => write!(
                f,
                "PLAIN buffer length {buffer_len} not a multiple of {value_width}"
            ),
            Self::UnderflowingPlainBuffer {
                value_width,
                buffer_len,
                requested_values,
            } => write!(
                f,
                "PLAIN buffer length {buffer_len} too short for {requested_values} \
                 values of width {value_width}"
            ),
            Self::BitWidthOutOfRange(w) => write!(f, "bit_width {w} outside supported [0, 64]"),
            Self::EmptyDictPageBody => write!(f, "dictionary-encoded data page had an empty body"),
            Self::DictIndexOutOfRange { index, dict_size } => {
                write!(f, "dictionary index {index} ≥ dict size {dict_size}")
            }
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Cancelled => write!(f, "parallel decode cancelled by token"),
        }
    }
}

impl From<FormatError> for CodecError {
    fn from(e: FormatError) -> Self {
        Self::Wire(e)
    }
}

impl std::error::Error for CodecError {}

pub type Result<T> = std::result::Result<T, CodecError>;
