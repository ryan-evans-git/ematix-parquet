use std::fmt;

#[derive(Debug)]
pub enum CodecError {
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
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
        }
    }
}

impl std::error::Error for CodecError {}

pub type Result<T> = std::result::Result<T, CodecError>;
