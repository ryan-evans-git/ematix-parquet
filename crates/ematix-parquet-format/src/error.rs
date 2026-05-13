use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    UnexpectedEof { needed: usize, remaining: usize },
    VarintOverflow,
    InvalidEnumValue { type_name: &'static str, value: i32 },
    InvalidFieldType(u8),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { needed, remaining } => write!(
                f,
                "unexpected EOF: needed {needed} bytes, {remaining} remaining"
            ),
            Self::VarintOverflow => write!(f, "varint exceeded 10 continuation bytes"),
            Self::InvalidEnumValue { type_name, value } => {
                write!(f, "invalid {type_name} discriminant: {value}")
            }
            Self::InvalidFieldType(b) => write!(f, "invalid compact field type nibble: {b:#x}"),
        }
    }
}

impl std::error::Error for FormatError {}

pub type Result<T> = std::result::Result<T, FormatError>;
