use crate::compact::FieldType;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    UnexpectedEof { needed: usize, remaining: usize },
    VarintOverflow,
    InvalidEnumValue { type_name: &'static str, value: i32 },
    InvalidFieldType(u8),
    UnknownStructField {
        struct_name: &'static str,
        field_id: i16,
    },
    MissingRequiredField {
        struct_name: &'static str,
        field_id: i16,
    },
    UnexpectedFieldType {
        struct_name: &'static str,
        field_id: i16,
    },
    UnexpectedListElementType {
        expected: FieldType,
        actual: FieldType,
    },
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
            Self::UnknownStructField { struct_name, field_id } => {
                write!(f, "{struct_name}: unknown field id {field_id}")
            }
            Self::MissingRequiredField { struct_name, field_id } => {
                write!(f, "{struct_name}: required field id {field_id} missing")
            }
            Self::UnexpectedFieldType { struct_name, field_id } => {
                write!(f, "{struct_name}: unexpected wire type for field id {field_id}")
            }
            Self::UnexpectedListElementType { expected, actual } => {
                write!(f, "list element type: expected {expected:?}, got {actual:?}")
            }
        }
    }
}

impl std::error::Error for FormatError {}

pub type Result<T> = std::result::Result<T, FormatError>;
