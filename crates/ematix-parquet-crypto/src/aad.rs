//! AAD (Additional Authenticated Data) construction per the PME spec.
//!
//! Spec: https://github.com/apache/parquet-format/blob/master/Encryption.md
//!
//! Each encrypted module (page, page header, column metadata, …) is
//! sealed with AES-GCM using a module-specific AAD computed as:
//!
//! ```text
//! AAD = aad_prefix? || aad_file_unique || module_type (1 byte)
//!         || row_group_ordinal (2 bytes LE)
//!         || column_ordinal (2 bytes LE)
//!         || page_ordinal? (4 bytes LE)
//! ```
//!
//! The page-ordinal suffix is only present for pages/page-headers —
//! footer, ColumnMetaData, indexes, and bloom filters omit it.

/// Module type byte per the PME spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModuleType {
    Footer = 0,
    ColumnMetaData = 1,
    DataPage = 2,
    DictionaryPage = 3,
    DataPageHeader = 4,
    DictionaryPageHeader = 5,
    ColumnIndex = 6,
    OffsetIndex = 7,
    BloomFilterHeader = 8,
    BloomFilterBitset = 9,
}

impl ModuleType {
    /// `true` if this module's AAD includes a page-ordinal suffix.
    pub fn has_page_ordinal(self) -> bool {
        matches!(
            self,
            Self::DataPage
                | Self::DictionaryPage
                | Self::DataPageHeader
                | Self::DictionaryPageHeader
        )
    }
}

/// Build the AAD bytes for one module-encrypt operation.
///
/// Panics in debug if `module.has_page_ordinal()` mismatches whether
/// `page_ordinal.is_some()` — that's a programmer error in the caller,
/// not user-visible. Release builds silently follow whichever the
/// caller passed.
pub fn build_module_aad(
    aad_prefix: Option<&[u8]>,
    aad_file_unique: &[u8],
    module: ModuleType,
    rg_ordinal: i16,
    col_ordinal: i16,
    page_ordinal: Option<i32>,
) -> Vec<u8> {
    debug_assert_eq!(
        module.has_page_ordinal(),
        page_ordinal.is_some(),
        "module {module:?} page_ordinal contract violated",
    );

    // Capacity: prefix? + file_unique + 1 (module) + 2 (rg) + 2 (col)
    // + 4 (page?). Tight estimate avoids reallocation.
    let prefix_len = aad_prefix.map_or(0, |p| p.len());
    let suffix_len = 1 + 2 + 2 + page_ordinal.map_or(0, |_| 4);
    let mut buf = Vec::with_capacity(prefix_len + aad_file_unique.len() + suffix_len);

    if let Some(prefix) = aad_prefix {
        buf.extend_from_slice(prefix);
    }
    buf.extend_from_slice(aad_file_unique);
    buf.push(module as u8);
    buf.extend_from_slice(&rg_ordinal.to_le_bytes());
    buf.extend_from_slice(&col_ordinal.to_le_bytes());
    if let Some(po) = page_ordinal {
        buf.extend_from_slice(&po.to_le_bytes());
    }
    buf
}
