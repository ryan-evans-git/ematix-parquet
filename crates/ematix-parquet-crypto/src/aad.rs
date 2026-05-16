//! AAD (Additional Authenticated Data) construction per the PME spec.
//!
//! Spec: https://github.com/apache/parquet-format/blob/master/Encryption.md
//!
//! Each encrypted module is sealed with AES-GCM under an AAD assembled
//! according to the module's type. There are three shapes:
//!
//! - **Footer:** `file_aad || module_byte` (5 bytes beyond file_aad).
//! - **DataPage / DataPageHeader:**
//!   `file_aad || module || rg_ord_LE || col_ord_LE || page_ord_LE`
//!   where every ordinal is `i16` little-endian (2 bytes each).
//! - **Everything else (ColumnMetaData, Dictionary*, ColumnIndex,
//!   OffsetIndex, BloomFilter*):**
//!   `file_aad || module || rg_ord_LE || col_ord_LE`.
//!
//! Note that `DictionaryPage` and `DictionaryPageHeader` do **not**
//! carry a page-ordinal suffix — the spec reserves that for data
//! pages only. parquet-mr / parquet-rs agree.
//!
//! `file_aad` itself is `aad_prefix || aad_file_unique` (in that
//! order); the builder concatenates them for you.

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
    /// Per spec: only `DataPage` and `DataPageHeader` get it.
    pub fn has_page_ordinal(self) -> bool {
        matches!(self, Self::DataPage | Self::DataPageHeader)
    }
}

/// Build the AAD bytes for one module-encrypt operation.
///
/// The `page_ordinal` arg is ignored for module types that don't
/// carry one (Footer / Dictionary* / ColumnMetaData / indexes /
/// bloom). Debug builds assert the caller passes
/// `page_ordinal.is_some()` iff `module.has_page_ordinal()`.
pub fn build_module_aad(
    aad_prefix: Option<&[u8]>,
    aad_file_unique: &[u8],
    module: ModuleType,
    rg_ordinal: i16,
    col_ordinal: i16,
    page_ordinal: Option<i16>,
) -> Vec<u8> {
    debug_assert_eq!(
        module.has_page_ordinal(),
        page_ordinal.is_some(),
        "module {module:?} page_ordinal contract violated",
    );

    let prefix_len = aad_prefix.map_or(0, |p| p.len());

    // Footer is the special case — module byte only, no rg/col/page.
    if matches!(module, ModuleType::Footer) {
        let mut buf = Vec::with_capacity(prefix_len + aad_file_unique.len() + 1);
        if let Some(prefix) = aad_prefix {
            buf.extend_from_slice(prefix);
        }
        buf.extend_from_slice(aad_file_unique);
        buf.push(module as u8);
        return buf;
    }

    let suffix_len = 1 + 2 + 2 + if module.has_page_ordinal() { 2 } else { 0 };
    let mut buf = Vec::with_capacity(prefix_len + aad_file_unique.len() + suffix_len);

    if let Some(prefix) = aad_prefix {
        buf.extend_from_slice(prefix);
    }
    buf.extend_from_slice(aad_file_unique);
    buf.push(module as u8);
    buf.extend_from_slice(&rg_ordinal.to_le_bytes());
    buf.extend_from_slice(&col_ordinal.to_le_bytes());
    if module.has_page_ordinal() {
        if let Some(po) = page_ordinal {
            buf.extend_from_slice(&po.to_le_bytes());
        }
    }
    buf
}
