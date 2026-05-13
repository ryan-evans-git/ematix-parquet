//! Writers for Parquet metadata structs.
//!
//! Builds the wire bytes that the corresponding `metadata` readers
//! consume. Round-trip is the contract: encode → decode → equal.
//!
//! This module covers the structs needed for a minimum-viable write
//! path: `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`.
//! `FileMetaData` and the row-group / column-chunk structs land in
//! Π.2a.3.

use crate::compact::FieldType;
use crate::compact_writer::Writer;
use crate::metadata::{DataPageHeader, DictionaryPageHeader, PageHeader};

/// Encode a `PageHeader` into the compact-protocol wire form. Returns
/// the encoded bytes; the caller stamps them into a file ahead of the
/// page body.
///
/// Writes only the fields the reader recognises today: page_type,
/// uncompressed_page_size, compressed_page_size, optional CRC,
/// optional DataPageHeader and DictionaryPageHeader. IndexPageHeader
/// + DataPageHeaderV2 are not part of Π.2a's write path.
pub fn write_page_header(hdr: &PageHeader<'_>) -> Vec<u8> {
    let mut w = Writer::new();
    encode_page_header(&mut w, hdr);
    w.into_bytes()
}

fn encode_page_header(w: &mut Writer, hdr: &PageHeader<'_>) {
    // 1: page_type (required, i32 enum)
    w.write_field_header(1, FieldType::I32, 0);
    w.write_zigzag_i32(hdr.page_type as i32);

    // 2: uncompressed_page_size (required, i32)
    w.write_field_header(2, FieldType::I32, 1);
    w.write_zigzag_i32(hdr.uncompressed_page_size);

    // 3: compressed_page_size (required, i32)
    w.write_field_header(3, FieldType::I32, 2);
    w.write_zigzag_i32(hdr.compressed_page_size);

    let mut prev: i16 = 3;

    // 4: crc (optional, i32)
    if let Some(crc) = hdr.crc {
        w.write_field_header(4, FieldType::I32, prev);
        w.write_zigzag_i32(crc);
        prev = 4;
    }

    // 5: data_page_header (optional, struct)
    if let Some(ref dph) = hdr.data_page_header {
        w.write_field_header(5, FieldType::Struct, prev);
        encode_data_page_header(w, dph);
        prev = 5;
    }

    // 6: index_page_header (optional, struct — empty body)
    if hdr.index_page_header.is_some() {
        w.write_field_header(6, FieldType::Struct, prev);
        w.write_field_stop();
        prev = 6;
    }

    // 7: dictionary_page_header (optional, struct)
    if let Some(ref dictph) = hdr.dictionary_page_header {
        w.write_field_header(7, FieldType::Struct, prev);
        encode_dictionary_page_header(w, dictph);
        prev = 7;
    }

    // 8: data_page_header_v2 — not yet supported on the write side.
    if hdr.data_page_header_v2.is_some() {
        panic!("data_page_header_v2 write not yet implemented (Π.2a focuses on v1)");
    }

    let _ = prev;
    w.write_field_stop();
}

fn encode_data_page_header(w: &mut Writer, dph: &DataPageHeader<'_>) {
    // 1: num_values (i32, required)
    w.write_field_header(1, FieldType::I32, 0);
    w.write_zigzag_i32(dph.num_values);

    // 2: encoding (i32, required)
    w.write_field_header(2, FieldType::I32, 1);
    w.write_zigzag_i32(dph.encoding as i32);

    // 3: definition_level_encoding (i32, required)
    w.write_field_header(3, FieldType::I32, 2);
    w.write_zigzag_i32(dph.definition_level_encoding as i32);

    // 4: repetition_level_encoding (i32, required)
    w.write_field_header(4, FieldType::I32, 3);
    w.write_zigzag_i32(dph.repetition_level_encoding as i32);

    // 5: statistics (struct, optional) — not yet wired on write.
    if dph.statistics.is_some() {
        // Π.2a writes pages without statistics. Stats land alongside
        // the column-chunk metadata writer in Π.2a.3.
        panic!("DataPageHeader.statistics write not yet implemented");
    }

    w.write_field_stop();
}

fn encode_dictionary_page_header(w: &mut Writer, dictph: &DictionaryPageHeader) {
    // 1: num_values (i32, required)
    w.write_field_header(1, FieldType::I32, 0);
    w.write_zigzag_i32(dictph.num_values);

    // 2: encoding (i32, required)
    w.write_field_header(2, FieldType::I32, 1);
    w.write_zigzag_i32(dictph.encoding as i32);

    // 3: is_sorted (bool, optional) — embedded in the field header itself.
    if let Some(sorted) = dictph.is_sorted {
        let bool_type = if sorted {
            FieldType::BoolTrue
        } else {
            FieldType::BoolFalse
        };
        w.write_field_header(3, bool_type, 2);
        // No body byte — the type code carries the value.
    }

    w.write_field_stop();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::Cursor;
    use crate::metadata::read_page_header;
    use crate::types::{Encoding, PageType};

    #[test]
    fn data_page_header_roundtrip_minimal() {
        let hdr = PageHeader {
            page_type: PageType::DataPage,
            uncompressed_page_size: 1024,
            compressed_page_size: 512,
            crc: None,
            data_page_header: Some(DataPageHeader {
                num_values: 1000,
                encoding: Encoding::Plain,
                definition_level_encoding: Encoding::Rle,
                repetition_level_encoding: Encoding::Rle,
                statistics: None,
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let mut cur = Cursor::new(&bytes);
        let decoded = read_page_header(&mut cur).unwrap();
        assert_eq!(decoded, hdr);
        assert_eq!(cur.remaining(), 0, "no trailing bytes");
    }

    #[test]
    fn data_page_header_roundtrip_with_crc() {
        let hdr = PageHeader {
            page_type: PageType::DataPage,
            uncompressed_page_size: 2048,
            compressed_page_size: 1024,
            crc: Some(0xDEAD_BEEFu32 as i32),
            data_page_header: Some(DataPageHeader {
                num_values: 500,
                encoding: Encoding::RleDictionary,
                definition_level_encoding: Encoding::Rle,
                repetition_level_encoding: Encoding::Rle,
                statistics: None,
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let decoded = read_page_header(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn dictionary_page_header_roundtrip() {
        let hdr = PageHeader {
            page_type: PageType::DictionaryPage,
            uncompressed_page_size: 256,
            compressed_page_size: 200,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(DictionaryPageHeader {
                num_values: 32,
                encoding: Encoding::Plain,
                is_sorted: Some(false),
            }),
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let decoded = read_page_header(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn dictionary_page_header_is_sorted_true() {
        // is_sorted=true exercises the BoolTrue field-type code path.
        let hdr = PageHeader {
            page_type: PageType::DictionaryPage,
            uncompressed_page_size: 256,
            compressed_page_size: 200,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(DictionaryPageHeader {
                num_values: 16,
                encoding: Encoding::Plain,
                is_sorted: Some(true),
            }),
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let decoded = read_page_header(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn dictionary_page_header_is_sorted_omitted() {
        // is_sorted: None — field 3 not emitted at all.
        let hdr = PageHeader {
            page_type: PageType::DictionaryPage,
            uncompressed_page_size: 128,
            compressed_page_size: 100,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(DictionaryPageHeader {
                num_values: 8,
                encoding: Encoding::Plain,
                is_sorted: None,
            }),
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let decoded = read_page_header(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn page_header_with_index_page_marker() {
        // IndexPageHeader carries no body fields, just the struct marker.
        let hdr = PageHeader {
            page_type: PageType::IndexPage,
            uncompressed_page_size: 64,
            compressed_page_size: 64,
            crc: None,
            data_page_header: None,
            index_page_header: Some(crate::metadata::IndexPageHeader),
            dictionary_page_header: None,
            data_page_header_v2: None,
        };
        let bytes = write_page_header(&hdr);
        let decoded = read_page_header(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, hdr);
    }
}
