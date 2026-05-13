//! Writers for Parquet metadata structs.
//!
//! Builds the wire bytes that the corresponding `metadata` readers
//! consume. Round-trip is the contract: encode → decode → equal.
//!
//! Covered today (minimum viable write):
//!   - PageHeader / DataPageHeader / DictionaryPageHeader (Π.2a.2)
//!   - FileMetaData → RowGroup → ColumnChunk → ColumnMetaData →
//!     SchemaElement (Π.2a.3)
//!
//! Not yet covered on write (panic with a clear message): KeyValue,
//! Statistics, SizeStatistics, PageEncodingStats, SortingColumn,
//! ColumnOrder, LogicalType payloads, DataPageHeaderV2. Their
//! presence on an input struct will cause the writer to fail loudly
//! — that's intentional, so consumers don't silently drop fields.

use crate::compact::FieldType;
use crate::compact_writer::Writer;
use crate::metadata::{
    ColumnChunk, ColumnMetaData, DataPageHeader, DictionaryPageHeader, FileMetaData, PageHeader,
    RowGroup, SchemaElement,
};

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

/// Encode a `FileMetaData` into compact-protocol wire form.
///
/// Minimum viable scope: version, schema, num_rows, row_groups,
/// optional created_by. KeyValue metadata, ColumnOrders, and any
/// nested optional struct fields (Statistics, SizeStatistics, etc.)
/// will panic — the writer fails loudly so they don't get silently
/// dropped.
pub fn write_file_metadata(md: &FileMetaData<'_>) -> Vec<u8> {
    let mut w = Writer::new();
    encode_file_metadata(&mut w, md);
    w.into_bytes()
}

fn encode_file_metadata(w: &mut Writer, md: &FileMetaData<'_>) {
    // 1: version (i32, required)
    w.write_field_header(1, FieldType::I32, 0);
    w.write_zigzag_i32(md.version);

    // 2: schema (list<SchemaElement>, required)
    w.write_field_header(2, FieldType::List, 1);
    w.write_list_header(md.schema.len(), FieldType::Struct);
    for se in &md.schema {
        encode_schema_element(w, se);
    }

    // 3: num_rows (i64, required)
    w.write_field_header(3, FieldType::I64, 2);
    w.write_zigzag_i64(md.num_rows);

    // 4: row_groups (list<RowGroup>, required)
    w.write_field_header(4, FieldType::List, 3);
    w.write_list_header(md.row_groups.len(), FieldType::Struct);
    for rg in &md.row_groups {
        encode_row_group(w, rg);
    }

    let mut prev: i16 = 4;

    // 5: key_value_metadata — not yet on the write path.
    if md.key_value_metadata.is_some() {
        panic!("FileMetaData.key_value_metadata write not yet implemented");
    }

    // 6: created_by (binary, optional)
    if let Some(cb) = md.created_by {
        w.write_field_header(6, FieldType::Binary, prev);
        w.write_binary(cb);
        prev = 6;
    }

    // 7: column_orders — not yet on the write path.
    if md.column_orders.is_some() {
        panic!("FileMetaData.column_orders write not yet implemented");
    }

    let _ = prev;
    w.write_field_stop();
}

fn encode_schema_element(w: &mut Writer, se: &SchemaElement<'_>) {
    let mut prev: i16 = 0;

    // 1: column_type (i32, optional — only leaves carry it)
    if let Some(ct) = se.column_type {
        w.write_field_header(1, FieldType::I32, prev);
        w.write_zigzag_i32(ct as i32);
        prev = 1;
    }

    // 2: type_length (i32, optional)
    if let Some(tl) = se.type_length {
        w.write_field_header(2, FieldType::I32, prev);
        w.write_zigzag_i32(tl);
        prev = 2;
    }

    // 3: repetition_type (i32, optional)
    if let Some(rt) = se.repetition_type {
        w.write_field_header(3, FieldType::I32, prev);
        w.write_zigzag_i32(rt as i32);
        prev = 3;
    }

    // 4: name (binary, required)
    w.write_field_header(4, FieldType::Binary, prev);
    w.write_binary(se.name);
    prev = 4;

    // 5: num_children (i32, optional — only group nodes carry it)
    if let Some(nc) = se.num_children {
        w.write_field_header(5, FieldType::I32, prev);
        w.write_zigzag_i32(nc);
        prev = 5;
    }

    // 6: converted_type (i32, optional)
    if let Some(ct) = se.converted_type {
        w.write_field_header(6, FieldType::I32, prev);
        w.write_zigzag_i32(ct as i32);
        prev = 6;
    }

    // 7: scale (i32, optional)
    if let Some(s) = se.scale {
        w.write_field_header(7, FieldType::I32, prev);
        w.write_zigzag_i32(s);
        prev = 7;
    }

    // 8: precision (i32, optional)
    if let Some(p) = se.precision {
        w.write_field_header(8, FieldType::I32, prev);
        w.write_zigzag_i32(p);
        prev = 8;
    }

    // 9: field_id (i32, optional)
    if let Some(fid) = se.field_id {
        w.write_field_header(9, FieldType::I32, prev);
        w.write_zigzag_i32(fid);
        prev = 9;
    }

    // 10: logical_type — not yet on the write path.
    if se.logical_type.is_some() {
        panic!("SchemaElement.logical_type write not yet implemented");
    }

    let _ = prev;
    w.write_field_stop();
}

fn encode_row_group(w: &mut Writer, rg: &RowGroup<'_>) {
    // 1: columns (list<ColumnChunk>, required)
    w.write_field_header(1, FieldType::List, 0);
    w.write_list_header(rg.columns.len(), FieldType::Struct);
    for cc in &rg.columns {
        encode_column_chunk(w, cc);
    }

    // 2: total_byte_size (i64, required)
    w.write_field_header(2, FieldType::I64, 1);
    w.write_zigzag_i64(rg.total_byte_size);

    // 3: num_rows (i64, required)
    w.write_field_header(3, FieldType::I64, 2);
    w.write_zigzag_i64(rg.num_rows);

    let mut prev: i16 = 3;

    // 4: sorting_columns — not yet on the write path.
    if rg.sorting_columns.is_some() {
        panic!("RowGroup.sorting_columns write not yet implemented");
    }

    // 5: file_offset (i64, optional)
    if let Some(fo) = rg.file_offset {
        w.write_field_header(5, FieldType::I64, prev);
        w.write_zigzag_i64(fo);
        prev = 5;
    }

    // 6: total_compressed_size (i64, optional)
    if let Some(tcs) = rg.total_compressed_size {
        w.write_field_header(6, FieldType::I64, prev);
        w.write_zigzag_i64(tcs);
        prev = 6;
    }

    // 7: ordinal (i16, optional)
    if let Some(ord) = rg.ordinal {
        w.write_field_header(7, FieldType::I16, prev);
        w.write_zigzag_i16(ord);
        prev = 7;
    }

    let _ = prev;
    w.write_field_stop();
}

fn encode_column_chunk(w: &mut Writer, cc: &ColumnChunk<'_>) {
    let mut prev: i16 = 0;

    // 1: file_path (binary, optional)
    if let Some(fp) = cc.file_path {
        w.write_field_header(1, FieldType::Binary, prev);
        w.write_binary(fp);
        prev = 1;
    }

    // 2: file_offset (i64, required)
    w.write_field_header(2, FieldType::I64, prev);
    w.write_zigzag_i64(cc.file_offset);
    prev = 2;

    // 3: meta_data (struct, optional but normally always present)
    if let Some(ref cm) = cc.meta_data {
        w.write_field_header(3, FieldType::Struct, prev);
        encode_column_metadata(w, cm);
        prev = 3;
    }

    // 4-7: offset/column index pointers (optional)
    if let Some(v) = cc.offset_index_offset {
        w.write_field_header(4, FieldType::I64, prev);
        w.write_zigzag_i64(v);
        prev = 4;
    }
    if let Some(v) = cc.offset_index_length {
        w.write_field_header(5, FieldType::I32, prev);
        w.write_zigzag_i32(v);
        prev = 5;
    }
    if let Some(v) = cc.column_index_offset {
        w.write_field_header(6, FieldType::I64, prev);
        w.write_zigzag_i64(v);
        prev = 6;
    }
    if let Some(v) = cc.column_index_length {
        w.write_field_header(7, FieldType::I32, prev);
        w.write_zigzag_i32(v);
        prev = 7;
    }

    let _ = prev;
    w.write_field_stop();
}

fn encode_column_metadata(w: &mut Writer, cm: &ColumnMetaData<'_>) {
    // 1: column_type (i32, required)
    w.write_field_header(1, FieldType::I32, 0);
    w.write_zigzag_i32(cm.column_type as i32);

    // 2: encodings (list<i32>, required)
    w.write_field_header(2, FieldType::List, 1);
    let enc_values: Vec<i32> = cm.encodings.iter().map(|&e| e as i32).collect();
    w.write_list_i32(&enc_values);

    // 3: path_in_schema (list<binary>, required)
    w.write_field_header(3, FieldType::List, 2);
    w.write_list_binary(&cm.path_in_schema);

    // 4: codec (i32, required)
    w.write_field_header(4, FieldType::I32, 3);
    w.write_zigzag_i32(cm.codec as i32);

    // 5: num_values (i64, required)
    w.write_field_header(5, FieldType::I64, 4);
    w.write_zigzag_i64(cm.num_values);

    // 6: total_uncompressed_size (i64, required)
    w.write_field_header(6, FieldType::I64, 5);
    w.write_zigzag_i64(cm.total_uncompressed_size);

    // 7: total_compressed_size (i64, required)
    w.write_field_header(7, FieldType::I64, 6);
    w.write_zigzag_i64(cm.total_compressed_size);

    let mut prev: i16 = 7;

    // 8: key_value_metadata — not yet on the write path.
    if cm.key_value_metadata.is_some() {
        panic!("ColumnMetaData.key_value_metadata write not yet implemented");
    }

    // 9: data_page_offset (i64, required)
    w.write_field_header(9, FieldType::I64, prev);
    w.write_zigzag_i64(cm.data_page_offset);
    prev = 9;

    // 10: index_page_offset (i64, optional)
    if let Some(v) = cm.index_page_offset {
        w.write_field_header(10, FieldType::I64, prev);
        w.write_zigzag_i64(v);
        prev = 10;
    }

    // 11: dictionary_page_offset (i64, optional)
    if let Some(v) = cm.dictionary_page_offset {
        w.write_field_header(11, FieldType::I64, prev);
        w.write_zigzag_i64(v);
        prev = 11;
    }

    // 12: statistics, 13: encoding_stats, 14/15: bloom filter,
    // 16: size_statistics — all panic; not yet on the write path.
    if cm.statistics.is_some() {
        panic!("ColumnMetaData.statistics write not yet implemented");
    }
    if cm.encoding_stats.is_some() {
        panic!("ColumnMetaData.encoding_stats write not yet implemented");
    }
    if cm.bloom_filter_offset.is_some() || cm.bloom_filter_length.is_some() {
        panic!("ColumnMetaData.bloom_filter_* write not yet implemented");
    }
    if cm.size_statistics.is_some() {
        panic!("ColumnMetaData.size_statistics write not yet implemented");
    }

    let _ = prev;
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

    // ---- FileMetaData / RowGroup / ColumnChunk / ColumnMetaData / SchemaElement ----

    fn minimal_i64_file_metadata<'a>() -> FileMetaData<'a> {
        // The smallest viable FileMetaData: a single-column INT64 file
        // with one row group, one data page, no dictionary, no
        // compression. Mirrors what Π.2a.4 will emit.
        let root = SchemaElement {
            column_type: None,
            type_length: None,
            repetition_type: None,
            name: b"root",
            num_children: Some(1),
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        };
        let leaf = SchemaElement {
            column_type: Some(crate::types::ParquetType::Int64),
            type_length: None,
            repetition_type: Some(crate::types::FieldRepetitionType::Required),
            name: b"value",
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        };
        let cm = ColumnMetaData {
            column_type: crate::types::ParquetType::Int64,
            encodings: vec![Encoding::Plain],
            path_in_schema: vec![b"value" as &[u8]],
            codec: crate::types::CompressionCodec::Uncompressed,
            num_values: 100,
            total_uncompressed_size: 800,
            total_compressed_size: 800,
            key_value_metadata: None,
            data_page_offset: 4, // right after PAR1 magic
            index_page_offset: None,
            dictionary_page_offset: None,
            statistics: None,
            encoding_stats: None,
            bloom_filter_offset: None,
            bloom_filter_length: None,
            size_statistics: None,
        };
        let cc = ColumnChunk {
            file_path: None,
            file_offset: 4,
            meta_data: Some(cm),
            offset_index_offset: None,
            offset_index_length: None,
            column_index_offset: None,
            column_index_length: None,
        };
        let rg = RowGroup {
            columns: vec![cc],
            total_byte_size: 800,
            num_rows: 100,
            sorting_columns: None,
            file_offset: None,
            total_compressed_size: None,
            ordinal: None,
        };
        FileMetaData {
            version: 1,
            schema: vec![root, leaf],
            num_rows: 100,
            row_groups: vec![rg],
            key_value_metadata: None,
            created_by: Some(b"ematix-parquet"),
            column_orders: None,
        }
    }

    #[test]
    fn file_metadata_roundtrip_minimal_i64() {
        let md = minimal_i64_file_metadata();
        let bytes = write_file_metadata(&md);
        let decoded = crate::metadata::read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, md);
    }

    #[test]
    fn file_metadata_roundtrip_no_created_by() {
        let mut md = minimal_i64_file_metadata();
        md.created_by = None;
        let bytes = write_file_metadata(&md);
        let decoded = crate::metadata::read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, md);
    }

    #[test]
    fn file_metadata_roundtrip_with_optional_offsets() {
        // Exercise the optional i64/i32 fields on RowGroup and
        // ColumnChunk (file_offset, total_compressed_size, etc.).
        let mut md = minimal_i64_file_metadata();
        md.row_groups[0].file_offset = Some(1024);
        md.row_groups[0].total_compressed_size = Some(800);
        md.row_groups[0].ordinal = Some(0);
        md.row_groups[0].columns[0].column_index_offset = Some(4096);
        md.row_groups[0].columns[0].column_index_length = Some(64);
        let bytes = write_file_metadata(&md);
        let decoded = crate::metadata::read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, md);
    }

    #[test]
    fn file_metadata_roundtrip_with_dict_offset() {
        let mut md = minimal_i64_file_metadata();
        // Pretend the file actually has a dictionary page before the
        // data page (more realistic shape).
        if let Some(ref mut cm) = md.row_groups[0].columns[0].meta_data {
            cm.dictionary_page_offset = Some(4);
            cm.data_page_offset = 128;
            cm.encodings = vec![Encoding::Plain, Encoding::RleDictionary];
        }
        let bytes = write_file_metadata(&md);
        let decoded = crate::metadata::read_file_metadata(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decoded, md);
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
