//! Oracle: ours emits DataPageV2 → parquet-rs reads values back, and
//! ours reads its own V2 files.
//!
//! Π.6b contract: `write_table_to_path_v2` produces files using
//! `PageType::DataPageV2`. For REQUIRED columns (no rep/def levels)
//! the V2 body is identical bytes to V1 — only the page header
//! differs. parquet-rs's V2 dispatch is the cross-check.

use ematix_parquet_codec::write::{write_table_to_path_v2, ColumnData};
use ematix_parquet_format::types::CompressionCodec;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn open_pq(path: &std::path::Path) -> SerializedFileReader<std::fs::File> {
    let f = std::fs::File::open(path).unwrap();
    SerializedFileReader::new(f).unwrap()
}

#[test]
fn v2_i64_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.parquet");

    let values: Vec<i64> = (0i64..1000).map(|i| i * 13 - 100).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_v2(&path, &cols, CompressionCodec::Uncompressed, usize::MAX).unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected Int64");
    };
    let mut out = Vec::with_capacity(1000);
    typed.read_records(1000, None, None, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn v2_byte_array_snappy_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_ba.parquet");

    let values: Vec<&[u8]> = (0..500)
        .map(|i| -> &'static [u8] {
            match i % 5 {
                0 => b"alpha",
                1 => b"bravo",
                2 => b"charlie",
                3 => b"delta",
                _ => b"echo",
            }
        })
        .collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::ByteArray(&values))];
    write_table_to_path_v2(&path, &cols, CompressionCodec::Snappy, usize::MAX).unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected ByteArray");
    };
    let mut out: Vec<PqByteArray> = Vec::with_capacity(500);
    typed.read_records(500, None, None, &mut out).unwrap();
    for (i, got) in out.iter().enumerate() {
        assert_eq!(got.data(), values[i]);
    }
}

#[test]
fn v2_self_round_trip() {
    use ematix_parquet_codec::read::read_column_i64;
    use ematix_parquet_io::ParquetFile;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("self_v2.parquet");

    let values: Vec<i64> = (0i64..600).map(|i| i * 7).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_v2(&path, &cols, CompressionCodec::Snappy, 150).unwrap();

    // Multi-RG V2 file. Walk every RG via our reader.
    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let n_rg = md.row_groups.len();
    assert_eq!(n_rg, 4);
    drop(md);

    let mut got: Vec<i64> = Vec::new();
    for rg_ix in 0..n_rg {
        got.extend(read_column_i64(&file, rg_ix, 0).unwrap());
    }
    assert_eq!(got, values);
}

#[test]
fn v2_page_type_appears_in_metadata() {
    // Sanity: peek at the first page header to confirm we wrote
    // DataPageV2 (and not accidentally DataPage).
    use ematix_parquet_format::compact::Cursor;
    use ematix_parquet_format::metadata::read_page_header;
    use ematix_parquet_format::types::PageType;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_check.parquet");
    let values: Vec<i64> = vec![1, 2, 3, 4];
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_v2(&path, &cols, CompressionCodec::Uncompressed, usize::MAX).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    // Skip PAR1 magic (4 bytes), then the first page header is right there.
    let mut cur = Cursor::new(&bytes[4..]);
    let hdr = read_page_header(&mut cur).unwrap();
    assert_eq!(hdr.page_type, PageType::DataPageV2);
    assert!(hdr.data_page_header_v2.is_some());
    assert!(hdr.data_page_header.is_none());
}
