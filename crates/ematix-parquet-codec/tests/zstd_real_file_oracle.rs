//! End-to-end real-file oracle for ZSTD-compressed pages.
//!
//! 1. Write a tiny parquet file via parquet-rs with INT64 + BYTE_ARRAY
//!    columns compressed with ZSTD.
//! 2. Walk pages with our `PageWalker`, decompress each page body via
//!    `decompress_zstd_into`, then decode (PLAIN i64 + dict-encoded
//!    byte arrays) through our pipeline.
//! 3. Read the same file via parquet-rs's typed column reader.
//! 4. Assert value-by-value match.
//!
//! Confirms our ZSTD codec interoperates with a real parquet writer
//! (page-body framing, decompressed-size discovery, in-place reuse).

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::compression::decompress_zstd_into;
use ematix_parquet_codec::dict::decode_rle_dictionary_into;
use ematix_parquet_codec::plain::{decode_plain_byte_array, decode_plain_i64};
use ematix_parquet_format::types::Encoding as EmEncoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::basic::{Compression, Repetition, Type as PhysicalType, ZstdLevel};
use parquet::column::reader::ColumnReader;
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn write_zstd_file(path: &std::path::Path, longs: &[i64], strs: &[&str]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(
                    SchemaType::primitive_type_builder("col_i64", PhysicalType::INT64)
                        .with_repetition(Repetition::REQUIRED)
                        .build()
                        .unwrap(),
                ),
                Arc::new(
                    SchemaType::primitive_type_builder("col_str", PhysicalType::BYTE_ARRAY)
                        .with_repetition(Repetition::REQUIRED)
                        .build()
                        .unwrap(),
                ),
            ])
            .build()
            .unwrap(),
    );

    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .build(),
    );

    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();

    // Column 0: INT64 (PLAIN by default for non-repeating numerics
    // when dict is enabled it'd dict-encode; for our oracle we don't
    // care — we walk encodings dynamically).
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int64ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(longs, None, None).unwrap();
    }
    col.close().unwrap();

    // Column 1: BYTE_ARRAY (dict by default).
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::ByteArrayColumnWriter(ref mut typed) = col.untyped() {
        let bas: Vec<ByteArray> = strs.iter().map(|s| ByteArray::from(*s)).collect();
        typed.write_batch(&bas, None, None).unwrap();
    }
    col.close().unwrap();

    row_group.close().unwrap();
    writer.close().unwrap();
}

fn ours_decode_i64_column(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> Vec<i64> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[rg_idx].columns[col_idx]
        .meta_data
        .as_ref()
        .unwrap();
    let start = cm.dictionary_page_offset.unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let chunk = file.read_range(start, length).unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp: Vec<u8> = Vec::new();
    let mut out: Vec<i64> = Vec::new();
    let mut dict: Option<Vec<i64>> = None;

    while let Some((hdr, body)) = walker.next_page().unwrap() {
        decompress_zstd_into(body, &mut decomp).unwrap();
        if hdr.dictionary_page_header.is_some() {
            // PLAIN-encoded dictionary values.
            dict = Some(decode_plain_i64(&decomp).unwrap());
            continue;
        }
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        let n = dph.num_values as usize;
        match dph.encoding {
            EmEncoding::Plain => {
                out.extend(decode_plain_i64(&decomp).unwrap());
            }
            EmEncoding::RleDictionary | EmEncoding::PlainDictionary => {
                let d = dict.as_ref().expect("dict before data");
                decode_rle_dictionary_into(&decomp, d, n, &mut out).unwrap();
            }
            other => panic!("unexpected i64 encoding {other:?}"),
        }
        if out.len() >= cm.num_values as usize {
            break;
        }
    }
    out
}

fn ours_decode_byte_array_column(
    file: &ParquetFile,
    rg_idx: usize,
    col_idx: usize,
) -> Vec<Vec<u8>> {
    let md = file.metadata().unwrap();
    let cm = md.row_groups[rg_idx].columns[col_idx]
        .meta_data
        .as_ref()
        .unwrap();
    let start = cm.dictionary_page_offset.unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let chunk = file.read_range(start, length).unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp: Vec<u8> = Vec::new();
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut dict: Option<Vec<Vec<u8>>> = None;

    while let Some((hdr, body)) = walker.next_page().unwrap() {
        decompress_zstd_into(body, &mut decomp).unwrap();
        if hdr.dictionary_page_header.is_some() {
            let borrowed = decode_plain_byte_array(&decomp).unwrap();
            dict = Some(borrowed.iter().map(|s| s.to_vec()).collect());
            continue;
        }
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        let n = dph.num_values as usize;
        match dph.encoding {
            EmEncoding::Plain => {
                let borrowed = decode_plain_byte_array(&decomp).unwrap();
                out.extend(borrowed.iter().map(|s| s.to_vec()));
            }
            EmEncoding::RleDictionary | EmEncoding::PlainDictionary => {
                let d = dict.as_ref().expect("dict before data");
                // decode indices, then gather strings.
                let identity: Vec<u32> = (0u32..d.len() as u32).collect();
                let mut idx: Vec<u32> = Vec::with_capacity(n);
                decode_rle_dictionary_into(&decomp, &identity, n, &mut idx).unwrap();
                for i in idx {
                    out.push(d[i as usize].clone());
                }
            }
            other => panic!("unexpected byte_array encoding {other:?}"),
        }
        if out.len() >= cm.num_values as usize {
            break;
        }
    }
    out
}

fn pr_read_i64(path: &std::path::Path) -> Vec<i64> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(0).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(0).unwrap() {
        ColumnReader::Int64ColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pr_read_byte_array(path: &std::path::Path) -> Vec<Vec<u8>> {
    let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let total = r.metadata().row_group(0).column(1).num_values() as usize;
    let rgr = r.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(1).unwrap() {
        ColumnReader::ByteArrayColumnReader(t) => t,
        _ => panic!(),
    };
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out.into_iter().map(|b| b.data().to_vec()).collect()
}

#[test]
fn zstd_real_file_roundtrips_through_our_pipeline() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let longs: Vec<i64> = (0..5_000)
        .map(|i| 1_700_000_000_000_000_000i64 + (i as i64) * 1_000_000)
        .collect();
    // String column with repetition to exercise dict encoding.
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let owned_strs: Vec<String> = (0..5_000).map(|i| words[i % words.len()].to_string()).collect();
    let strs: Vec<&str> = owned_strs.iter().map(|s| s.as_str()).collect();

    write_zstd_file(&path, &longs, &strs);

    let file = ParquetFile::open(&path).expect("open");
    let ours_i64 = ours_decode_i64_column(&file, 0, 0);
    let ours_str = ours_decode_byte_array_column(&file, 0, 1);

    let theirs_i64 = pr_read_i64(&path);
    let theirs_str = pr_read_byte_array(&path);

    assert_eq!(ours_i64.len(), longs.len());
    assert_eq!(ours_i64, longs, "ours i64 vs original");
    assert_eq!(ours_i64, theirs_i64, "ours vs parquet-rs i64 under ZSTD");

    assert_eq!(ours_str.len(), strs.len());
    let expected_str: Vec<Vec<u8>> = strs.iter().map(|s| s.as_bytes().to_vec()).collect();
    assert_eq!(ours_str, expected_str, "ours byte_array vs original");
    assert_eq!(ours_str, theirs_str, "ours vs parquet-rs byte_array under ZSTD");

    eprintln!(
        "PASS: {} i64 + {} byte_array values decoded via our pipeline (ZSTD pages) match parquet-rs",
        ours_i64.len(),
        ours_str.len()
    );
}
