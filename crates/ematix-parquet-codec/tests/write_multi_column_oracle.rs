//! Oracle test for the Π.2d multi-column write path.
//!
//! Writes a small table (mixed types) and validates every column
//! round-trips through parquet-rs. Confirms the row-group assembly
//! puts each ColumnChunk's `data_page_offset` at the right place and
//! the schema's num_children / leaf list lines up with the actual
//! columns.

use ematix_parquet_codec::write::{write_table_to_path, ColumnData};
use ematix_parquet_format::types::CompressionCodec;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn rg_meta(path: &std::path::Path) -> SerializedFileReader<std::fs::File> {
    let f = std::fs::File::open(path).unwrap();
    SerializedFileReader::new(f).unwrap()
}

fn pq_read_i64_col(r: &SerializedFileReader<std::fs::File>, col_idx: usize) -> Vec<i64> {
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(col_idx).unwrap();
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("col {col_idx} expected INT64");
    };
    let total = rg.metadata().column(col_idx).num_values() as usize;
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_f64_col(r: &SerializedFileReader<std::fs::File>, col_idx: usize) -> Vec<f64> {
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(col_idx).unwrap();
    let ColumnReader::DoubleColumnReader(mut typed) = cr else {
        panic!("col {col_idx} expected DOUBLE");
    };
    let total = rg.metadata().column(col_idx).num_values() as usize;
    let mut out: Vec<f64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_byte_array_col(
    r: &SerializedFileReader<std::fs::File>,
    col_idx: usize,
) -> Vec<Vec<u8>> {
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(col_idx).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("col {col_idx} expected BYTE_ARRAY");
    };
    let total = rg.metadata().column(col_idx).num_values() as usize;
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

#[test]
fn three_column_mixed_type_table_roundtrips_through_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.parquet");

    let ids: Vec<i64> = (0..500i64).collect();
    let prices: Vec<f64> = (0..500).map(|i| i as f64 * 1.25).collect();
    let names: Vec<Vec<u8>> = (0..500)
        .map(|i| format!("row-{:03}", i).into_bytes())
        .collect();
    let name_refs: Vec<&[u8]> = names.iter().map(|v| v.as_slice()).collect();

    let cols: Vec<(&str, ColumnData<'_>)> = vec![
        ("id", ColumnData::I64(&ids)),
        ("price", ColumnData::F64(&prices)),
        ("name", ColumnData::ByteArray(&name_refs)),
    ];
    write_table_to_path(&path, &cols, CompressionCodec::Uncompressed).unwrap();

    let r = rg_meta(&path);
    assert_eq!(r.metadata().num_row_groups(), 1);
    assert_eq!(r.metadata().row_group(0).num_columns(), 3);
    assert_eq!(r.metadata().file_metadata().num_rows(), 500);

    assert_eq!(pq_read_i64_col(&r, 0), ids);
    let read_prices = pq_read_f64_col(&r, 1);
    for (a, b) in read_prices.iter().zip(prices.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(pq_read_byte_array_col(&r, 2), names);
}

#[test]
fn multi_column_with_snappy_codec_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snappy_mixed.parquet");

    let ids: Vec<i64> = (0..2000i64).map(|i| i % 32).collect();
    let names: Vec<Vec<u8>> = (0..2000)
        .map(|i| format!("user-{:04}-tag", i % 8).into_bytes())
        .collect();
    let name_refs: Vec<&[u8]> = names.iter().map(|v| v.as_slice()).collect();

    let cols: Vec<(&str, ColumnData<'_>)> = vec![
        ("user_id", ColumnData::I64(&ids)),
        ("tag", ColumnData::ByteArray(&name_refs)),
    ];
    write_table_to_path(&path, &cols, CompressionCodec::Snappy).unwrap();

    let r = rg_meta(&path);
    assert_eq!(pq_read_i64_col(&r, 0), ids);
    assert_eq!(pq_read_byte_array_col(&r, 1), names);

    // Both columns should report SNAPPY in their per-column metadata.
    for i in 0..2 {
        assert_eq!(
            r.metadata().row_group(0).column(i).compression(),
            parquet::basic::Compression::SNAPPY
        );
    }
}

#[test]
fn empty_table_rejected_with_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.parquet");
    let cols: Vec<(&str, ColumnData<'_>)> = vec![];
    let err = write_table_to_path(&path, &cols, CompressionCodec::Uncompressed).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("at least one column") || msg.to_lowercase().contains("empty"),
        "unexpected error: {msg}"
    );
}

#[test]
fn mismatched_column_lengths_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.parquet");
    let a: Vec<i64> = vec![1, 2, 3];
    let b: Vec<i64> = vec![10, 20]; // shorter
    let cols: Vec<(&str, ColumnData<'_>)> =
        vec![("a", ColumnData::I64(&a)), ("b", ColumnData::I64(&b))];
    let err = write_table_to_path(&path, &cols, CompressionCodec::Uncompressed).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("length") || msg.to_lowercase().contains("row"),
        "unexpected error: {msg}"
    );
}
