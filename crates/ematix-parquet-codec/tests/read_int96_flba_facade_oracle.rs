//! Oracle: high-level façade dispatches INT96 and FLBA.
//!
//! Π.5b contract: `read_column_int96` and `read_column_flba`
//! collapse the page-walk + decompress + decode boilerplate the
//! same way `read_column_i64` does.

use std::sync::Arc;

use ematix_parquet_codec::plain::Int96 as EmInt96;
use ematix_parquet_codec::read::{read_column_flba, read_column_int96};
use ematix_parquet_io::ParquetFile;

use parquet::data_type::{ByteArray, FixedLenByteArray, Int96 as PqInt96};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn write_int96(path: &std::path::Path, values: &[PqInt96]) {
    let schema = parse_message_type("message s { REQUIRED INT96 ts; }").unwrap();
    // PLAIN-only: parquet-rs would otherwise dict-encode by default,
    // and INT96 dict isn't supported by our facade today.
    let props = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, Arc::new(schema), Arc::new(props)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<parquet::data_type::Int96Type>()
        .write_batch(values, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_flba(path: &std::path::Path, values: &[FixedLenByteArray], type_length: i32) {
    let msg = format!("message s {{ REQUIRED FIXED_LEN_BYTE_ARRAY({type_length}) v; }}");
    let schema = parse_message_type(&msg).unwrap();
    let props = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, Arc::new(schema), Arc::new(props)).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<parquet::data_type::FixedLenByteArrayType>()
        .write_batch(values, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

#[test]
fn int96_facade_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int96.parquet");

    let values: Vec<PqInt96> = (0..32)
        .map(|i| {
            let mut v = PqInt96::new();
            v.set_data((i as u32) * 7, (i as u32) * 13 + 1, (i as u32) * 17 + 2);
            v
        })
        .collect();
    write_int96(&path, &values);

    let file = ParquetFile::open(&path).unwrap();
    let got: Vec<EmInt96> = read_column_int96(&file, 0, 0).unwrap();
    assert_eq!(got.len(), values.len());

    for (i, (g, w)) in got.iter().zip(values.iter()).enumerate() {
        let wd = w.data();
        assert_eq!(g.0, [wd[0], wd[1], wd[2]], "row {i}");
    }
}

#[test]
fn flba_facade_round_trip_uuid_width() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flba.parquet");

    // 16-byte UUIDs.
    let values: Vec<FixedLenByteArray> = (0u8..50)
        .map(|i| {
            let mut buf = [0u8; 16];
            for (j, slot) in buf.iter_mut().enumerate() {
                *slot = i.wrapping_mul(j as u8).wrapping_add(j as u8);
            }
            FixedLenByteArray::from(ByteArray::from(buf.to_vec()))
        })
        .collect();
    write_flba(&path, &values, 16);

    let file = ParquetFile::open(&path).unwrap();
    let got: Vec<Vec<u8>> = read_column_flba(&file, 0, 0).unwrap();
    assert_eq!(got.len(), values.len());
    for (i, (g, w)) in got.iter().zip(values.iter()).enumerate() {
        assert_eq!(g.as_slice(), w.data(), "row {i}");
    }
}

#[test]
fn flba_facade_round_trip_decimal_width() {
    // 5-byte width — common for DECIMAL(10, 2) shapes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flba_5.parquet");

    let values: Vec<FixedLenByteArray> = (0u8..40)
        .map(|i| {
            let buf = [
                i,
                i.wrapping_add(1),
                i.wrapping_add(2),
                i.wrapping_add(3),
                i.wrapping_add(4),
            ];
            FixedLenByteArray::from(ByteArray::from(buf.to_vec()))
        })
        .collect();
    write_flba(&path, &values, 5);

    let file = ParquetFile::open(&path).unwrap();
    let got: Vec<Vec<u8>> = read_column_flba(&file, 0, 0).unwrap();
    assert_eq!(got.len(), values.len());
    for (g, w) in got.iter().zip(values.iter()) {
        assert_eq!(g.as_slice(), w.data());
        assert_eq!(g.len(), 5);
    }
}
