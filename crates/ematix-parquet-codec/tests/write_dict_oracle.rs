//! Oracle: ours writes PLAIN_DICTIONARY → parquet-rs reads values
//! back, and ours reads its own dict files.
//!
//! Π.4c contract: low-cardinality columns can be encoded as a
//! dictionary page (PLAIN-encoded unique values) + a data page of
//! RLE/bit-pack-encoded indices. Round-trip via parquet-rs proves
//! we got the wire shape right; round-trip via our own reader
//! proves the encoder agrees with our existing decoder.

use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path_with_codec,
    write_f64_column_dict_to_path, write_i32_column_dict_to_path,
    write_i64_column_dict_to_path,
};
use ematix_parquet_format::types::CompressionCodec;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn open_pq(path: &std::path::Path) -> SerializedFileReader<std::fs::File> {
    let f = std::fs::File::open(path).expect("open");
    SerializedFileReader::new(f).expect("parquet-rs reader")
}

// ---- byte_array dict round-trip via parquet-rs ----

#[test]
fn byte_array_dict_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_dict.parquet");

    // 5 distinct strings repeated 200 times (1000 rows total) — the
    // canonical dict-friendly shape.
    let palette: [&[u8]; 5] = [b"alpha", b"bravo", b"charlie", b"delta", b"echo"];
    let mut values: Vec<&[u8]> = Vec::with_capacity(1000);
    for i in 0..1000 {
        values.push(palette[i % 5]);
    }

    write_byte_array_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed)
        .unwrap();

    // Read back via parquet-rs.
    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected ByteArray reader");
    };
    let mut out: Vec<PqByteArray> = Vec::with_capacity(1000);
    typed.read_records(1000, None, None, &mut out).unwrap();
    assert_eq!(out.len(), 1000);
    for (i, got) in out.iter().enumerate() {
        assert_eq!(got.data(), palette[i % 5], "row {i}");
    }
}

// ---- byte_array dict round-trip via our own reader ----

#[test]
fn byte_array_dict_self_round_trip() {
    use ematix_parquet_codec::read::read_column_byte_array;
    use ematix_parquet_io::ParquetFile;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_self.parquet");

    let palette: [&[u8]; 3] = [b"red", b"green", b"blue"];
    let values: Vec<&[u8]> = (0..600).map(|i| palette[i % 3]).collect();

    write_byte_array_column_dict_to_path(&path, "color", &values, CompressionCodec::Snappy)
        .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let read_back = read_column_byte_array(&file, 0, 0).unwrap();
    let want: Vec<Vec<u8>> = values.iter().map(|s| s.to_vec()).collect();
    assert_eq!(read_back, want);
}

// ---- i64 dict round-trip ----

#[test]
fn i64_dict_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_dict.parquet");

    let palette: [i64; 4] = [-100, 0, 42, 1000];
    let values: Vec<i64> = (0..800).map(|i| palette[i % 4]).collect();

    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed).unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected Int64 reader");
    };
    let mut out: Vec<i64> = Vec::with_capacity(800);
    typed.read_records(800, None, None, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn i64_dict_self_round_trip() {
    use ematix_parquet_codec::read::read_column_i64;
    use ematix_parquet_io::ParquetFile;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_self.parquet");

    let palette: [i64; 7] = [1, 2, 3, 5, 8, 13, 21];
    let values: Vec<i64> = (0..500).map(|i| palette[i % 7]).collect();

    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let read_back = read_column_i64(&file, 0, 0).unwrap();
    assert_eq!(read_back, values);
}

// ---- i32 dict round-trip ----

#[test]
fn i32_dict_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32_dict.parquet");

    let palette: [i32; 6] = [-50, -1, 0, 1, 50, i32::MAX];
    let values: Vec<i32> = (0..600).map(|i| palette[i % 6]).collect();

    write_i32_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed).unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int32ColumnReader(mut typed) = cr else {
        panic!("expected Int32 reader");
    };
    let mut out: Vec<i32> = Vec::with_capacity(600);
    typed.read_records(600, None, None, &mut out).unwrap();
    assert_eq!(out, values);
}

// ---- f64 dict round-trip ----

#[test]
fn f64_dict_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64_dict.parquet");

    let palette: [f64; 4] = [1.5, -2.0, 3.14, 0.0];
    let values: Vec<f64> = (0..400).map(|i| palette[i % 4]).collect();

    write_f64_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed).unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::DoubleColumnReader(mut typed) = cr else {
        panic!("expected Double reader");
    };
    let mut out: Vec<f64> = Vec::with_capacity(400);
    typed.read_records(400, None, None, &mut out).unwrap();
    assert_eq!(out, values);
}

// ---- column metadata advertises dict encoding correctly ----

#[test]
fn column_metadata_advertises_dict_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meta.parquet");

    let palette: [&[u8]; 3] = [b"a", b"b", b"c"];
    let values: Vec<&[u8]> = (0..30).map(|i| palette[i % 3]).collect();

    write_byte_array_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed)
        .unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cc = rg.metadata().column(0);

    // dictionary_page_offset must be set (this is what enables dict
    // pruning + dict-encoded scans on the read side).
    assert!(
        cc.dictionary_page_offset().is_some(),
        "dictionary_page_offset should be set on a dict-encoded chunk"
    );

    // Encodings list should mention PLAIN_DICTIONARY.
    let encs: Vec<parquet::basic::Encoding> = cc.encodings().collect();
    let has_dict = encs
        .iter()
        .any(|e| matches!(*e, parquet::basic::Encoding::PLAIN_DICTIONARY));
    assert!(
        has_dict,
        "encodings must include PLAIN_DICTIONARY, got {encs:?}"
    );
}

// ---- dict-encoded file is smaller than plain-encoded for the same input ----

#[test]
fn dict_encoded_file_shrinks_for_low_cardinality() {
    let dir = tempfile::tempdir().unwrap();
    let plain_path = dir.path().join("plain.parquet");
    let dict_path = dir.path().join("dict.parquet");

    // 10 distinct ~16-byte strings, 10 000 rows. Plain payload ≈
    // ~160KB; dict encoding should be a fraction of that.
    let palette: [&[u8]; 10] = [
        b"alpha-zero-zero",
        b"bravo-zero-zero",
        b"charlie-zero-z0",
        b"delta-zero-zero",
        b"echo-zero-zerozz",
        b"foxtrot-zero-z0",
        b"golf-zero-zerozz",
        b"hotel-zero-zero",
        b"india-zero-zero",
        b"juliett-zero-zz",
    ];
    let values: Vec<&[u8]> = (0..10_000).map(|i| palette[i % 10]).collect();

    // Same compression on both sides so we're measuring encoding,
    // not codec.
    write_byte_array_column_to_path_with_codec(
        &plain_path,
        "v",
        &values,
        CompressionCodec::Uncompressed,
    )
    .unwrap();
    write_byte_array_column_dict_to_path(&dict_path, "v", &values, CompressionCodec::Uncompressed)
        .unwrap();

    let plain_size = std::fs::metadata(&plain_path).unwrap().len();
    let dict_size = std::fs::metadata(&dict_path).unwrap().len();

    assert!(
        dict_size < plain_size / 2,
        "dict ({dict_size} B) should be < half of plain ({plain_size} B) for 10 distinct values × 10k rows"
    );
}

// ---- single-value column (bit_width == 0) ----

#[test]
fn single_distinct_value_uses_bit_width_zero() {
    // All-same column → dict has 1 entry → bit_width = 0 → indices
    // body is just an RLE run of zeros. Validates the bit_width=0
    // edge case end-to-end.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.parquet");

    let one: &[u8] = b"only";
    let values: Vec<&[u8]> = vec![one; 500];

    write_byte_array_column_dict_to_path(&path, "v", &values, CompressionCodec::Uncompressed)
        .unwrap();

    let r = open_pq(&path);
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!()
    };
    let mut out: Vec<PqByteArray> = Vec::with_capacity(500);
    typed.read_records(500, None, None, &mut out).unwrap();
    assert_eq!(out.len(), 500);
    for got in &out {
        assert_eq!(got.data(), one);
    }
}
