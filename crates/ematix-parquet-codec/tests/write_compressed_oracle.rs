//! Oracle tests for the Π.2c compressed write path.
//!
//! Validates Snappy- and Zstd-compressed column writes against the
//! parquet-rs reader. The tests use enough rows that the body is
//! actually compressible (Snappy/Zstd on tiny inputs can produce
//! output larger than input, which exposes any "compressed_size <
//! uncompressed_size" assumptions in the writer — there are none,
//! but the tests still want a real compression ratio so they catch
//! header / metadata bugs that small inputs would hide).

use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_codec::write::{
    write_byte_array_column_to_path_with_codec, write_i64_column_to_path_with_codec,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn pq_read_i64(path: &std::path::Path) -> Vec<i64> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected INT64");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out
}

fn pq_read_byte_array(path: &std::path::Path) -> Vec<Vec<u8>> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected BYTE_ARRAY");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<ByteArray> = Vec::with_capacity(total);
    typed.read_records(total, None, None, &mut out).unwrap();
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

fn pq_codec(path: &std::path::Path) -> parquet::basic::Compression {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    r.metadata().row_group(0).column(0).compression()
}

// ---- Snappy --------------------------------------------------------

#[test]
fn snappy_i64_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snappy.parquet");
    // 10k repeated-pattern rows so Snappy actually compresses.
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 16).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    // Sanity: parquet-rs sees the codec field set to SNAPPY.
    assert_eq!(pq_codec(&path), parquet::basic::Compression::SNAPPY);
    // And the values round-trip.
    assert_eq!(pq_read_i64(&path), values);
}

#[test]
fn snappy_i64_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snappy_self.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| i * 3 - 100).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i64(&file, 0, 0).unwrap(), values);
}

#[test]
fn snappy_byte_array_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snappy_ba.parquet");
    // Repeating strings → high snappy compression ratio.
    let values: Vec<Vec<u8>> = (0..2_000)
        .map(|i| format!("row-{:04}-very-compressible-payload", i % 8).into_bytes())
        .collect();
    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path_with_codec(&path, "v", &refs, CompressionCodec::Snappy)
        .unwrap();
    assert_eq!(pq_codec(&path), parquet::basic::Compression::SNAPPY);
    assert_eq!(pq_read_byte_array(&path), values);
}

// ---- Zstd ----------------------------------------------------------

#[test]
fn zstd_i64_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 32).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Zstd).unwrap();

    // parquet-rs's Compression::ZSTD enum carries a `ZstdLevel`
    // payload; match on the discriminant rather than equality.
    matches!(pq_codec(&path), parquet::basic::Compression::ZSTD(_));
    assert_eq!(pq_read_i64(&path), values);
}

#[test]
fn zstd_i64_reads_back_via_ours() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_self.parquet");
    let values: Vec<i64> = (0..5_000i64).map(|i| i.pow(2)).collect();
    write_i64_column_to_path_with_codec(&path, "v", &values, CompressionCodec::Zstd).unwrap();
    let file = ParquetFile::open(&path).unwrap();
    assert_eq!(read_column_i64(&file, 0, 0).unwrap(), values);
}

#[test]
fn zstd_byte_array_reads_back_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_ba.parquet");
    let values: Vec<Vec<u8>> = (0..1_500)
        .map(|i| format!("Lorem ipsum dolor sit amet {}", i % 16).into_bytes())
        .collect();
    let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path_with_codec(&path, "v", &refs, CompressionCodec::Zstd).unwrap();
    assert_eq!(pq_read_byte_array(&path), values);
}

// ---- Compression actually shrinks ----------------------------------

#[test]
fn snappy_actually_compresses_repeating_pattern() {
    // Compare on-disk size of Uncompressed vs Snappy for a highly
    // compressible payload. This catches a class of bugs where the
    // codec field is set correctly but the body is left uncompressed
    // (or vice versa) — both would still round-trip via the reader's
    // codec dispatch, so the round-trip oracle alone wouldn't notice.
    let dir = tempfile::tempdir().unwrap();
    let uncompressed = dir.path().join("u.parquet");
    let snappy = dir.path().join("s.parquet");
    let values: Vec<i64> = vec![42i64; 50_000];
    write_i64_column_to_path_with_codec(
        &uncompressed,
        "v",
        &values,
        CompressionCodec::Uncompressed,
    )
    .unwrap();
    write_i64_column_to_path_with_codec(&snappy, "v", &values, CompressionCodec::Snappy).unwrap();

    let u_size = std::fs::metadata(&uncompressed).unwrap().len();
    let s_size = std::fs::metadata(&snappy).unwrap().len();
    assert!(
        s_size < u_size / 4,
        "snappy {s_size} should be much smaller than uncompressed {u_size}"
    );
}
