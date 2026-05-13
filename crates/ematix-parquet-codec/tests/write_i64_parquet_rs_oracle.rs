//! Oracle test: ours writes → parquet-rs reads.
//!
//! Validates that the bytes our writer emits are a valid Parquet
//! file that the mature parquet-rs reader can parse and read back
//! with the same values.

use ematix_parquet_codec::write::write_i64_column_to_path;

use parquet::column::reader::ColumnReader;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn read_back_i64(path: &std::path::Path) -> Vec<i64> {
    let f = std::fs::File::open(path).expect("open written file");
    let r = SerializedFileReader::new(f).expect("parquet-rs reader");
    let rg = r.get_row_group(0).expect("rg 0");
    let cr = rg.get_column_reader(0).expect("col 0");
    let ColumnReader::Int64ColumnReader(mut typed) = cr else {
        panic!("expected INT64 column 0");
    };
    let total = rg.metadata().column(0).num_values() as usize;
    let mut out: Vec<i64> = Vec::with_capacity(total);
    typed
        .read_records(total, None, None, &mut out)
        .expect("read_records");
    out
}

#[test]
fn write_then_read_back_with_parquet_rs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ours.parquet");

    let values: Vec<i64> = (0..1000i64).collect();
    write_i64_column_to_path(&path, "value", &values).expect("write");

    let read_back = read_back_i64(&path);
    assert_eq!(read_back, values);
}

#[test]
fn write_empty_column() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.parquet");

    let values: Vec<i64> = Vec::new();
    write_i64_column_to_path(&path, "value", &values).expect("write");

    let read_back = read_back_i64(&path);
    assert_eq!(read_back, values);
}

#[test]
fn ours_writes_ours_reads_roundtrip() {
    // Symmetric oracle: ours writes → ours reads. Confirms the
    // facade and the writer agree about the on-disk layout.
    use ematix_parquet_codec::read::read_column_i64;
    use ematix_parquet_io::ParquetFile;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("self.parquet");

    let values: Vec<i64> = (0..512i64).map(|i| i * 7 - 100).collect();
    write_i64_column_to_path(&path, "value", &values).expect("write");

    let file = ParquetFile::open(&path).expect("open");
    let read_back = read_column_i64(&file, 0, 0).expect("read");
    assert_eq!(read_back, values);
}

#[test]
fn write_negative_and_extremes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("extremes.parquet");

    let values: Vec<i64> = vec![0, 1, -1, i64::MAX, i64::MIN, 1_000_000_000_000];
    write_i64_column_to_path(&path, "v", &values).expect("write");

    let read_back = read_back_i64(&path);
    assert_eq!(read_back, values);
}
