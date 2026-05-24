//! Integration tests for [`compute_source_fingerprint`] (Π.17a).
//!
//! The fingerprint is the contract between a sidecar `.parquet.idx`
//! and the source `.parquet` it indexes. Two invariants matter:
//!
//! 1. **Stable.** Two computations against the same on-disk file
//!    must produce identical [`SourceFingerprint`] values, byte for
//!    byte. If this breaks, every sidecar built by one process
//!    becomes unreadable by another, and `IndexBuilder` ↔ `ParquetIndex`
//!    handoff breaks.
//!
//! 2. **Sensitive.** Any meaningful rewrite of the source file —
//!    different row count, different schema, different number of row
//!    groups, different page byte offsets — must produce a different
//!    fingerprint. Otherwise a stale sidecar can outlive its source
//!    and silently return rows that no longer exist.
//!
//! These tests use `tempfile` + the codec's own writer to build small
//! reference files. They do not depend on `parquet-rs` or any
//! external corpus.

use ematix_parquet_codec::index::{compute_source_fingerprint, SourceFingerprint};
use ematix_parquet_codec::write::{write_i32_column_to_path, write_i64_column_to_path};
use ematix_parquet_io::ParquetFile;

fn fp_of(path: &std::path::Path) -> SourceFingerprint {
    let file = ParquetFile::open(path).expect("open parquet");
    compute_source_fingerprint(&file).expect("compute fingerprint")
}

#[test]
fn stable_across_repeated_opens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stable.parquet");
    let values: Vec<i64> = (0..1_000i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let a = fp_of(&path);
    let b = fp_of(&path);
    let c = fp_of(&path);

    // Three independent opens of the same on-disk bytes must yield
    // bit-identical fingerprints.
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn captures_row_count_and_rg_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counts.parquet");
    let values: Vec<i64> = (0..1_000i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let fp = fp_of(&path);
    assert_eq!(fp.num_rows, 1_000);
    // `write_i64_column_to_path` writes a single row group by default.
    assert_eq!(fp.num_row_groups, 1);
    // Footer is non-empty.
    assert!(fp.footer_length > 0);
    // CRC32 of a non-empty buffer is never exactly zero (vanishingly
    // unlikely for a real footer — if this fires, the CRC impl is
    // broken).
    assert_ne!(fp.footer_crc32, 0);
}

#[test]
fn differs_when_row_count_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path_a = dir.path().join("a.parquet");
    let path_b = dir.path().join("b.parquet");

    let values_a: Vec<i64> = (0..1_000).collect();
    let values_b: Vec<i64> = (0..1_001).collect(); // one extra row

    write_i64_column_to_path(&path_a, "v", &values_a).unwrap();
    write_i64_column_to_path(&path_b, "v", &values_b).unwrap();

    let fp_a = fp_of(&path_a);
    let fp_b = fp_of(&path_b);

    assert_ne!(fp_a, fp_b);
    assert_eq!(fp_a.num_rows, 1_000);
    assert_eq!(fp_b.num_rows, 1_001);
}

#[test]
fn differs_when_schema_changes() {
    // Same number of rows, different physical type. The footer
    // encodes the schema in the `SchemaElement` list, so a type
    // change must shift the CRC and likely the footer length too.
    let dir = tempfile::tempdir().unwrap();
    let path_i64 = dir.path().join("i64.parquet");
    let path_i32 = dir.path().join("i32.parquet");

    let values_i64: Vec<i64> = (0..500i64).collect();
    let values_i32: Vec<i32> = (0..500i32).collect();

    write_i64_column_to_path(&path_i64, "v", &values_i64).unwrap();
    write_i32_column_to_path(&path_i32, "v", &values_i32).unwrap();

    let fp_i64 = fp_of(&path_i64);
    let fp_i32 = fp_of(&path_i32);

    // num_rows match but the rest of the fingerprint must differ.
    assert_eq!(fp_i64.num_rows, fp_i32.num_rows);
    assert_ne!(fp_i64, fp_i32);
}

#[test]
fn differs_when_data_payload_changes_even_at_same_shape() {
    // Same shape (1000 rows, INT64, one row group) but different
    // values. The footer carries `ColumnMetaData.data_page_offset`
    // and the per-RG byte sizes, so different page bodies → different
    // offsets → different CRC.
    let dir = tempfile::tempdir().unwrap();
    let path_low = dir.path().join("low.parquet");
    let path_hi = dir.path().join("hi.parquet");

    let values_low: Vec<i64> = (0..1_000).collect();
    // Big-magnitude values compress differently, shifting page byte
    // sizes recorded in the column metadata.
    let values_hi: Vec<i64> = (0..1_000).map(|i| i * 1_000_000_000).collect();

    write_i64_column_to_path(&path_low, "v", &values_low).unwrap();
    write_i64_column_to_path(&path_hi, "v", &values_hi).unwrap();

    let fp_low = fp_of(&path_low);
    let fp_hi = fp_of(&path_hi);

    // num_rows and num_row_groups are equal — same shape. But the
    // CRC32 must differ because statistics/offsets are different.
    assert_eq!(fp_low.num_rows, fp_hi.num_rows);
    assert_eq!(fp_low.num_row_groups, fp_hi.num_row_groups);
    assert_ne!(fp_low.footer_crc32, fp_hi.footer_crc32);
}

#[test]
fn footer_length_matches_byte_count() {
    // Sanity: the `footer_length` we record must equal the byte
    // length the file's own trailer declares. If a writer ever
    // pads the footer, this is the canary.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("len.parquet");
    write_i64_column_to_path(&path, "v", &[1, 2, 3]).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let fp = compute_source_fingerprint(&file).unwrap();
    // `ParquetFile` exposes `footer_bytes()` — these have to agree.
    assert_eq!(fp.footer_length as usize, file.footer_bytes().len());
}
