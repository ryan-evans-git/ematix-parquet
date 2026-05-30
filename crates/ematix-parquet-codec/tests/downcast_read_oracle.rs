//! REV.12 slice 2 oracle: `read_column_i64_downcast` narrows an INT64
//! column to the smallest width its row-group statistics prove safe, and
//! the re-widened values round-trip the originals exactly.
//!
//! End-to-end through the real write → file → decode path (not just the
//! byte primitives), so it exercises stats parsing + the narrow-during-
//! decode orchestrator on a genuine parquet file.

use ematix_parquet_codec::downcast::IntTarget;
use ematix_parquet_codec::read::read_column_i64_downcast;
use ematix_parquet_codec::write::write_i64_column_to_path;
use ematix_parquet_io::ParquetFile;

fn roundtrip(values: &[i64], want: IntTarget) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.parquet");
    write_i64_column_to_path(&path, "v", values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let narrowed = read_column_i64_downcast(&file, 0, 0).unwrap();

    assert_eq!(
        narrowed.target(),
        want,
        "wrong narrowing target for {values:?}"
    );
    assert_eq!(
        narrowed.to_i64(),
        values.to_vec(),
        "values did not round-trip for {values:?}"
    );
    // Footprint actually shrank (unless we kept i64).
    if want != IntTarget::I64 {
        assert!(
            narrowed.byte_size() < values.len() * 8,
            "expected a narrower footprint than i64 for {values:?}"
        );
    }
}

#[test]
fn narrows_to_i8_for_tiny_range() {
    // l_linenumber-shaped (1..7) — narrows 8x.
    roundtrip(&[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], IntTarget::I8);
}

#[test]
fn narrows_to_i16_for_mid_range() {
    roundtrip(&[0, 30_000, -100, 12_345, -32_000], IntTarget::I16);
}

#[test]
fn narrows_to_i32_for_sf100_orderkey_range() {
    // l_orderkey at SF=100 (~600M) fits i32.
    roundtrip(&[1, 600_000_000, 250_000_000, 42, 599_999_999], IntTarget::I32);
}

#[test]
fn narrows_to_u32_when_exceeds_i32_but_nonneg() {
    roundtrip(&[0, 3_000_000_000, 100, 4_000_000_000], IntTarget::U32);
}

#[test]
fn keeps_i64_when_range_needs_full_width() {
    roundtrip(&[0, 10_000_000_000, -5, 9_999_999_999], IntTarget::I64);
}

#[test]
fn larger_column_roundtrips_and_narrows() {
    // 5000 rows, max 500M -> I32; exercises multi-value pages.
    let values: Vec<i64> = (1..=5000).map(|i| i * 100_000).collect();
    roundtrip(&values, IntTarget::I32);
}
