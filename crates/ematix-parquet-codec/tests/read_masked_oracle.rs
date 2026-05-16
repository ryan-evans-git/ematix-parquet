//! Π.10a oracle: `read_column_*_masked_into` parity vs full-decode-
//! then-filter.
//!
//! For every (type × encoding × selectivity) cell, the masked path
//! must produce a `Vec<T>` byte-identical to the materialise-then-
//! filter reference. This is the load-bearing correctness check —
//! if it ever drifts, late-materialization is silently dropping or
//! duplicating values.
//!
//! Mask format: packed bitmap, bit `i` of byte `k` is row `8k + i`,
//! sized `ceil(chunk_num_values / 8)` bytes.

use ematix_parquet_codec::read::{
    build_packed_mask, read_column_f64, read_column_f64_masked_into, read_column_i32,
    read_column_i32_masked_into, read_column_i64, read_column_i64_masked_into,
};
use ematix_parquet_codec::write::{
    write_f64_column_dict_to_path, write_f64_column_to_path, write_i32_column_dict_to_path,
    write_i32_column_to_path, write_i64_column_dict_to_path, write_i64_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

// ============================================================
// helpers
// ============================================================

/// Reference: read the full column, then filter by mask.
fn reference_i64(path: &std::path::Path, mask: &[u8]) -> Vec<i64> {
    let f = ParquetFile::open(path).unwrap();
    let full = read_column_i64(&f, 0, 0).unwrap();
    full.iter()
        .enumerate()
        .filter(|(row, _)| (mask[row / 8] >> (row % 8)) & 1 == 1)
        .map(|(_, v)| *v)
        .collect()
}
fn reference_i32(path: &std::path::Path, mask: &[u8]) -> Vec<i32> {
    let f = ParquetFile::open(path).unwrap();
    let full = read_column_i32(&f, 0, 0).unwrap();
    full.iter()
        .enumerate()
        .filter(|(row, _)| (mask[row / 8] >> (row % 8)) & 1 == 1)
        .map(|(_, v)| *v)
        .collect()
}
fn reference_f64(path: &std::path::Path, mask: &[u8]) -> Vec<f64> {
    let f = ParquetFile::open(path).unwrap();
    let full = read_column_f64(&f, 0, 0).unwrap();
    full.iter()
        .enumerate()
        .filter(|(row, _)| (mask[row / 8] >> (row % 8)) & 1 == 1)
        .map(|(_, v)| *v)
        .collect()
}

/// Build a mask of length `ceil(n/8)` bytes with set bit at
/// every-`stride`-th row. e.g. stride=100 → 1% selectivity.
fn stride_mask(n: usize, stride: usize) -> Vec<u8> {
    build_packed_mask(n, |i| i % stride == 0)
}

fn all_set_mask(n: usize) -> Vec<u8> {
    let mut m = vec![0xFFu8; n.div_ceil(8)];
    // Clear bits past n in the last byte so reference filter agrees.
    let tail = n % 8;
    if tail != 0 {
        let last = m.len() - 1;
        m[last] &= (1u8 << tail) - 1;
    }
    m
}

fn empty_mask(n: usize) -> Vec<u8> {
    vec![0u8; n.div_ceil(8)]
}

// ============================================================
// i64 PLAIN (full sweep)
// ============================================================

#[test]
fn i64_plain_selectivity_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_plain.parquet");
    let values: Vec<i64> = (0..10_000i64).map(|i| i * 17 - 100).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let n = values.len();
    let masks = vec![
        ("0%", empty_mask(n)),
        ("0.1%", stride_mask(n, 1000)),
        ("1%", stride_mask(n, 100)),
        ("10%", stride_mask(n, 10)),
        ("50%", stride_mask(n, 2)),
        ("100%", all_set_mask(n)),
    ];

    let file = ParquetFile::open(&path).unwrap();
    for (label, mask) in masks {
        let want = reference_i64(&path, &mask);
        let mut got = Vec::new();
        read_column_i64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
        assert_eq!(got, want, "i64 PLAIN @ {label}: mask mismatch");
    }
}

// ============================================================
// i64 RLE_DICTIONARY (full sweep)
// ============================================================

#[test]
fn i64_dict_selectivity_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_dict.parquet");
    let palette: [i64; 10] = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110];
    let values: Vec<i64> = (0..10_000).map(|i| palette[i % 10]).collect();
    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let n = values.len();
    let masks = vec![
        ("0%", empty_mask(n)),
        ("0.1%", stride_mask(n, 1000)),
        ("1%", stride_mask(n, 100)),
        ("10%", stride_mask(n, 10)),
        ("50%", stride_mask(n, 2)),
        ("100%", all_set_mask(n)),
    ];

    let file = ParquetFile::open(&path).unwrap();
    for (label, mask) in masks {
        let want = reference_i64(&path, &mask);
        let mut got = Vec::new();
        read_column_i64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
        assert_eq!(got, want, "i64 dict @ {label}: mask mismatch");
    }
}

// ============================================================
// i32 / f64 spot checks
// ============================================================

#[test]
fn i32_plain_and_dict_match_reference() {
    let dir = tempfile::tempdir().unwrap();

    // PLAIN
    let path = dir.path().join("i32_plain.parquet");
    let values: Vec<i32> = (0..5_000i32).collect();
    write_i32_column_to_path(&path, "v", &values).unwrap();
    let mask = stride_mask(values.len(), 7);
    let want = reference_i32(&path, &mask);
    let file = ParquetFile::open(&path).unwrap();
    let mut got = Vec::new();
    read_column_i32_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got, want);

    // DICT
    let path2 = dir.path().join("i32_dict.parquet");
    let palette: [i32; 4] = [-1, 0, 1, 2];
    let vals2: Vec<i32> = (0..3_000).map(|i| palette[i % 4]).collect();
    write_i32_column_dict_to_path(&path2, "v", &vals2, CompressionCodec::Snappy).unwrap();
    let mask2 = stride_mask(vals2.len(), 3);
    let want2 = reference_i32(&path2, &mask2);
    let file2 = ParquetFile::open(&path2).unwrap();
    let mut got2 = Vec::new();
    read_column_i32_masked_into(&file2, 0, 0, &mask2, &mut got2).unwrap();
    assert_eq!(got2, want2);
}

#[test]
fn f64_plain_and_dict_match_reference() {
    let dir = tempfile::tempdir().unwrap();

    // PLAIN
    let path = dir.path().join("f64_plain.parquet");
    let values: Vec<f64> = (0..2_500).map(|i| i as f64 * 0.25 - 100.0).collect();
    write_f64_column_to_path(&path, "v", &values).unwrap();
    let mask = stride_mask(values.len(), 13);
    let want = reference_f64(&path, &mask);
    let file = ParquetFile::open(&path).unwrap();
    let mut got = Vec::new();
    read_column_f64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got, want);

    // DICT
    let path2 = dir.path().join("f64_dict.parquet");
    let palette: [f64; 5] = [1.1, 2.2, 3.3, 4.4, 5.5];
    let vals2: Vec<f64> = (0..2_000).map(|i| palette[i % 5]).collect();
    write_f64_column_dict_to_path(&path2, "v", &vals2, CompressionCodec::Snappy).unwrap();
    let mask2 = stride_mask(vals2.len(), 5);
    let want2 = reference_f64(&path2, &mask2);
    let file2 = ParquetFile::open(&path2).unwrap();
    let mut got2 = Vec::new();
    read_column_f64_masked_into(&file2, 0, 0, &mask2, &mut got2).unwrap();
    assert_eq!(got2, want2);
}

// ============================================================
// Append semantics: out is NOT cleared
// ============================================================

#[test]
fn masked_decode_appends_not_clears() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append.parquet");
    let values: Vec<i64> = (0..100i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let mask = stride_mask(100, 10); // 10 matches: rows 0,10,20,...,90
    let file = ParquetFile::open(&path).unwrap();

    let mut out: Vec<i64> = vec![999, 1000, 1001]; // prefill
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut out).unwrap();

    // Expect prefill + matched values, NOT cleared.
    let mut expected = vec![999i64, 1000, 1001];
    expected.extend((0..100i64).filter(|i| i % 10 == 0));
    assert_eq!(out, expected, "must append, not clear");

    // Second call appends again.
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut out).unwrap();
    expected.extend((0..100i64).filter(|i| i % 10 == 0));
    assert_eq!(out, expected, "second call must also append");
}

// ============================================================
// Multi-page: mask transitions across page edges
// ============================================================

#[test]
fn i64_dict_multi_page_mask_transitions() {
    // Force multiple pages by writing many rows. The default writer
    // emits pages around 8K-20K values; 200K rows guarantees several.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_page.parquet");
    let palette: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let values: Vec<i64> = (0..200_000).map(|i| palette[i % 8]).collect();
    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    // Make sure the mask hits enough variety: every 37th row + a
    // contiguous block in the middle.
    let n = values.len();
    let mut mask = stride_mask(n, 37);
    for row in 100_000..100_100 {
        mask[row / 8] |= 1u8 << (row % 8);
    }

    let want = reference_i64(&path, &mask);
    let file = ParquetFile::open(&path).unwrap();
    let mut got = Vec::new();
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got, want);
}

// ============================================================
// Per-page popcount-skip: fully-dead page handled correctly
// ============================================================

#[test]
fn i64_dict_skip_dead_pages_via_first_half_mask() {
    // Many pages; mask the first half only. Pages whose row range
    // lies entirely past the mask's last set bit get popcount==0
    // and should skip entirely. (Result still matches reference.)
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dead_pages.parquet");
    let palette: [i64; 5] = [10, 20, 30, 40, 50];
    let values: Vec<i64> = (0..150_000).map(|i| palette[i % 5]).collect();
    write_i64_column_dict_to_path(&path, "v", &values, CompressionCodec::Snappy).unwrap();

    let n = values.len();
    let mut mask = empty_mask(n);
    // Set every 10th bit in the first quarter only.
    for row in (0..(n / 4)).step_by(10) {
        mask[row / 8] |= 1u8 << (row % 8);
    }

    let want = reference_i64(&path, &mask);
    let file = ParquetFile::open(&path).unwrap();
    let mut got = Vec::new();
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got, want);
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn mask_too_small_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.parquet");
    let values: Vec<i64> = (0..1_000i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let undersized_mask = vec![0u8; 50]; // need ≥ ceil(1000/8) = 125
    let mut out = Vec::new();
    let r = read_column_i64_masked_into(&file, 0, 0, &undersized_mask, &mut out);
    assert!(r.is_err(), "undersized mask must error");
}

#[test]
fn empty_mask_yields_no_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.parquet");
    let values: Vec<i64> = (0..500i64).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mask = empty_mask(500);
    let mut out = Vec::new();
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn full_mask_recovers_full_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.parquet");
    let values: Vec<i64> = (0..1_000i64).map(|i| i * 3).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mask = all_set_mask(1_000);
    let mut got = Vec::new();
    read_column_i64_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got, values);
}

// ============================================================
// build_packed_mask helper
// ============================================================

#[test]
fn build_packed_mask_sets_bits_per_predicate() {
    let m = build_packed_mask(20, |i| i % 3 == 0);
    // Bits 0,3,6,9,12,15,18 set
    assert_eq!(m.len(), 3); // ceil(20/8)
    for i in 0..20 {
        let bit = (m[i / 8] >> (i % 8)) & 1;
        assert_eq!(bit, if i % 3 == 0 { 1 } else { 0 }, "i={i}");
    }
}

#[test]
fn build_packed_mask_zero_rows() {
    let m = build_packed_mask(0, |_| true);
    assert!(m.is_empty());
}
