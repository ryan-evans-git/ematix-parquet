//! Unit coverage for page-skip selectors.
//!
//! The selector operates on a hand-constructed `ColumnIndex` whose
//! `min_values`/`max_values` are physical-type encoded byte slices
//! (i32 → 4 LE bytes, i64 → 8 LE bytes). For each page, the predicate
//! is "page [min,max] overlaps user [lo,hi]".

use ematix_parquet_codec::page_index::{select_pages_overlapping_i32, select_pages_overlapping_i64};
use ematix_parquet_format::metadata::ColumnIndex;
use ematix_parquet_format::types::BoundaryOrder;

fn i32_bytes(v: i32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn i64_bytes(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

#[test]
fn i32_selects_only_overlapping_pages() {
    // 4 pages, min/max:
    //   page 0: [  0,  99]
    //   page 1: [100, 199]
    //   page 2: [200, 299]
    //   page 3: [300, 399]
    let mins: Vec<Vec<u8>> = [0i32, 100, 200, 300].iter().map(|v| i32_bytes(*v)).collect();
    let maxs: Vec<Vec<u8>> = [99i32, 199, 299, 399].iter().map(|v| i32_bytes(*v)).collect();
    let idx = ColumnIndex {
        null_pages: vec![false; 4],
        min_values: mins.iter().map(|v| v.as_slice()).collect(),
        max_values: maxs.iter().map(|v| v.as_slice()).collect(),
        boundary_order: BoundaryOrder::Ascending,
        null_counts: None,
    };

    // Query [150, 250]: overlaps page 1 + page 2 only.
    let keep = select_pages_overlapping_i32(&idx, 150, 250).unwrap();
    assert_eq!(keep, vec![false, true, true, false]);

    // Query [-50, 50]: only page 0.
    let keep = select_pages_overlapping_i32(&idx, -50, 50).unwrap();
    assert_eq!(keep, vec![true, false, false, false]);

    // Query [1000, 2000]: no pages.
    let keep = select_pages_overlapping_i32(&idx, 1000, 2000).unwrap();
    assert_eq!(keep, vec![false; 4]);

    // Query covering everything.
    let keep = select_pages_overlapping_i32(&idx, i32::MIN, i32::MAX).unwrap();
    assert_eq!(keep, vec![true; 4]);
}

#[test]
fn i32_null_pages_never_selected() {
    // Even if [min,max] overlaps, a null-page contributes no rows
    // matching a non-null predicate. min/max are placeholder bytes.
    let mins: Vec<Vec<u8>> = vec![i32_bytes(0), i32_bytes(0)];
    let maxs: Vec<Vec<u8>> = vec![i32_bytes(0), i32_bytes(0)];
    let idx = ColumnIndex {
        null_pages: vec![true, false],
        min_values: mins.iter().map(|v| v.as_slice()).collect(),
        max_values: maxs.iter().map(|v| v.as_slice()).collect(),
        boundary_order: BoundaryOrder::Unordered,
        null_counts: None,
    };
    let keep = select_pages_overlapping_i32(&idx, -10, 10).unwrap();
    assert_eq!(keep, vec![false, true]);
}

#[test]
fn i32_boundary_edges_are_inclusive() {
    let mins: Vec<Vec<u8>> = vec![i32_bytes(100)];
    let maxs: Vec<Vec<u8>> = vec![i32_bytes(200)];
    let idx = ColumnIndex {
        null_pages: vec![false],
        min_values: mins.iter().map(|v| v.as_slice()).collect(),
        max_values: maxs.iter().map(|v| v.as_slice()).collect(),
        boundary_order: BoundaryOrder::Unordered,
        null_counts: None,
    };
    // lo touches max
    assert_eq!(select_pages_overlapping_i32(&idx, 200, 999).unwrap(), vec![true]);
    // hi touches min
    assert_eq!(select_pages_overlapping_i32(&idx, -10, 100).unwrap(), vec![true]);
    // strictly above
    assert_eq!(select_pages_overlapping_i32(&idx, 201, 999).unwrap(), vec![false]);
    // strictly below
    assert_eq!(select_pages_overlapping_i32(&idx, -10, 99).unwrap(), vec![false]);
}

#[test]
fn i64_selects_only_overlapping_pages() {
    let base = 1_700_000_000_000_000_000i64;
    let mins: Vec<Vec<u8>> = (0..5).map(|i| i64_bytes(base + i * 1_000_000)).collect();
    let maxs: Vec<Vec<u8>> = (0..5)
        .map(|i| i64_bytes(base + i * 1_000_000 + 999_999))
        .collect();
    let idx = ColumnIndex {
        null_pages: vec![false; 5],
        min_values: mins.iter().map(|v| v.as_slice()).collect(),
        max_values: maxs.iter().map(|v| v.as_slice()).collect(),
        boundary_order: BoundaryOrder::Ascending,
        null_counts: None,
    };
    // overlap pages 2..=3
    let keep = select_pages_overlapping_i64(&idx, base + 2_500_000, base + 3_500_000).unwrap();
    assert_eq!(keep, vec![false, false, true, true, false]);
}

#[test]
fn i32_wrong_byte_width_errors() {
    let mins: Vec<Vec<u8>> = vec![vec![0u8; 7]];
    let maxs: Vec<Vec<u8>> = vec![vec![0u8; 7]];
    let idx = ColumnIndex {
        null_pages: vec![false],
        min_values: mins.iter().map(|v| v.as_slice()).collect(),
        max_values: maxs.iter().map(|v| v.as_slice()).collect(),
        boundary_order: BoundaryOrder::Unordered,
        null_counts: None,
    };
    assert!(select_pages_overlapping_i32(&idx, 0, 0).is_err());
}
