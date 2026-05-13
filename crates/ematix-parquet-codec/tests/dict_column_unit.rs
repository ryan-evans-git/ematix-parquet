//! Unit-level coverage for `DictColumnChunk`: hand-build small
//! columns and check filter/collect/count_matching/gather semantics
//! against trivial reference implementations.

use std::sync::Arc;

use ematix_parquet_codec::column::{DictColumnChunk, Segment};

fn dict_chunk_i32(
    dict: Vec<i32>,
    segments: Vec<Segment<i32>>,
    n: usize,
) -> DictColumnChunk<i32> {
    DictColumnChunk::new(Some(Arc::new(dict)), segments, n)
}

#[test]
fn collect_all_dict_segments() {
    let chunk = dict_chunk_i32(
        vec![10, 20, 30, 40],
        vec![
            Segment::DictIndices(vec![0, 1, 2]),
            Segment::DictIndices(vec![3, 0, 1]),
        ],
        6,
    );
    assert_eq!(chunk.collect(), vec![10, 20, 30, 40, 10, 20]);
}

#[test]
fn collect_mixed_dict_and_plain() {
    // First two pages dict-encoded, third page is PLAIN (writer
    // fallback). The PLAIN page can carry values NOT in the dict.
    let chunk = dict_chunk_i32(
        vec![100, 200],
        vec![
            Segment::DictIndices(vec![0, 1, 0]),
            Segment::Plain(vec![999, 1000, 1001]),
        ],
        6,
    );
    assert_eq!(chunk.collect(), vec![100, 200, 100, 999, 1000, 1001]);
}

#[test]
fn filter_evaluates_predicate_against_dict_then_indexes() {
    let chunk = dict_chunk_i32(
        vec![10, 20, 30, 40, 50],
        vec![Segment::DictIndices(vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4])],
        10,
    );
    // Predicate: value >= 30
    let mask = chunk.filter(|v| v >= 30);
    assert_eq!(
        mask,
        vec![false, false, true, true, true, false, false, true, true, true]
    );
}

#[test]
fn filter_plain_segment_uses_per_value_predicate() {
    let chunk = dict_chunk_i32(
        vec![10],
        vec![
            Segment::DictIndices(vec![0]),
            Segment::Plain(vec![5, 100, 15]),
        ],
        4,
    );
    let mask = chunk.filter(|v| v > 9);
    assert_eq!(mask, vec![true, false, true, true]);
}

#[test]
fn count_matching_equals_filter_then_count_trues() {
    let chunk = dict_chunk_i32(
        vec![1, 2, 3, 4, 5],
        vec![
            Segment::DictIndices(vec![0, 1, 2, 3, 4, 0, 1, 2]),
            Segment::Plain(vec![100, 200, 300]),
        ],
        11,
    );
    let pred = |v: i32| v > 2;
    let mask = chunk.filter(pred);
    let by_filter: usize = mask.iter().filter(|&&b| b).count();
    assert_eq!(chunk.count_matching(pred), by_filter);
    // Direct: dict matches are 3, 4, 5; indices touching those are
    // [2, 3, 4, 2] → 4 from dict segment. Plus all 3 plain values
    // > 2. Total = 7.
    assert_eq!(chunk.count_matching(pred), 7);
}

#[test]
fn gather_keeps_only_masked_rows() {
    let chunk = dict_chunk_i32(
        vec![10, 20, 30],
        vec![
            Segment::DictIndices(vec![0, 1, 2]),
            Segment::Plain(vec![100, 200]),
        ],
        5,
    );
    let mask = vec![false, true, false, true, false];
    assert_eq!(chunk.gather(&mask), vec![20, 100]);
}

#[test]
fn empty_dict_chunk_returns_empty_results() {
    let chunk = DictColumnChunk::<i32>::new(None, vec![], 0);
    assert_eq!(chunk.collect(), Vec::<i32>::new());
    assert_eq!(chunk.count_matching(|_| true), 0);
}

#[test]
fn collect_then_count_matches_filter_count_for_random_predicate() {
    // 50 dict entries, 1000 indices, mixed segments.
    let dict: Vec<i32> = (0..50).collect();
    let indices: Vec<u32> = (0..1000).map(|i| (i % 50) as u32).collect();
    let plain: Vec<i32> = (1000..1100).collect();

    let chunk = dict_chunk_i32(
        dict.clone(),
        vec![Segment::DictIndices(indices), Segment::Plain(plain)],
        1100,
    );

    // Predicate: even values >= 20
    let pred = |v: i32| v >= 20 && v % 2 == 0;
    let collected = chunk.collect();
    let collected_count = collected.iter().filter(|&&v| pred(v)).count();
    let filtered = chunk.filter(pred);
    let filter_count = filtered.iter().filter(|&&b| b).count();
    let count_matching = chunk.count_matching(pred);

    assert_eq!(collected_count, filter_count);
    assert_eq!(collected_count, count_matching);
}
