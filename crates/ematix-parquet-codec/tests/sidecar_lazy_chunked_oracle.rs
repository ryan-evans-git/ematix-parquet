//! Oracle for v3 CHUNKED sidecar bodies + the lazy footer-pruned
//! reader.
//!
//! v1/v2 wrote the whole sorted-index body as ONE row group, and
//! `ParquetIndex::open` eagerly decodes all of it — ~1s and ~0.5-1 GB
//! per open on a 19M-value lineitem-part index, which made an indexed
//! point lookup SLOWER than a full table scan at SF100. v3 cuts the
//! body every `SIDECAR_RG_ROWS` rows; because index rows are sorted
//! by value, each row group's footer min/max form ordered ranges, and
//! [`LazyParquetIndex`] decodes only the group(s) containing the key.
//!
//! The load-bearing invariants pinned here:
//!   - chunked builds carry the v3 manifest key + an honest
//!     `sidecar_row_group_count`, and the EAGER reader still returns
//!     full parity by concatenating the groups;
//!   - the LAZY reader matches the eager reader / full-scan baseline
//!     for keys in the first group, an interior group, the last
//!     group, absent keys, and — the subtle one — a duplicate run
//!     STRADDLING a row-group boundary (two groups must both decode);
//!   - lazy also serves legacy single-group v2 sidecars.

use ematix_parquet_codec::index::{IndexBuilder, LazyParquetIndex, ParquetIndex, SIDECAR_RG_ROWS};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_io::ParquetFile;

const V3_KEY: &str = "ematix_index_manifest_v3";

fn baseline_eq(source: &ParquetFile, col: usize, key: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let md = source.metadata().unwrap();
    for rg in 0..md.row_groups.len() {
        let values = read_column_i64(source, rg, col).unwrap();
        out.extend(values.iter().copied().filter(|v| *v == key));
    }
    out
}

/// A fixture big enough for 3 sidecar row groups, with one duplicated
/// value planted so its index rows straddle the first RG boundary.
/// Returns (src, idx, straddle_key, last_key).
fn chunked_fixture(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf, i64, i64) {
    let n = SIDECAR_RG_ROWS * 2 + SIDECAR_RG_ROWS / 2; // 2.5 groups of distinct values
    let straddle_key = (SIDECAR_RG_ROWS - 50) as i64;
    let last_key = (n - 1) as i64;
    // Values 0..n unique, PLUS 100 extra copies of straddle_key. After
    // the builder sorts by value, the straddle run occupies index rows
    // [straddle_key, straddle_key+101) — crossing the SIDECAR_RG_ROWS
    // cut. (Source order: interleave the dups mid-stream; order
    // doesn't matter to the builder's sort.)
    let mut values: Vec<i64> = (0..n as i64).collect();
    values.extend(std::iter::repeat(straddle_key).take(100));
    let src = dir.join("chunked.parquet");
    let idx = dir.join("chunked.parquet.idx");
    ematix_parquet_codec::write::write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .expect("build chunked sidecar");
    (src, idx, straddle_key, last_key)
}

#[test]
fn chunked_body_shape_and_eager_parity() {
    let dir = tempfile::tempdir().unwrap();
    let (src, idx, straddle_key, last_key) = chunked_fixture(dir.path());

    // Shape: >1 sidecar row group, v3 manifest key, honest count.
    let f = ParquetFile::open(&idx).unwrap();
    let md = f.metadata().unwrap();
    assert!(
        md.row_groups.len() >= 3,
        "expected a chunked body, got {} row group(s)",
        md.row_groups.len()
    );
    let keys: Vec<String> = md
        .key_value_metadata
        .iter()
        .flatten()
        .map(|kv| String::from_utf8_lossy(kv.key).into_owned())
        .collect();
    assert!(keys.iter().any(|k| k == V3_KEY), "v3 key missing: {keys:?}");

    let source = ParquetFile::open(&src).unwrap();
    let pidx = ParquetIndex::open(&idx, &source).unwrap();
    assert_eq!(
        pidx.manifest().indexes[0].sidecar_row_group_count as usize,
        md.row_groups.len(),
        "manifest count must match the physical row groups"
    );

    // Eager reader concatenates the chunks: full parity.
    for key in [0i64, straddle_key, last_key, last_key + 7] {
        let indexed = pidx
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        assert_eq!(indexed, baseline_eq(&source, 0, key), "eager key={key}");
    }
}

#[test]
fn lazy_parity_including_boundary_straddle() {
    let dir = tempfile::tempdir().unwrap();
    let (src, idx, straddle_key, last_key) = chunked_fixture(dir.path());
    let source = ParquetFile::open(&src).unwrap();
    let lazy = LazyParquetIndex::open(&idx, &source).unwrap();

    // First group, interior, last group, absent — and the straddler,
    // whose 101 index rows span TWO sidecar row groups.
    for key in [
        0i64,
        7_777,
        straddle_key,
        (SIDECAR_RG_ROWS + 123) as i64,
        last_key,
        last_key + 42,
    ] {
        let got = lazy
            .read_column_i64_where_eq(&source, "idx_v", key, 0)
            .unwrap();
        let expect = baseline_eq(&source, 0, key);
        assert_eq!(got, expect, "lazy key={key}");
    }
    // The straddler must actually have its duplicates: 101 rows.
    let got = lazy
        .read_column_i64_where_eq(&source, "idx_v", straddle_key, 0)
        .unwrap();
    assert_eq!(got.len(), 101, "straddle run row count");
}

#[test]
fn lazy_serves_legacy_single_group_sidecars() {
    // A small (single-RG, v3-but-count-1) sidecar and lookups through
    // the lazy path — the same code must serve unchunked bodies.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<i64> = (0..10_000i64).map(|i| i % 1_000).collect();
    let src = dir.path().join("small.parquet");
    let idx = dir.path().join("small.parquet.idx");
    ematix_parquet_codec::write::write_i64_column_to_path(&src, "v", &values).unwrap();
    let source = ParquetFile::open(&src).unwrap();
    IndexBuilder::new(&source)
        .write_sorted_i64(&idx, "idx_v", 0)
        .unwrap();
    let lazy = LazyParquetIndex::open(&idx, &source).unwrap();
    for key in [0i64, 500, 999, 1_000] {
        assert_eq!(
            lazy.read_column_i64_where_eq(&source, "idx_v", key, 0)
                .unwrap(),
            baseline_eq(&source, 0, key),
            "key={key}"
        );
    }
}
