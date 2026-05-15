//! Oracle: ours emits N row groups → parquet-rs and our own reader
//! both walk all of them and recover every value.
//!
//! Π.4b contract: `write_table_to_path_with_row_group_size` cuts a
//! new row group every `row_group_size` rows. Stats are per-RG so
//! readers can prune individual row groups, not just the whole file.

use ematix_parquet_codec::write::{
    write_table_to_path_with_row_group_size, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;

use parquet::column::reader::ColumnReader;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics as PqStats;

fn open_parquet_rs(path: &std::path::Path) -> SerializedFileReader<std::fs::File> {
    let f = std::fs::File::open(path).expect("open");
    SerializedFileReader::new(f).expect("parquet-rs reader")
}

fn read_back_all_i64(path: &std::path::Path) -> Vec<i64> {
    let r = open_parquet_rs(path);
    let n_rg = r.metadata().num_row_groups();
    let mut out: Vec<i64> = Vec::new();
    for rg_ix in 0..n_rg {
        let rg = r.get_row_group(rg_ix).expect("rg");
        let total = rg.metadata().column(0).num_values() as usize;
        let cr = rg.get_column_reader(0).expect("col 0");
        let ColumnReader::Int64ColumnReader(mut typed) = cr else {
            panic!("expected INT64");
        };
        let mut chunk: Vec<i64> = Vec::with_capacity(total);
        typed
            .read_records(total, None, None, &mut chunk)
            .expect("read_records");
        out.extend(chunk);
    }
    out
}

// ---- shape: 1000 rows × RG-size 100 → 10 RGs ----

#[test]
fn ten_row_groups_of_one_hundred() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.parquet");

    let values: Vec<i64> = (0i64..1000).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 100)
        .unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 10);
    assert_eq!(r.metadata().file_metadata().num_rows(), 1000);

    // Per-RG num_rows is 100.
    for rg_ix in 0..10 {
        assert_eq!(r.get_row_group(rg_ix).unwrap().metadata().num_rows(), 100);
    }
}

// ---- last RG carries the remainder when total isn't divisible ----

#[test]
fn uneven_last_row_group_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("uneven.parquet");

    let values: Vec<i64> = (0i64..1050).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 200)
        .unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 6); // 5×200 + 1×50
    let last = r.get_row_group(5).unwrap();
    assert_eq!(last.metadata().num_rows(), 50);
}

// ---- values round-trip end-to-end across all row groups ----

#[test]
fn values_round_trip_via_parquet_rs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rt.parquet");

    let values: Vec<i64> = (0i64..1000).map(|i| i * 13 - 500).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Snappy, 250).unwrap();

    let read_back = read_back_all_i64(&path);
    assert_eq!(read_back, values);
}

// ---- per-row-group stats are tight (sorted column) ----

#[test]
fn per_row_group_stats_are_tight() {
    // A sorted column with RG-size 100: RG 0 has min=0/max=99,
    // RG 1 has min=100/max=199, etc. Confirms stats are computed
    // per row group (not file-wide).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sorted.parquet");

    let values: Vec<i64> = (0i64..500).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 100)
        .unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 5);

    for rg_ix in 0..5 {
        let cc = r
            .get_row_group(rg_ix)
            .unwrap()
            .metadata()
            .column(0)
            .clone();
        let PqStats::Int64(ts) = cc.statistics().expect("stats") else {
            panic!("expected Int64 stats");
        };
        let want_min = (rg_ix as i64) * 100;
        let want_max = want_min + 99;
        assert_eq!(ts.min_opt(), Some(&want_min), "rg {rg_ix} min");
        assert_eq!(ts.max_opt(), Some(&want_max), "rg {rg_ix} max");
    }
}

// ---- our own reader walks all row groups ----

#[test]
fn our_reader_walks_all_row_groups() {
    use ematix_parquet_codec::read::read_column_i64;
    use ematix_parquet_io::ParquetFile;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("self.parquet");

    let values: Vec<i64> = (0i64..600).map(|i| i * 7).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 150)
        .unwrap();

    let file = ParquetFile::open(&path).expect("open");
    // Concatenate every row group via our high-level façade.
    let md = file.metadata().expect("metadata");
    let n_rg = md.row_groups.len();
    assert_eq!(n_rg, 4); // 600 / 150
    drop(md);

    let mut out: Vec<i64> = Vec::with_capacity(values.len());
    for rg_ix in 0..n_rg {
        out.extend(read_column_i64(&file, rg_ix, 0).unwrap());
    }
    assert_eq!(out, values);
}

// ---- single row group when row_group_size >= total_rows ----

#[test]
fn rg_size_larger_than_data_yields_one_rg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.parquet");

    let values: Vec<i64> = (0i64..50).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 10_000)
        .unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 1);
}

// ---- existing single-RG entry point unaffected ----

#[test]
fn default_write_table_still_single_rg() {
    use ematix_parquet_codec::write::write_table_to_path;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("default.parquet");

    let values: Vec<i64> = (0i64..1000).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    write_table_to_path(&path, &cols, CompressionCodec::Uncompressed).unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 1);
}

// ---- multi-column, multi-RG: every column's stats are per-RG ----

#[test]
fn multi_column_multi_rg_stats_per_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_col.parquet");

    // Two columns, opposite sort directions to make per-RG bounds
    // distinguishable from file-wide ones.
    let ids: Vec<i64> = (0i64..400).collect();
    let neg: Vec<i64> = (0i64..400).map(|i| -i).collect();
    let cols: Vec<(&str, ColumnData<'_>)> = vec![
        ("id", ColumnData::I64(&ids)),
        ("neg", ColumnData::I64(&neg)),
    ];
    write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Snappy, 100).unwrap();

    let r = open_parquet_rs(&path);
    assert_eq!(r.metadata().num_row_groups(), 4);

    // RG 2 covers rows 200..300.
    let rg2 = r.get_row_group(2).unwrap();
    let PqStats::Int64(s_id) = rg2.metadata().column(0).statistics().unwrap() else {
        panic!()
    };
    assert_eq!(s_id.min_opt(), Some(&200));
    assert_eq!(s_id.max_opt(), Some(&299));

    let PqStats::Int64(s_neg) = rg2.metadata().column(1).statistics().unwrap() else {
        panic!()
    };
    assert_eq!(s_neg.min_opt(), Some(&-299));
    assert_eq!(s_neg.max_opt(), Some(&-200));
}

// ---- row_group_size = 0 is rejected ----

#[test]
fn zero_row_group_size_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.parquet");

    let values: Vec<i64> = vec![1, 2, 3];
    let cols: Vec<(&str, ColumnData<'_>)> = vec![("v", ColumnData::I64(&values))];
    let err =
        write_table_to_path_with_row_group_size(&path, &cols, CompressionCodec::Uncompressed, 0)
            .expect_err("must reject zero");
    let msg = format!("{err:?}");
    assert!(msg.contains("row_group_size"), "{msg}");
}
