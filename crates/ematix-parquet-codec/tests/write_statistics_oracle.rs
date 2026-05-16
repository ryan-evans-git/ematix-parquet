//! Oracle: ours writes → parquet-rs sees the right column-chunk stats.
//!
//! Π.4a contract: every column we emit advertises `min`, `max`, and
//! `null_count` in `ColumnMetaData.statistics`, using the modern
//! (`min_value`/`max_value`) plus deprecated (`min`/`max`) field
//! pairs. Downstream predicate pushdown depends on this.

use ematix_parquet_codec::write::{
    write_bool_column_to_path, write_byte_array_column_to_path, write_f64_column_to_path,
    write_i32_column_to_path, write_i64_column_to_path, write_table_to_path, ColumnData,
};
use ematix_parquet_format::types::CompressionCodec;

use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics as PqStats;

fn column_stats(path: &std::path::Path, col_ix: usize) -> PqStats {
    let f = std::fs::File::open(path).expect("open");
    let r = SerializedFileReader::new(f).expect("parquet-rs reader");
    let rg = r.get_row_group(0).expect("rg 0");
    rg.metadata()
        .column(col_ix)
        .statistics()
        .expect("statistics on column chunk")
        .clone()
}

// ---- i64 ----

#[test]
fn i64_stats_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.parquet");

    let values: Vec<i64> = vec![7, -3, 42, 0, 100, -1000, 999];
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Int64(ts) = s else {
        panic!("expected Int64 stats, got {s:?}");
    };
    assert_eq!(ts.min_opt(), Some(&-1000));
    assert_eq!(ts.max_opt(), Some(&999));
    assert_eq!(ts.null_count_opt(), Some(0));
}

#[test]
fn i64_stats_single_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_one.parquet");

    let values: Vec<i64> = vec![42];
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Int64(ts) = s else {
        panic!("expected Int64 stats");
    };
    assert_eq!(ts.min_opt(), Some(&42));
    assert_eq!(ts.max_opt(), Some(&42));
}

#[test]
fn i64_stats_extremes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_extremes.parquet");

    let values: Vec<i64> = vec![i64::MIN, 0, i64::MAX];
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Int64(ts) = s else {
        panic!("expected Int64 stats");
    };
    assert_eq!(ts.min_opt(), Some(&i64::MIN));
    assert_eq!(ts.max_opt(), Some(&i64::MAX));
}

// ---- i32 ----

#[test]
fn i32_stats_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.parquet");

    let values: Vec<i32> = vec![5, -2, 17, 0, i32::MIN, i32::MAX];
    write_i32_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Int32(ts) = s else {
        panic!("expected Int32 stats, got {s:?}");
    };
    assert_eq!(ts.min_opt(), Some(&i32::MIN));
    assert_eq!(ts.max_opt(), Some(&i32::MAX));
    assert_eq!(ts.null_count_opt(), Some(0));
}

// ---- f64 ----

#[test]
fn f64_stats_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.parquet");

    let values: Vec<f64> = vec![1.5, -2.0, 3.14, 0.0, -7.25];
    write_f64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Double(ts) = s else {
        panic!("expected Double stats, got {s:?}");
    };
    assert_eq!(ts.min_opt(), Some(&-7.25));
    assert_eq!(ts.max_opt(), Some(&3.14));
    assert_eq!(ts.null_count_opt(), Some(0));
}

#[test]
fn f64_stats_skip_nan() {
    // NaN must not appear in min/max. The remaining real values
    // determine the bounds.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64_nan.parquet");

    let values: Vec<f64> = vec![1.0, f64::NAN, 5.0, f64::NAN, -2.0];
    write_f64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Double(ts) = s else {
        panic!("expected Double stats");
    };
    assert_eq!(ts.min_opt(), Some(&-2.0));
    assert_eq!(ts.max_opt(), Some(&5.0));
}

#[test]
fn f64_stats_zero_normalisation() {
    // +0.0 in input → -0.0 in min; -0.0 in input → +0.0 in max.
    // Keeps range predicates that straddle zero from incorrectly
    // pruning pages that contain a zero of either sign.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64_zero.parquet");

    let values: Vec<f64> = vec![0.0, 0.0, 0.0];
    write_f64_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Double(ts) = s else {
        panic!("expected Double stats");
    };
    // -0.0 < +0.0 by bit pattern, so:
    assert_eq!(
        ts.min_opt().map(|x| x.to_bits()),
        Some((-0.0_f64).to_bits())
    );
    assert_eq!(ts.max_opt().map(|x| x.to_bits()), Some(0.0_f64.to_bits()));
}

// ---- bool ----

#[test]
fn bool_stats_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bool.parquet");

    let values: Vec<bool> = vec![true, false, true, true, false];
    write_bool_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Boolean(ts) = s else {
        panic!("expected Boolean stats, got {s:?}");
    };
    assert_eq!(ts.min_opt(), Some(&false));
    assert_eq!(ts.max_opt(), Some(&true));
    assert_eq!(ts.null_count_opt(), Some(0));
}

#[test]
fn bool_stats_all_true() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bool_all_true.parquet");

    let values: Vec<bool> = vec![true; 8];
    write_bool_column_to_path(&path, "v", &values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::Boolean(ts) = s else {
        panic!("expected Boolean stats");
    };
    assert_eq!(ts.min_opt(), Some(&true));
    assert_eq!(ts.max_opt(), Some(&true));
}

// ---- byte_array ----

#[test]
fn byte_array_stats_lex_ordering() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba.parquet");

    let values: &[&[u8]] = &[b"cherry", b"apple", b"banana", b"date"];
    write_byte_array_column_to_path(&path, "v", values).unwrap();

    let s = column_stats(&path, 0);
    let PqStats::ByteArray(ts) = s else {
        panic!("expected ByteArray stats, got {s:?}");
    };
    let want_min: PqByteArray = b"apple".to_vec().into();
    let want_max: PqByteArray = b"date".to_vec().into();
    assert_eq!(ts.min_opt(), Some(&want_min));
    assert_eq!(ts.max_opt(), Some(&want_max));
}

// ---- multi-column table ----

#[test]
fn multi_column_table_stats() {
    // write_table_to_path computes stats per column independently.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table.parquet");

    let ids: Vec<i64> = vec![10, 5, 20, 15];
    let prices: Vec<f64> = vec![1.5, 9.0, 3.0, 4.5];
    let names: Vec<&[u8]> = vec![b"z", b"a", b"m", b"k"];

    let cols: Vec<(&str, ColumnData<'_>)> = vec![
        ("id", ColumnData::I64(&ids)),
        ("price", ColumnData::F64(&prices)),
        ("name", ColumnData::ByteArray(&names)),
    ];
    write_table_to_path(&path, &cols, CompressionCodec::Snappy).unwrap();

    let PqStats::Int64(s_id) = column_stats(&path, 0) else {
        panic!("id")
    };
    assert_eq!(s_id.min_opt(), Some(&5));
    assert_eq!(s_id.max_opt(), Some(&20));

    let PqStats::Double(s_price) = column_stats(&path, 1) else {
        panic!("price")
    };
    assert_eq!(s_price.min_opt(), Some(&1.5));
    assert_eq!(s_price.max_opt(), Some(&9.0));

    let PqStats::ByteArray(s_name) = column_stats(&path, 2) else {
        panic!("name")
    };
    let want_min: PqByteArray = b"a".to_vec().into();
    let want_max: PqByteArray = b"z".to_vec().into();
    assert_eq!(s_name.min_opt(), Some(&want_min));
    assert_eq!(s_name.max_opt(), Some(&want_max));
}

// ---- predicate-pushdown smoke ----
//
// Predicate pushdown in `parquet-rs` (and any other reader) reads the
// column-chunk `Statistics` that we just verified above. So a stats
// round-trip oracle implicitly covers "pushdown sees what it needs."
//
// To make the dependency explicit, this test reaches into the
// row-group metadata and simulates the pruning decision a reader
// would make: for a predicate `v >= 5000` against a file whose
// max is 999, the chunk should be skippable.

#[test]
fn pushdown_decision_for_out_of_range_predicate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sorted.parquet");

    let values: Vec<i64> = (0i64..1000).collect();
    write_i64_column_to_path(&path, "v", &values).unwrap();

    let f = std::fs::File::open(&path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let cc = r.get_row_group(0).unwrap().metadata().column(0).clone();
    let s = cc.statistics().expect("stats present");
    let PqStats::Int64(ts) = s else {
        panic!("expected Int64");
    };
    let chunk_max = *ts.max_opt().expect("max present");

    // A real reader prunes when threshold > chunk_max (for >=
    // predicates). We assert the decision a reader will make, given
    // the stats we emit.
    let threshold = 5000i64;
    assert!(
        threshold > chunk_max,
        "stats permit pruning: threshold {threshold} > chunk_max {chunk_max}"
    );
}
