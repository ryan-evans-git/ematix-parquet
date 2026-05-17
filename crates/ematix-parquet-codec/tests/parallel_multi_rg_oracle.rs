//! Π.15a — `read_columns_parallel` end-to-end across a real
//! multi-row-group file.
//!
//! Build a 4-RG file via `write_table_to_path_with_row_group_size`,
//! then decode every (RG, col) pair via the parallel runner and
//! compare against the sequential reference.

#![cfg(feature = "parallel")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ematix_parquet_codec::error::CodecError;
use ematix_parquet_codec::parallel::{
    read_columns_parallel, CancellationToken, ParallelDecodeOptions,
};
use ematix_parquet_codec::read::read_column_i32;
use ematix_parquet_codec::write::{write_table_to_path_with_row_group_size, ColumnData};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;
use tempfile::NamedTempFile;

fn build_multi_rg_i32_file() -> (NamedTempFile, Vec<i32>) {
    // 4096 rows × 1 column (i32). row_group_size=1024 → 4 RGs.
    let values: Vec<i32> = (0..4096).map(|i| i * 3 - 7).collect();
    let tmp = NamedTempFile::new().unwrap();
    write_table_to_path_with_row_group_size(
        tmp.path(),
        &[("id", ColumnData::I32(&values))],
        CompressionCodec::Uncompressed,
        1024,
    )
    .unwrap();
    (tmp, values)
}

#[test]
fn parallel_decode_matches_sequential_across_4_row_groups() {
    let (tmp, values) = build_multi_rg_i32_file();
    let file = ParquetFile::open(tmp.path()).unwrap();
    let md = file.metadata().expect("metadata");
    assert_eq!(md.row_groups.len(), 4, "test relies on 4 row groups");

    let targets: Vec<(usize, usize)> = (0..4).map(|rg| (rg, 0)).collect();
    let parallel: Vec<Vec<i32>> = read_columns_parallel(
        &file,
        &targets,
        ParallelDecodeOptions::default(),
        read_column_i32,
    )
    .unwrap();

    // Sequential reference.
    let sequential: Vec<Vec<i32>> = (0..4)
        .map(|rg| read_column_i32(&file, rg, 0).unwrap())
        .collect();

    assert_eq!(
        parallel, sequential,
        "parallel output must match sequential"
    );

    // Sanity: concatenated parallel result reconstructs the original values.
    let flat: Vec<i32> = parallel.into_iter().flatten().collect();
    assert_eq!(flat, values, "concat of RGs must equal the source");
}

#[test]
fn parallel_decode_with_caller_pool() {
    let (tmp, values) = build_multi_rg_i32_file();
    let file = ParquetFile::open(tmp.path()).unwrap();

    // Caller-owned 2-thread pool — exercises the `opts.pool` path.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    let opts = ParallelDecodeOptions {
        pool: Some(pool.clone()),
        cancel: None,
    };

    let targets: Vec<(usize, usize)> = (0..4).map(|rg| (rg, 0)).collect();
    let parallel: Vec<Vec<i32>> =
        read_columns_parallel(&file, &targets, opts, read_column_i32).unwrap();

    let flat: Vec<i32> = parallel.into_iter().flatten().collect();
    assert_eq!(flat, values);
}

#[test]
fn cancellation_mid_flight_returns_cancelled() {
    // Build a many-target workload so some land before cancel and
    // others after. We can't reliably claim "exactly N completed
    // before cancel", but we *can* assert that the runner surfaces
    // `CodecError::Cancelled` when targets are still queued at the
    // moment cancel() fires.
    let (tmp, _) = (NamedTempFile::new().unwrap(), ());
    {
        let values: Vec<i32> = (0..1024).collect();
        ematix_parquet_codec::write::write_table_to_path_with_row_group_size(
            tmp.path(),
            &[("id", ematix_parquet_codec::write::ColumnData::I32(&values))],
            ematix_parquet_format::types::CompressionCodec::Uncompressed,
            128, // 8 row groups
        )
        .unwrap();
    }
    let file = ParquetFile::open(tmp.path()).unwrap();

    let cancel = CancellationToken::new();
    let cancel_in_closure = cancel.clone();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_in_closure = counter.clone();

    // Constrain to a single-worker pool so the closure runs targets
    // sequentially — gives us a deterministic "after Nth call, fire
    // cancel" hook without needing barriers.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap(),
    );
    let opts = ParallelDecodeOptions {
        pool: Some(pool),
        cancel: Some(cancel.clone()),
    };
    let targets: Vec<(usize, usize)> = (0..8).map(|rg| (rg, 0)).collect();

    let r: Result<Vec<Vec<i32>>, _> = read_columns_parallel(&file, &targets, opts, |f, rg, col| {
        // Fire cancel after the second target has started decoding.
        let n = counter_in_closure.fetch_add(1, Ordering::SeqCst);
        if n == 1 {
            cancel_in_closure.cancel();
        }
        read_column_i32(f, rg, col)
    });

    assert!(matches!(r, Err(CodecError::Cancelled)));
    // The first 2 targets should have run; later ones short-circuit.
    let observed = counter.load(Ordering::SeqCst);
    assert!(
        (2..=3).contains(&observed),
        "expected 2 or 3 decode_one calls before cancel kicked in (got {observed})"
    );
}
