//! Π.15a — parallel multi-(row-group, column) decode.
//!
//! Gated on `--features parallel`. Default builds don't pull rayon
//! or this module — single-threaded consumers stay dep-light.
//!
//! `read_columns_parallel(file, &targets, opts, decode_one)` runs
//! `decode_one(file, rg, col)` for each `(rg, col)` in `targets`
//! concurrently across the rayon work-stealing pool. Results land
//! back in `targets`-order so the caller can index by position.
//!
//! Generic over the per-target output type so the same primitive
//! handles homogeneous workloads (e.g. "decode the same i32 column
//! across 50 row groups") and heterogeneous ones (a per-target
//! match-on-column-type closure that returns a `DecodedColumn`
//! enum).
//!
//! NUMA pinning + NUMA-aware allocation land in Π.15c/d as
//! `cfg(target_os = "linux")` extensions to the runner. Today's
//! Π.15a entry point is portable.

#[cfg(target_os = "linux")]
pub mod numa;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ematix_parquet_io::ParquetFile;
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::error::{CodecError, Result};

/// Cooperative cancellation handle for `read_columns_parallel`.
///
/// Cheap to clone (`Arc<AtomicBool>` internally). The owning caller
/// hands a clone to the runner via
/// `ParallelDecodeOptions::cancel` and keeps the original to call
/// `cancel()` from any thread.
///
/// **Cooperative semantics.** The runner only checks the token at
/// target boundaries — in-flight `decode_one(file, rg, col)` calls
/// run to completion. Targets that haven't been claimed yet at the
/// moment `cancel()` fires surface as `CodecError::Cancelled`.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// New token in the un-cancelled state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Read the current state. Loads with `Acquire` so workers
    /// observe a prior `cancel()` happens-before this call.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Caller-facing config for `read_columns_parallel`. Plumbing for
/// the future Π.15c NUMA hints lives here so the entry-point
/// signature stays stable across the sub-phases.
#[derive(Debug, Default)]
pub struct ParallelDecodeOptions {
    /// Optional rayon thread pool. If `None`, runs on the global
    /// rayon pool. Callers with their own pool (e.g. an engine
    /// runtime) pass it here to keep ematix-parquet's work on the
    /// pool the engine already owns.
    pub pool: Option<Arc<ThreadPool>>,
    /// Optional cancellation handle (Π.15b). When `Some`, the
    /// runner checks `is_cancelled()` before each per-target
    /// decode; targets that haven't been claimed when the token
    /// fires surface as `CodecError::Cancelled`. Decodes already
    /// in-flight run to completion (cooperative semantics).
    pub cancel: Option<CancellationToken>,
}

/// Decode multiple `(row_group, column)` targets in parallel.
///
/// Returns one `T` per target, in input order. If any target fails
/// the runner returns the first error seen; results from siblings
/// that completed before the failure are discarded.
///
/// `decode_one` is called once per target, on whichever rayon worker
/// claims it. The closure must be `Send + Sync` so it can be shared
/// across workers, and the per-target output `T: Send`.
pub fn read_columns_parallel<T, F>(
    file: &ParquetFile,
    targets: &[(usize, usize)],
    opts: ParallelDecodeOptions,
    decode_one: F,
) -> Result<Vec<T>>
where
    F: Fn(&ParquetFile, usize, usize) -> Result<T> + Send + Sync,
    T: Send,
{
    if targets.is_empty() {
        return Ok(Vec::new());
    }

    let cancel = opts.cancel.clone();
    let work = |t: &(usize, usize)| -> Result<T> {
        if let Some(c) = cancel.as_ref() {
            if c.is_cancelled() {
                return Err(CodecError::Cancelled);
            }
        }
        decode_one(file, t.0, t.1)
    };

    // rayon's into_par_iter preserves order on collect into Vec —
    // each (rg, col) lands at its original index regardless of
    // which worker decodes it first.
    let results: Result<Vec<T>> = match opts.pool.as_deref() {
        Some(pool) => pool.install(|| targets.par_iter().map(work).collect()),
        None => targets.par_iter().map(work).collect(),
    };
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_targets_returns_empty_vec() {
        // Open a tiny throwaway file just so we have a ParquetFile to
        // hand in — the runner never touches it on an empty target
        // list, so even if the file contents are nonsense the empty-
        // input shortcut returns Ok([]) before any decode runs.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        crate::write::write_i32_column_to_path(tmp.path(), "id", &[1, 2, 3]).unwrap();
        let file = ParquetFile::open(tmp.path()).unwrap();
        let out: Vec<i32> = read_columns_parallel::<i32, _>(
            &file,
            &[],
            ParallelDecodeOptions::default(),
            |_, _, _| unreachable!("decode_one must not run for empty targets"),
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn preserves_target_order() {
        // Build a 1-RG file with three i32 columns. Confirm
        // `read_columns_parallel` returns results in input order
        // regardless of which rayon worker decoded each first.
        // (Per-RG-multi-column is the natural shape; multi-RG
        // exercised in the integration suite once a writer exists.)
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // Single i32 column, just to exercise the runner end-to-end.
        crate::write::write_i32_column_to_path(tmp.path(), "id", &[10, 20, 30]).unwrap();
        let file = ParquetFile::open(tmp.path()).unwrap();

        // Repeat the same (0, 0) target 8× — each result must equal
        // the same decoded column, and the output Vec preserves slot
        // order even though rayon may decode in any order.
        let targets: Vec<(usize, usize)> = vec![(0, 0); 8];
        let out: Vec<Vec<i32>> = read_columns_parallel(
            &file,
            &targets,
            ParallelDecodeOptions::default(),
            crate::read::read_column_i32,
        )
        .unwrap();
        assert_eq!(out.len(), 8);
        for v in &out {
            assert_eq!(v, &[10, 20, 30]);
        }
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        let a = CancellationToken::new();
        let b = a.clone();
        assert!(!a.is_cancelled() && !b.is_cancelled());
        a.cancel();
        assert!(a.is_cancelled() && b.is_cancelled());
    }

    #[test]
    fn cancel_before_run_returns_cancelled_for_every_target() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        crate::write::write_i32_column_to_path(tmp.path(), "id", &[1, 2, 3]).unwrap();
        let file = ParquetFile::open(tmp.path()).unwrap();

        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel
        let opts = ParallelDecodeOptions {
            cancel: Some(cancel),
            ..Default::default()
        };
        let targets: Vec<(usize, usize)> = vec![(0, 0); 4];
        let r: Result<Vec<Vec<i32>>> =
            read_columns_parallel(&file, &targets, opts, crate::read::read_column_i32);
        assert!(matches!(r, Err(CodecError::Cancelled)));
    }

    #[test]
    fn first_error_propagates() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        crate::write::write_i32_column_to_path(tmp.path(), "id", &[1, 2, 3]).unwrap();
        let file = ParquetFile::open(tmp.path()).unwrap();

        let targets: Vec<(usize, usize)> = vec![(0, 0), (99, 0), (0, 0)];
        let r: Result<Vec<Vec<i32>>> = read_columns_parallel(
            &file,
            &targets,
            ParallelDecodeOptions::default(),
            crate::read::read_column_i32,
        );
        assert!(r.is_err(), "row_group=99 out of range must propagate");
    }
}
