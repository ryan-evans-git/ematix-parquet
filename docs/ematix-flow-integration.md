# Integrating sidecar indexes into ematix-flow

This guide shows how `ematix-flow` (or any downstream query engine on top of this codec) consumes sidecar indexes for predicate pushdown. Every API mentioned here is on `main`; the same patterns work for any consumer crate.

## The opt-in principle

A sidecar adds capability but never changes the source. ematix-flow's existing read paths work unchanged whether or not a sidecar is present. The integration points are:

1. **Open the sidecar lazily** when the planner sees an indexed predicate.
2. **Use the index's hits** to drive a masked-decode of the projected columns.
3. **Fall back to full scan** when no sidecar exists or the index doesn't cover the predicate.

No file-format changes. No new write paths required. ematix-flow can ship indexed read support without coordinating with whatever produced the source Parquet.

## Pattern 1 — plain Parquet (no Iceberg)

The common case: ematix-flow opens a list of `.parquet` files (local FS, S3, GCS). For each file, look for a sidebar at `<source>.idx`. If present, use it. Otherwise scan.

```rust
use std::path::Path;
use ematix_parquet_codec::index::{ParquetIndex, ManifestError};
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_io::ParquetFile;

/// Read `target_col` from `source` keeping only rows whose
/// `idx_customer`-column value equals `customer_id`. Uses the
/// sidecar if one is present and current; falls back to full
/// scan otherwise.
fn read_customer_rows(
    source_path: &Path,
    customer_id: i64,
    target_col: usize,
) -> anyhow::Result<Vec<i64>> {
    let source = ParquetFile::open(source_path)?;
    let sidecar_path = source_path.with_extension("parquet.idx");

    if sidecar_path.exists() {
        match ParquetIndex::open(&sidecar_path, &source) {
            Ok(idx) => {
                // Sidecar fresh and our build understands it.
                return Ok(idx.read_column_i64_where_eq(
                    &source, "idx_customer", customer_id, target_col,
                )?);
            }
            Err(e) if is_stale(&e) => {
                // Source was rewritten; rebuild offline, fall through.
                tracing::warn!(?sidecar_path, "stale sidecar, falling back");
            }
            Err(e) => return Err(e.into()),
        }
    }

    // No sidecar (or stale) — full scan + scalar filter.
    let mut out = Vec::new();
    let n_rg = source.metadata()?.row_groups.len();
    for rg in 0..n_rg {
        let key_col = read_column_i64(&source, rg, /*index col*/ 0)?;
        let val_col = read_column_i64(&source, rg, target_col)?;
        for (k, v) in key_col.into_iter().zip(val_col) {
            if k == customer_id { out.push(v); }
        }
    }
    Ok(out)
}

fn is_stale(err: &impl std::fmt::Debug) -> bool {
    format!("{err:?}").contains("SourceFingerprintMismatch")
}
```

Three takeaways:

- The path convention `<source>.parquet.idx` is just a default. Anywhere you'd resolve a file path works.
- `SourceFingerprintMismatch` is a **recoverable** error — treat it as "no sidecar" and fall through; emit a metric so a background job can rebuild.
- The "index column ordinal" the builder used is recorded inside the sidecar's manifest. ematix-flow doesn't need to know it; the index `name` ("idx_customer") is the only identifier the query layer carries.

## Pattern 2 — Iceberg datasets

When ematix-flow reads from an Iceberg table (single source of truth, many files, snapshot semantics), the per-file sidecar still works — but you also want to **prune at the file level before opening any sidecar**, so a 10M-file dataset doesn't open 10M sidecars to find the 12 with matching rows.

That's what `ematix-iceberg` provides. Enable the `iceberg` feature and use:

```rust
use ematix_iceberg::iceberg_rs::{
    collect_data_files, prune_data_files_eq, pair_with_extensions,
};
use ematix_parquet_codec::index::{Key, ParquetIndex};
use ematix_parquet_io::ParquetFile;

async fn query_iceberg_indexed(
    table: &iceberg::table::Table,
    customer_id: i64,
    target_col: usize,
) -> anyhow::Result<Vec<i64>> {
    // 1. Walk the current snapshot's manifests.
    let files = collect_data_files(table).await?;

    // 2. File-level prune via the per-file IndexSummary embedded in
    //    each data_file's key_metadata. O(1) per file — no I/O.
    let pruned = prune_data_files_eq(&files, "idx_customer", &Key::I64(customer_id))?;

    // 3. Pair surviving files with their decoded extension + resolved
    //    sidecar URI. Drops files that have no sidecar.
    let candidates =
        pair_with_extensions(pruned.iter().map(|d| (*d).clone()).collect())?;

    // 4. For each candidate, open + lookup + masked-decode.
    let mut out = Vec::new();
    for c in &candidates {
        let source = ParquetFile::open(strip_uri_scheme(c.data_file.file_path()))?;
        let idx = ParquetIndex::open(strip_uri_scheme(&c.sidecar_uri), &source)?;
        let rows = idx.read_column_i64_where_eq(
            &source, "idx_customer", customer_id, target_col,
        )?;
        out.extend(rows);
    }
    Ok(out)
}

fn strip_uri_scheme(uri: &str) -> &str {
    uri.strip_prefix("file://").unwrap_or(uri)
}
```

The pruning hierarchy fans out: **manifest summary** (cheap, in-memory) → **sidecar lookup** (one parquet open per surviving file) → **masked decode** (only the rows that match). Cost grows with how many files survive each gate, not with table size.

URI resolution today expects local paths (or `file://` URIs). When `ematix-parquet-async` grows an async `ParquetIndex` opener (planned), the same pattern will work over `object_store` directly.

## Pattern 3 — write-side: producing sidecars

ematix-flow doesn't have to be the producer, but if it owns the ingest path it should emit sidecars at write time. Add this immediately after writing each Parquet file:

```rust
use ematix_parquet_codec::index::IndexBuilder;
use ematix_parquet_io::ParquetFile;

fn emit_sidecar(parquet_path: &Path, indexed_cols: &[(&str, usize)]) -> anyhow::Result<()> {
    let source = ParquetFile::open(parquet_path)?;
    let sidecar_path = parquet_path.with_extension("parquet.idx");
    let mut builder = IndexBuilder::new(&source);
    for (name, col) in indexed_cols {
        builder = builder.write_sorted_i64(&sidecar_path, name, *col)?;
    }
    Ok(())
}
```

Call this after every successful parquet write, in the same atomic-rename pattern used for the data file. The sidecar's footer-fingerprint binding ensures the two are consistent — a stale sidecar fails fast at open time.

For an Iceberg writer, also stamp the per-file summary onto the `data_file.key_metadata` before committing the manifest:

```rust
use ematix_iceberg::iceberg_rs::{attach_extension, encode_key_metadata};
use ematix_iceberg::{EmatixDataFileExtension, IndexSummary, SummaryKey};

let ext = EmatixDataFileExtension {
    sidecar_relative_path: "data.parquet.idx".into(),
    summaries: vec![
        IndexSummary::new("idx_customer")
            .with_range(SummaryKey::I64(min_seen), SummaryKey::I64(max_seen)),
    ],
};
let data_file_builder = attach_extension(data_file_builder, &ext);
```

`min_seen` / `max_seen` are typically already computed for Parquet's per-column statistics — just thread them into the summary.

## Operational concerns

**Sidecar staleness on object stores.** S3 / GCS PUT is atomic; sidecars and data files can be written in either order as long as a reader retries on `SourceFingerprintMismatch`. The typical pattern: PUT data → PUT sidecar → atomic-rename data into final location → atomic-rename sidecar. Readers that race in see no sidecar, fall back to scan.

**Cost attribution.** Sidecar I/O shows up as `*.idx` reads in object-store metrics. Tag them with the same trace as the data-file read so query latency attribution stays clean.

**Index lifecycle.** Sidecars are not part of the Parquet spec — they live in the same prefix but the table format doesn't garbage-collect them. If you drop a data file, drop its sidecar too. The Iceberg layer's manifest already names the sidecar per `data_file`, so GC can iterate manifests to enumerate sidecars to keep.

**Build cost vs. query cost.** A sidecar build is ~10× the cost of a full read of the indexed column (see [sidecar-indexes.md § Build cost](sidecar-indexes.md#build-cost)). Build is amortized across all future queries — for write-once / read-many workloads it pays back after a handful of queries.

## Selectivity-aware fallback

The bench in [`bench_indexed_lookup`](../crates/ematix-parquet-codec/examples/bench_indexed_lookup.rs) shows the crossover at ~60% selectivity. ematix-flow's planner can use the per-file `IndexSummary` *before* opening the sidecar to decide:

```rust
// Cheap planner heuristic: if the predicate covers >= 80% of the
// file's value range, scan is faster than sidecar lookup.
let summary = pair.extension.summary("idx_customer").unwrap();
let estimated_selectivity = estimate_range_overlap(summary, &query_range);
if estimated_selectivity > 0.8 {
    // full_scan_path(&source, target_col, &query_range)
} else {
    // indexed_path(&source, &idx, target_col, &query_range)
}
```

This is the right place to plug a cost-based optimizer signal. The codec itself stays cost-blind — it does what the caller asks; the *planner* decides which path to ask for.

## See also

- [`docs/sidecar-indexes.md`](sidecar-indexes.md) — full reference for the index types, wire format, and perf characteristics.
- [`crates/ematix-iceberg/src/iceberg_rs.rs`](../crates/ematix-iceberg/src/iceberg_rs.rs) — public surface for the Iceberg integration.
- [`crates/ematix-iceberg/tests/iceberg_oracle.rs`](../crates/ematix-iceberg/tests/iceberg_oracle.rs) — end-to-end test exercising the whole flow (walker → prune → pair) against a real Iceberg fixture.
