# Sidecar indexes

Add Postgres-style indexes to existing Parquet files **without rewriting them**. A sidecar (`.parquet.idx`) lives next to the source `.parquet` and is consumed at read time to skip whole row groups and masked-decode only the matching rows within the row groups that survive.

The sidecar itself is a standard Parquet file (using this codec's writer) — `parquet-tools` can inspect it; only this library knows what the schema means semantically.

## When to use one

- Selective equality / narrow-range predicates on a column the source file isn't sorted on.
- Multi-column composite filters (`a = X AND b = Y`).
- Text search (single-token contains).
- Any case where Iceberg's native file-level `min/max` doesn't prune tightly enough — e.g. high-cardinality columns where every file's range covers most of the key space.

## When *not* to use one

- Predicates that match >60% of rows (the index lookup tax outweighs the skip-decompress win — see [Performance](#performance)).
- Predicates on a column Iceberg / Parquet's native page-level `ColumnIndex` already prunes tightly (well-sorted columns with narrow per-page ranges).
- One-shot scans on small files — build cost dominates.

## API at a glance

```rust
use ematix_parquet_codec::index::{IndexBuilder, ParquetIndex, Tokenizer};
use ematix_parquet_io::ParquetFile;

let source = ParquetFile::open("data.parquet")?;

// Build — pick one or more index types per source file
IndexBuilder::new(&source)
    .write_sorted_i64(&"data.parquet.idx", "idx_customer", /*col*/ 0)?;
//  .write_sorted_i32(...)
//  .write_sorted_byte_array(...)
//  .write_bloom_page_i64(..., /*target_fpp*/ 0.01)?
//  .write_sorted_composite_prefix_i64_i64(..., (col_a, col_b))?
//  .write_inverted_byte_array(..., col, Tokenizer::WhitespaceLowercaseV1)?;

// Query
let idx = ParquetIndex::open("data.parquet.idx", &source)?;
let rows = idx.read_column_i64_where_eq(&source, "idx_customer", 42, /*target col*/ 3)?;
```

The "target col" is the column to *materialize* (the projected output). It can be different from the column the index was built on — the typical pattern is "index on `customer_id`, project `extended_price`".

## Index types

### Sorted

| Source type  | Build                            | Query (eq + range)                       |
| ------------ | -------------------------------- | ---------------------------------------- |
| `INT64`      | `write_sorted_i64`               | `read_column_i64_where_eq` / `_range`    |
| `INT32`      | `write_sorted_i32`               | `read_column_i32_where_eq` / `_range`    |
| `BYTE_ARRAY` | `write_sorted_byte_array`        | `read_column_byte_array_where_eq` / `_range` |

B-tree-like — for each distinct value, the sidecar stores `(value, target_rg, target_page, target_rowset)`. Equality looks up one key; range scans the sidecar's `[lo, hi]` window. Sidecar lookup is `O(log n)` via Parquet's native `ColumnIndex` on the sidecar's sorted-key column.

### Per-page Bloom (`INT64` only today)

| Build                | Query (eq)                                                                            |
| -------------------- | ------------------------------------------------------------------------------------- |
| `write_bloom_page_i64(..., target_fpp)` | `bloom_probe` (low-level) / `read_column_i64_via_bloom_eq` (convenience)              |

Stores one `SplitBlockBloomFilter` per source page. `bloom_probe(key)` returns the pages whose Bloom said "maybe"; the convenience entry decodes those pages and filters in-memory so false positives are eliminated. Lower memory overhead than sorted indexes for high-cardinality columns.

### Composite leading-prefix (`INT64 × INT64` today)

| Build                                          | Query                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| `write_sorted_composite_prefix_i64_i64`        | `read_column_i64_where_composite_eq`, `_where_composite_prefix`            |

Two-column sorted index. Supports `a = X AND b = Y` (full), `a = X` alone (leading-prefix), and `a = X AND b IN [...]`. Useful when the workload always filters on `a` and sometimes additionally on `b` — a single index covers both.

### Inverted text (`BYTE_ARRAY` only today)

| Build                                                     | Query                                  |
| --------------------------------------------------------- | -------------------------------------- |
| `write_inverted_byte_array(..., Tokenizer::WhitespaceLowercaseV1)` | `read_column_byte_array_where_token`   |

Posting list per token. Tokenizer choice is recorded in the manifest so the reader applies the same transform to query terms. Today's tokenizer splits on ASCII whitespace + ASCII-lowercases; the trait is forward-compatible for stemmers, Unicode-aware tokenizers, etc.

## Performance

Bench: `cargo run --release --example bench_indexed_lookup`. 1M-row `INT64` source, 100 distinct values in sorted runs, 100 row groups of 10K rows each. Median of 8 measured iterations on Apple Silicon:

| Predicate (selectivity) | Baseline (full scan) | Indexed (sidecar) | Speedup |
| ----------------------- | -------------------- | ----------------- | ------- |
| `eq @ min`              | 4.64 ms              | 0.12 ms           | **40×** |
| `eq @ mid`              | 3.77 ms              | 0.14 ms           | **26×** |
| `eq @ max`              | 3.76 ms              | 0.12 ms           | **33×** |
| `range 1%`              | 3.79 ms              | 0.12 ms           | **33×** |
| `range 5%`              | 3.84 ms              | 0.41 ms           | **9.5×** |
| `range 10%`             | 3.86 ms              | 0.79 ms           | 4.9×    |
| `range 50%`             | 6.17 ms              | 3.82 ms           | 1.6×    |
| `range 100%`            | 7.14 ms              | 8.20 ms           | 0.87×   |

The win comes from two stacked mechanisms:

1. **Row-group skip-decompress** — the sidecar pinpoints the 1 row group containing the key; the other 99 row groups are eliminated entirely (zero CPU spent on decompress/decode).
2. **Masked decode within the surviving group** — `read_column_*_masked_into` only emits the bytes that match.

The crossover where indexing stops paying for itself is around **60% selectivity**. Below that, the indexed path wins; above it, fall back to a full scan. The Iceberg layer's `IndexSummary` pruning handles this automatically — file-level prune before opening any sidecar.

## Wire format

The sidecar Parquet's row groups hold the index data; the footer's `KeyValueMetadata` under `ematix_index_manifest_v1` carries a JSON manifest:

```jsonc
{
  "version": "v1",
  "source_fingerprint": {
    "footer_length": 12345,
    "footer_crc32": 3735928559,
    "num_rows": 5000000,
    "num_row_groups": 16
  },
  "indexes": [
    { "name": "idx_customer", "type": "sorted",
      "source_column": "l_customer", "physical_type": "INT64",
      "sidecar_row_group": 0 },
    ...
  ]
}
```

JSON over Thrift because (a) it matches the shape Spark / Iceberg / Hudi already use; (b) the codec stays serde-free; (c) the per-sidecar overhead is sub-1 KB and compactness doesn't matter.

## Staleness + rebuild

A sidecar is **bound to a specific footer of its source** via the fingerprint (`footer_length + footer_crc32 + num_rows + num_row_groups`). Rewriting the source — even adding an extra row group at the end — invalidates the sidecar.

`ParquetIndex::open` returns `Err(ManifestError::SourceFingerprintMismatch)` on any mismatch. Detect that variant, rebuild the sidecar, retry. Two fresh builds against the same source produce byte-identical fingerprints, so detection is straightforward.

There is no incremental update: append-only workloads need a rebuild step. Acceptable for the Parquet model — files are usually immutable once written.

## Build cost

The builder loads the indexed column fully into memory, sorts (for sorted indexes) or hashes (for Bloom) the values, and allocates a per-page row bitmap per distinct value. Memory ceiling: `O(distinct_values × max_page_num_values / 8)` bytes plus the column itself.

For low-cardinality columns this is small. For unique-per-row columns it can balloon — use a page-Bloom index or a composite index instead. As a rough rule, sorted indexes target columns with `distinct_count ≤ num_rows / 100`.

Build wall-time on a 1M-row INT64 source: roughly 10× the full-scan read time (one pass to bucket values, one pass to write the sidecar). Amortizes across all future queries against that file.

## Failure modes

| Failure                                          | Variant                                          | Recovery                            |
| ------------------------------------------------ | ------------------------------------------------ | ----------------------------------- |
| Source file was rewritten                        | `ManifestError::SourceFingerprintMismatch`       | Rebuild the sidecar.                |
| Sidecar's `ematix_index_manifest_v1` key absent  | `ManifestError::Missing`                         | Not a real sidecar — check path.    |
| Sidecar version newer than this build supports   | `ManifestError::UnsupportedVersion`              | Upgrade `ematix-parquet-codec`.     |
| Query's `Key` variant ≠ index's physical type    | `CodecError::InvalidInput`                       | Planner picked the wrong index.     |
| Lookup against a kind it doesn't support         | `CodecError::InvalidInput` ("not a sorted index" etc.) | Use the matching API for that kind. |

All errors surface fast (before any decode work happens) — no silent wrong results.

## What's *not* yet supported

- Floating-point physical types (`FLOAT`, `DOUBLE`)
- `FIXED_LEN_BYTE_ARRAY`
- Multi-token text queries (AND / OR composition)
- Range queries on the page-Bloom index (eq only)
- Composite indexes beyond 2 leading-prefix columns
- Async / `object_store` sidecar I/O (codec-side is sync only today; the codec primitives compose into an async wrapper but `ParquetIndex::open` doesn't take an async reader yet)

All of the above are additive — adding them doesn't break the existing wire format.

## See also

- `docs/ematix-flow-integration.md` — concrete integration patterns for ematix-flow.
- `crates/ematix-iceberg/` — dataset-level layer (file-level prune before opening any sidecar).
- `crates/ematix-parquet-codec/examples/bench_indexed_lookup.rs` — reproducible perf numbers.
- Codec-side oracle tests under `crates/ematix-parquet-codec/tests/sidecar_*_oracle.rs` — every API exercised against full-scan baselines, bit-identical equivalence.
