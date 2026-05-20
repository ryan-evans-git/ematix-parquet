# ematix-parquet

**A fast, dependency-light Apache Parquet codec in Rust — built for analytical workloads on modern CPUs.**

Hand-tuned SIMD on AArch64 (NEON) and x86_64 (AVX2). Full read and write coverage of the Parquet spec. Sync and async (object-store) façades. No C dependencies.

```toml
[dependencies]
ematix-parquet-codec = "0.12"
ematix-parquet-io    = "0.12"
```

## Why

- **Fast where it matters.** Dict-encoded numeric columns decode at 76–96 GB/s output through specialised SIMD kernels (raw-indices: bw=1, 2, 3, 4, 5, 8, 12, 14, 15, 16, 17, 18, 20, 21 on both NEON and AVX2; fused unpack + dict-gather: bw=4, 6, 8, 12, 14, 16, 17). Predicate-fused decode collapses unpack + filter + bitmap pack into a single pass — 3.7–6.3× faster than materialise-then-filter at low selectivity.
- **Complete on read and write.** Every physical type, every encoding (PLAIN, dict, DELTA_BINARY_PACKED, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT), every mainstream codec (Snappy, Zstd, Gzip, Brotli, LZ4_RAW), V1 + V2 pages, page indexes, bloom filters, Parquet Modular Encryption.
- **Light footprint.** The sync read/write stack has no third-party deps beyond the chosen compression codecs. Async, encryption, and parallel decode are opt-in features that pull deps only when you ask for them.
- **Built for engines.** Decode-into-caller-buffer APIs, late-materialization (`*_masked_into`), Arrow-style `(bytes, offsets)` BYTE_ARRAY shape, dict-preserving readers for direct `DictionaryArray` construction, streaming batched decode, adaptive runtime dispatch on observed selectivity, and parallel multi-row-group decode with NUMA-aware worker pinning on Linux.

## Quick start

**Reading.**

```rust
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_io::ParquetFile;

let file = ParquetFile::open("data.parquet")?;
let values: Vec<i64> = read_column_i64(&file, 0, 0)?;
```

**Writing.**

```rust
use ematix_parquet_codec::write::{write_table_to_path, ColumnData};
use ematix_parquet_format::types::CompressionCodec;

let cols = vec![
    ("id",   ColumnData::I64(&[1, 2, 3])),
    ("name", ColumnData::ByteArray(&[b"a" as &[u8], b"bb", b"ccc"])),
];
write_table_to_path("out.parquet", &cols, CompressionCodec::Snappy)?;
```

**Async (over S3, GCS, Azure, or any `object_store`).**

```rust
use std::sync::Arc;
use object_store::local::LocalFileSystem;
use ematix_parquet_async::{AsyncParquetFile, read_column_i64_async};

let store = Arc::new(LocalFileSystem::new());
let file = AsyncParquetFile::open(store, "data.parquet".into()).await?;
let values: Vec<i64> = read_column_i64_async(&file, 0, 0).await?;
```

## Performance

End-to-end column decode on TPC-H lineitem (Apple Silicon, median of 12 iters):

| Column                        | Type       | Width | Per-call |
| ----------------------------- | ---------- | ----- | -------- |
| `l_orderkey`                  | INT64      | mixed | ~3.8 ms  |
| `l_suppkey`                   | INT64      | bw=14 | ~0.8 ms  |
| `l_shipdate`                  | INT32      | bw=12 | ~0.8 ms  |
| `l_returnflag` (offsets API)  | BYTE_ARRAY | bw=2  | ~2.7 ms  |

SIMD unpackers hit the ~76–96 GB/s output ceiling on every specialised width — full hand-tuned coverage on both NEON (M-series) and AVX2 (x86_64).

The byte_array offsets API (`read_column_byte_array_offsets_into`) returns Arrow-style flat bytes + offsets and skips the per-row `Vec` allocation that dominates the standard `Vec<Vec<u8>>` path. Pair it with late-materialization (`*_masked_into`) for selective scans:

```rust
let mask = build_packed_mask(num_rows, |i| predicate(i));
read_column_byte_array_offsets_masked_into(
    &file, rg, col, &mask, &mut bytes, &mut offsets,
)?;
```

## Feature highlights

**Predicate-fused decode.** `decode_rle_dictionary_predicate_bitmap(body, n, dict_mask, out)` walks the index stream, gathers the dict, and packs the result bitmap in one SIMD pass. Build the mask with `build_dict_predicate_mask(&dict, bw, predicate)`. Rows that fail the predicate never materialise.

**Adaptive dispatch.** `read_column_*_predicate_adaptive` probes the first few pages of a chunk, measures selectivity, and decides per-chunk whether to emit a bitmap (wins at low selectivity) or a values vector (wins at high). Optional telemetry hook exposes the dispatch decision.

**Parallel decode.** `read_columns_parallel(file, &targets, opts, decode_one)` (enable the `parallel` feature) decodes a slice of `(row_group, column)` targets concurrently over rayon, with cooperative cancellation. On Linux a `parallel::numa` submodule adds topology detection, per-thread worker pinning, and first-touch local-buffer allocation — no `libnuma` C dep.

**Async streaming.** Every read façade has an `_async` sibling, plus `_async_stream` returning `Stream<Item = Result<Vec<T>>>`. Cold-open issues ≤ 2 round trips via the trailing-bytes footer trick.

**Per-column write options.** `write_table_with_options_to_path(path, columns, &WriteOptions { ... })` bundles row-group size, page version, default codec, plus per-column slices for codec, dict-encoding opt-in, and bloom-filter target FPP. Different columns in the same row group can use different codecs.

**Buffer reuse on the hot path.** `ParquetFile::read_range_into(&mut Vec<u8>, offset, length)` lets callers pre-allocate one chunk buffer per row group and reuse it across column reads — eliminates the per-call alloc + zero-fill that dominates profiles of scan-heavy workloads.

**Parquet Modular Encryption.** AES-GCM read + write for both PME modes (plaintext footer / encrypted footer) behind a default-off `encryption` feature.

## Crate layout

| Crate                    | Purpose                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| `ematix-parquet-format`  | Thrift compact-protocol + Parquet metadata types. No deps beyond `std`. |
| `ematix-parquet-io`      | `ParquetFile` (pread-based lock-free byte-range reads) + `PageWalker`.  |
| `ematix-parquet-codec`   | Decoders, encoders, SIMD bit-unpackers, compression, read/write façades. Optional `parallel` (rayon) and `encryption` (AES-GCM) features. |
| `ematix-parquet-crypto`  | AES-GCM primitives + AAD construction for PME.                          |
| `ematix-parquet-async`   | Async façade over any `object_store::ObjectStore` (S3, GCS, Azure, …).  |

## Testing

~700 tests across 50+ test binaries. Oracle tests round-trip every codec × type combination against an independent Parquet implementation in both directions — anything we write, a reference reader reads; anything a reference writer writes, we read. Plus unit tests on bit-unpack, RLE, predicate fusion, page-index parsing, compact-protocol primitives, and the encrypted code paths.

## Status

v1.0 cut criteria are met: every Parquet shape covered, predicate pushdown end-to-end, decode hot paths SIMD-tuned on both architectures. API is settling; pin by version range until v1.0. See [crates.io](https://crates.io/crates/ematix-parquet-codec) for the latest version and [`docs/plans/CURRENT.md`](docs/plans/CURRENT.md) for the per-phase history and what's still open.

## Build

```
cargo build --release
cargo test
```

NEON kernels engage on AArch64; AVX2 kernels engage on x86_64 with AVX2 support detected at run time. Other targets fall back to the const-generic scalar unpacker, which auto-vectorises well on widths that align.

## License

Apache-2.0.
