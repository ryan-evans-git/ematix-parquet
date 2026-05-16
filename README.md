# ematix-parquet

A hand-rolled Apache Parquet implementation in Rust — both read and
write — optimised for AArch64 (NEON) and analytical query workloads.
The goal is a self-contained crate that competes with `parquet-rs` on
the shapes where most analytical queries actually live.

Sibling project to [ematix-flow](https://github.com/ryan-evans-git/ematix-flow);
designed to be a clean drop-in for high-throughput columnar scans and
a write path that produces files the rest of the ecosystem can read.

## Status

`v0.3.x` — v1.0 cut criteria are met (every Parquet shape we read
or write is covered, predicate pushdown lights up end-to-end,
TPC-H lineitem decode beats `parquet-rs` and `polars-parquet`).
The v0.2 cycle landed Photon-inspired analytical hot paths
(decode-into-caller-buffer, width-generic predicate fusion across
every NEON kernel, streaming batched-decode iterator). The v0.3
cycle adds **late-materialization** —
`read_column_*_masked_into(file, rg, col, &mask, &mut out)` for
scalar types + byte_array, composing the Π.9 primitives with
per-page popcount-skip — so consumers can decode only matching
rows after a filter pass instead of full-decode-then-filter.
API is settling but the write side still has a few rough edges
(per-column encoding choice on multi-column writes); pin by SHA
or version range until we tag v1.0.

## What's implemented

### Read path

**Physical types** — INT32, INT64, INT96, FLOAT, DOUBLE, BOOLEAN,
BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY. Every Parquet primitive.

**Encodings** — PLAIN, PLAIN_DICTIONARY, RLE_DICTIONARY (the
RLE + bit-pack hybrid for dictionary indices and definition /
repetition levels), DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,
DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT. Every encoding the spec
defines.

**Compression** — Snappy (hand-rolled fast path + `snap` fallback),
Zstd, Gzip, Brotli, LZ4_RAW, Uncompressed. The complete set of
mainstream codecs.

**Metadata** — Thrift compact-protocol parse of file footer, row
groups, column chunks, page headers (V1 + V2), schema (logical +
converted types), statistics (min / max / null count). Page index
+ column index parsed and wired into the read façade via
`read_column_{i32,i64}_with_range(file, rg, col, lo, hi)` —
pages whose `[min, max]` doesn't overlap `[lo, hi]` are skipped
without decompression. Bloom-filter decoder (Split-Block + XXHash64)
ships too.

### Write path

**Single-column writers** — one entry point per scalar type, two
shapes per codec (default-uncompressed + explicit codec):

```rust
ematix_parquet_codec::write::write_i64_column_to_path(path, name, values);
ematix_parquet_codec::write::write_i32_column_to_path_with_codec(
    path, name, values, CompressionCodec::Snappy,
);
```

Variants exist for `i32`, `i64`, `f64`, `bool`, `byte_array`. All
emit PLAIN-encoded REQUIRED columns inside a single row group.

**Multi-column writer** — for tables with mixed types:

```rust
let columns = vec![
    ("id",    ColumnData::I64(&ids)),
    ("price", ColumnData::F64(&prices)),
    ("name",  ColumnData::ByteArray(&name_refs)),
];
write_table_to_path(path, &columns, CompressionCodec::Snappy)?;
```

All five mainstream codecs (Snappy / Zstd / Gzip / Brotli / LZ4_RAW)
are wired on the write side. Round-trip oracles validate every
codec × type combination against `parquet-rs`.

### Performance

**End-to-end TPC-H lineitem decode** (Apple Silicon, median of 12 iters):

| Column                       | Type        | Width | vs `parquet-rs` (SF=1)  | vs `polars-parquet` |
| ---------------------------- | ----------- | ----- | ----------------------- | ------------------- |
| `l_orderkey`                 | INT64       | mixed | +4-7% faster            | 72-99% faster       |
| `l_suppkey`                  | INT64       | bw=14 | **+17% faster**         | 88-97% faster       |
| `l_shipdate`                 | INT32       | bw=12 | **+12% faster**         | 90-99% faster       |
| `l_returnflag` (`Vec<Vec<u8>>`)| BYTE_ARRAY | bw=2  | **+7-12% faster**       | 83-99% faster       |
| `l_returnflag` (offsets API) | BYTE_ARRAY  | bw=2  | **+91% faster (10×)**   | **98-99% faster (60×)** |

`bench_decode` is the harness — `cargo run --release --example bench_decode`
with `TPCH_DATA_DIR` pointing at a TPC-H lineitem.parquet.

The byte_array "offsets API" (`read_column_byte_array_offsets`)
returns Arrow-style flat bytes + offsets and skips the per-row
`Vec` allocation that dominates the standard `Vec<Vec<u8>>` path.
Use it for any consumer that doesn't need owned per-row Vecs.

**SIMD bit-unpackers (NEON, AArch64)** — every specialised kernel
hits the ~76-96 GB/s output ceiling on M-series. Covered widths:

- **bw=12**: dates (`l_shipdate`, `l_commitdate`, `l_receiptdate`).
- **bw=14**: keys (`l_suppkey` 100%) — 16× the scalar baseline.
- **bw=15, 16, 18**: tail of the price/key columns. bw=16 is byte-
  aligned and is the fastest kernel of all (~96 GB/s output).
- **bw=17**: prices/keys (`l_extendedprice`, `l_partkey`,
  `l_orderkey`) — the dominant gather width.

Each width has both a raw-indices kernel (for index-only consumers)
and a fused-gather lookup kernel that interleaves dict gather with
the unpack, eliminating an intermediate `Vec<u32>`.

**Scalar fallback** — const-generic per-bit-width unpacker covers
every bit-width from 1 to 32 on every target. The `unpack_chunks`
hot-loop uses raw pointer writes (capacity reserved by the caller),
so the scalar path is also free of `Vec::push`'s capacity check on
every value. Benefits from LLVM auto-vectorisation on widths that
align well.

**Predicate-fused decode** (width-generic since v0.2) —
`decode_rle_dictionary_predicate_bitmap(body, n, dict_mask, out)`
performs the RLE walk + dict-index gather + bitmap pack in a
single pass. NEON-fused for bw ∈ {12, 14, 15, 16, 17, 18}; scalar
fallback for the rest. Build the mask with
`build_dict_predicate_mask(&dict, bw, predicate)` and feed it to
the decoder — the matching bitmap drops out the other end without
ever materialising the values that fail the predicate. At Q14-
shape selectivity (~1%) this is 3.7–6.3× faster than the
materialise-then-filter baseline across every width.

**Sparse gather** — `gather_dict_at_bitmap_into<T>` reads only the
dictionary entries selected by an upstream bitmap, with an 8-row
bitmap-byte skip for cold selectivity. Pairs with the fused decode
to keep the slow path off the hot loop.

**Smart RLE/bit-pack encoder on writes** — dict-encoded columns
emit RLE runs for repeated indices (length ≥ 8) instead of always
bit-packing. Borrow-and-realign keeps intermediate bit-pack runs
on multiples of 8 without leaking padding into the value stream.
For long-run-dominated input (e.g., a sorted dimension key
repeated thousands of times), the index stream shrinks 10-100×.

**Decode into caller buffer** (v0.2) — every `read_column_*` entry
point has a `_into(&mut Vec<T>)` sibling that reuses caller-owned
memory across calls. Steady-state savings on hot read paths: on
TPC-H lineitem `l_suppkey` the per-call cost drops from 1.51 ms
(allocating) to 0.82 ms (`_into`) = **46% faster**; on
`l_returnflag` byte_array-offsets from 3.26 ms → 2.70 ms.

**Streaming batched decode** (v0.2) —
`read_column_{i64,i32,f64}_batches(file, rg, col, batch_size)`
returns an iterator that emits `Vec<T>` in batches sized to the
caller's preference (typically Arrow's RecordBatch size). Lets
the engine pipeline (process batch N while we decode batch N+1)
and bounds working-set memory for huge row groups.

**Late-materialization read façade** (v0.3) — after a filter pass
produces a packed mask, decode only the matching rows instead of
full-decode-then-filter. Available for scalar types and byte_array
(both `Vec<Vec<u8>>` and Arrow-style `(bytes, offsets)` shapes):

```rust
let mask = build_packed_mask(num_rows, |i| predicate(i));
read_column_i64_masked_into(file, rg, col, &mask, &mut out)?;
read_column_byte_array_offsets_masked_into(file, rg, col, &mask,
                                           &mut bytes, &mut offsets)?;
```

Composes Π.9's primitives — `gather_dict_at_bitmap_into` for
dict-encoded pages, new `plain_sparse_decode_*` for PLAIN — with
a per-page popcount-skip that drops fully-dead pages without
decompression. End-to-end Q14 codec-layer measurement on TPC-H
lineitem SF=1: late-mat 13.26 ms vs baseline 14.03 ms (5.5%
faster at 1.4% selectivity). The bigger ~2 ms Polars-gap win
lives at the engine layer where Arrow construction is skipped
on filtered rows.

### Test coverage

Oracle tests against `parquet-rs` cover both directions on every
codec and every type. ~495 tests across 40+ test binaries. The
matrix:

**Read side** — `parquet-rs` writes files we then decode and
diff value-by-value:

- TPC-H SF=1 lineitem column oracles (i64 / dict / byte_array /
  Q14-shape multi-column / page-index range selection)
- DELTA_BINARY_PACKED, DELTA_BYTE_ARRAY round-trip
- Gzip / Brotli / LZ4_RAW / Zstd round-trip
- INT96, FIXED_LEN_BYTE_ARRAY PLAIN decode
- High-level façade: i32 / i64 / f64 / byte_array against
  TPC-H lineitem

**Write side** — we write the file and `parquet-rs` reads it
back, plus a symmetric check via our own reader:

- PLAIN i64 / i32 / f64 / bool / byte_array round-trip
- Snappy / Zstd / Gzip / Brotli / LZ4_RAW round-trip per codec
- Multi-column mixed-type table
- Empty-column + i64-extremes edge cases
- Compression-shrinks-file check (catches "codec field set but body
  uncompressed" bugs that round-trip would miss)

Plus ~30 unit tests on compact-protocol writer primitives,
page-header writers, file-footer writers, bit-pack, RLE, predicate
fusion, page-index parsing.

### Benchmarks

Under `crates/ematix-parquet-codec/examples/`:

| Example              | What it shows                                                           |
| -------------------- | ----------------------------------------------------------------------- |
| `bench_decode`       | End-to-end column decode vs `parquet-rs` and `polars` for i64/i32/utf8 |
| `bench_late_mat`     | Q14 predicate via late materialisation: ~14.6 ms end-to-end (TPC-H SF=1) |
| `bench_unpack`       | Bit-unpack microbench across widths 1–32 (drives SIMD ROI calls)        |
| `bench_snappy`       | Snappy decompression vs `snap` crate                                    |
| `probe_bitwidths`    | Reports actual bit-widths per column on a given Parquet file            |

The Q14 win comes from the fused decode path: a single NEON loop
unpacks + filters + packs a bitmap, then a sparse gather visits only
the matching rows. Same techniques apply to any selective scan, not
just Q14 — but Q14 is what we've benchmarked end-to-end so far.

## Crate layout

Three crates with sharp boundaries:

| Crate                    | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `ematix-parquet-format`  | Thrift compact-protocol reader + writer + Parquet metadata types |
| `ematix-parquet-io`      | File / page-header reading on top of `std::io::Read`   |
| `ematix-parquet-codec`   | Column decoders + encoders, bit-unpackers, compression, NEON kernels, high-level read/write façades |

`format` has no deps beyond `std`. `io` depends on `format`. `codec`
depends on both — its low-level decoders are byte-slice based and
don't pull in `io`, but the high-level read/write façades do.

## Using it

As a git dependency (pin to a SHA; the API is moving):

```toml
[dependencies]
ematix-parquet-codec  = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-format = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-io     = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
```

**Reading** — high-level façade collapses the page-walk boilerplate:

```rust
use ematix_parquet_codec::read::read_column_i64;
use ematix_parquet_io::ParquetFile;

let file = ParquetFile::open("data.parquet")?;
let values: Vec<i64> = read_column_i64(&file, 0, 0)?;
```

**Writing** — single-call entry points per type:

```rust
use ematix_parquet_codec::write::{write_table_to_path, ColumnData};
use ematix_parquet_format::types::CompressionCodec;

let cols = vec![
    ("id",   ColumnData::I64(&[1, 2, 3])),
    ("name", ColumnData::ByteArray(&[b"a" as &[u8], b"bb", b"ccc"])),
];
write_table_to_path("out.parquet", &cols, CompressionCodec::Snappy)?;
```

For zero-copy or per-page access (Arrow integration, predicate-fused
decode, custom dictionary handling), drop down to the low-level
decoders directly. `ematix-flow`'s
[`ematix_parquet_bridge.rs`](https://github.com/ryan-evans-git/ematix-flow/blob/main/crates/ematix-flow-core/src/ematix_parquet_bridge.rs)
is the canonical example — it produces Arrow `RecordBatch`es from
ematix-parquet column chunks without going through `parquet-rs`'s
`ParquetRecordBatchReader`.

## What's still open

v1.0 cut criteria for the codec are met: every Parquet shape we
read or write is covered, predicate pushdown lights up end-to-end,
and TPC-H lineitem decode beats `parquet-rs` and `polars-parquet`
on Apple Silicon.

The road from v0.2 to v2.0 broadens platform coverage and pushes
the perf ceiling further — **see `docs/plans/CURRENT.md`** for
the full Π.10 – Π.15 phase definitions. In summary:

| Phase | Theme                                                              |
| ----- | ------------------------------------------------------------------ |
| Π.10  | Async / object-store integration (S3 / GCS / Azure)                |
| Π.11  | x86 SIMD parity (AVX2 / AVX-512 kernels mirroring NEON)            |
| Π.12  | Parquet Modular Encryption (read + write, AES-GCM)                 |
| Π.13  | Adaptive runtime dispatch on observed selectivity                  |
| Π.14  | NUMA awareness + work-stealing for multi-RG parallel decode        |
| Π.15  | Custom LLVM codegen for hot decode paths (Photon-style, speculative) |

Sequencing rationale: async (Π.10) and x86 (Π.11) come first
because they unlock real consumer deployments (cloud storage and
Linux server hardware). Encryption (Π.12) lands once those are in
so the crypto work benefits the broadest deployment surface.
Π.13 – Π.14 are server-class polish; Π.15 is the most speculative
item and may never ship if const-generic monomorphization keeps
covering the workloads we see.

Smaller items picked up opportunistically (also in the plan doc):

| Item                                                                  | Status   |
| --------------------------------------------------------------------- | -------- |
| NEON prefetching in dict gather (Π.9d)                                | Open — pld one cache line ahead; benchmark first |
| u8 dict indices when bw ≤ 8                                            | Open — saves Vec<u32> overhead on l_returnflag-class columns |
| BYTE_ARRAY batched API                                                 | Open — needs `T: Clone`-based gather |
| Per-column encoding choice on `write_table_*`                          | Open — single-column dict exists; multi-column needs `WriteOptions` |
| Bloom-filter writer                                                    | Open — decoder ships; writer is the symmetric work |
| Additional NEON kernels for small widths (1, 4, 5, 8, 20, 21)         | Open — scalar already at ~7-9 GB/s; gather dominates |
| DELTA_BINARY_PACKED u64-output unpacker                                | Open (TODO in `delta.rs`) |

## Build

```
cargo build --release
cargo test
```

Requires AArch64 for the NEON kernels; on other targets the scalar
fallback is used and tests still pass.

## License

Apache-2.0.
