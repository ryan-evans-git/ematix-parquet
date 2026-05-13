# ematix-parquet

A hand-rolled Apache Parquet decoder in Rust, optimised for AArch64
(NEON) and analytical query workloads. Goal: a self-contained read
(soon also write) path that competes with `parquet-rs` on the shapes
where most analytical queries actually live.

Sibling project to [ematix-flow](https://github.com/ryan-evans-git/ematix-flow);
designed to be a clean drop-in alternative for high-throughput
columnar scans.

## Status

`v0.0.1` — pre-1.0, API still moving. Read path is feature-complete
for the most common Parquet shapes; write path is the headline open
item for v1.0. Downstream consumers should pin by SHA.

## What's implemented

### Read path

**Physical types** — INT32, INT64, FLOAT, DOUBLE, BOOLEAN, BYTE_ARRAY
(INT96 and FIXED_LEN_BYTE_ARRAY parsed in metadata but not yet
decoded).

**Encodings** — PLAIN, PLAIN_DICTIONARY, RLE_DICTIONARY (with the RLE +
bit-pack hybrid for dictionary indices and definition / repetition
levels), DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY.

**Compression** — Snappy (hand-rolled fast path + `snap` fallback),
Zstd, Uncompressed. Gzip / Brotli / LZ4 are stubs for v1.0.

**Metadata** — Thrift compact-protocol parse of file footer, row
groups, column chunks, page headers, schema (logical + converted
types), statistics (min / max / null count). Page index + column
index parsed; pruning is wired but only fires when stats are tight
enough to skip pages.

### Performance

**SIMD bit-unpackers (NEON, AArch64)** —
- bw=12: hot for date columns (`l_shipdate`, `l_commitdate`,
  `l_receiptdate`) — ~10× the scalar baseline.
- bw=17: hot for price / key columns (`l_extendedprice`,
  `l_partkey`, `l_orderkey`) — ~11× on the dominant gather width.

**Scalar fallback** — const-generic per-bit-width macro-unrolled
unpacker covers every bit-width from 1 to 32 on every target.
Compiles to per-width inlined loops; benefits from LLVM
auto-vectorisation on a wide range of widths even without hand-rolled
SIMD.

**Predicate-fused decode** —
`decode_rle_dictionary_predicate_bitmap_bw12` performs the RLE
walk + dict-index gather + dictionary-value comparison + bitmap
pack in a single NEON loop. The point is to never materialise the
~98% of values a selective predicate is going to throw away.

**Sparse gather** — `gather_dict_at_bitmap_into<T>` reads only the
dictionary entries selected by an upstream bitmap, with an 8-row
bitmap-byte skip for cold selectivity. Pairs with the fused decode
to keep the slow path off the hot loop.

### Test coverage

Oracle tests compare every decoder against `parquet-rs` on real
TPC-H SF=1 lineitem.parquet (and synthetic round-trips written by
`parquet-rs`):

- `lineitem_full_column_oracle` — INT64 full column-chunk parity
- `lineitem_dict_oracle` — dictionary-encoded column parity
- `lineitem_byte_array_oracle` — BYTE_ARRAY parity
- `lineitem_multi_column_oracle` — Q14-shape end-to-end parity
- `delta_real_file_oracle` — `parquet-rs` writes DELTA_BINARY_PACKED →
  ours reads
- `zstd_real_file_oracle` — `parquet-rs` writes Zstd → ours reads
- `page_index_real_file_oracle` — column-index range selection parity

Plus ~20 unit tests across compression, dict, delta, bitpack, levels,
page index, predicate fusion.

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
| `ematix-parquet-format`  | Thrift compact-protocol decoder + Parquet metadata types |
| `ematix-parquet-io`      | File / page-header reading on top of `std::io::Read`   |
| `ematix-parquet-codec`   | Column decoders, bit-unpackers, compression, NEON kernels |

`format` has no deps beyond `std`. `io` depends on `format`. `codec`
depends on `format` (for types) but not `io` — so codec can be fed
bytes from any source (mmap, network, in-memory test buffer).

## Using it

As a git dependency (pin to a SHA; the API is moving):

```toml
[dependencies]
ematix-parquet-codec  = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-format = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-io     = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
```

Today the public surface is low-level: `ParquetFile::open` → iterate
row groups → iterate column chunks → `PageWalker` over the chunk →
match encoding → decompress → decode. A higher-level façade
(`read_column<T>(file, rg, col) -> Iter<T>`) lands in v1.0.

The canonical integration example is ematix-flow's
[`ematix_parquet_bridge.rs`](https://github.com/ryan-evans-git/ematix-flow/blob/main/crates/ematix-flow-core/src/ematix_parquet_bridge.rs),
which produces Arrow `RecordBatch`es from ematix-parquet column chunks
without going through `parquet-rs`'s `ParquetRecordBatchReader`.

## Roadmap to v1.0

| Item                                                               | Status   |
| ------------------------------------------------------------------ | -------- |
| Read: numeric + bool + byte_array, PLAIN + dict + DELTA, Snappy + Zstd | Done     |
| NEON kernels for hot bit-widths (12, 17)                           | Done     |
| Oracle test suite vs `parquet-rs` on real TPC-H lineitem            | Done     |
| Predicate-fused decode + sparse gather                             | Done     |
| **High-level `read_column<T>` façade**                             | Open     |
| **Write path (PLAIN + dict + Snappy + footer write)**              | Open     |
| Round-trip oracle (ours writes → ours reads, parity vs `parquet-rs`) | Open     |
| GZIP / Brotli / LZ4 / LZ4_RAW compression                          | Open     |
| INT96 + FIXED_LEN_BYTE_ARRAY decoders                              | Open     |
| BYTE_STREAM_SPLIT encoding                                         | Open     |
| Bloom-filter decoder                                               | Open     |
| NEON kernels for mid-range widths (1, 4, 5, 8, 18, 20, 21)          | Open     |

Not on the v1.0 path: encrypted modules, async / object-store
integration (the `io` crate is sync over `Read`), x86 SIMD targets
(scalar fallback works on any target; NEON is the only hand-tuned path).

## Build

```
cargo build --release
cargo test
```

Requires AArch64 for the NEON kernels; on other targets the scalar
fallback is used and tests still pass.

## License

Apache-2.0.
