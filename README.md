# ematix-parquet

A hand-rolled Apache Parquet implementation in Rust — both read and
write — optimised for AArch64 (NEON) and analytical query workloads.
The goal is a self-contained crate that competes with `parquet-rs` on
the shapes where most analytical queries actually live.

Sibling project to [ematix-flow](https://github.com/ryan-evans-git/ematix-flow);
designed to be a clean drop-in for high-throughput columnar scans and
a write path that produces files the rest of the ecosystem can read.

## Status

`v0.1.x` — v1.0 cut criteria are met (every Parquet shape we read
or write is covered, predicate pushdown lights up end-to-end,
TPC-H lineitem decode beats `parquet-rs` and `polars-parquet`).
API is settling but the write side still has a few rough edges
(per-column encoding choice on multi-column writes, smarter RLE
encoder); pin by SHA until we tag v1.0.

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

| Column        | Type        | Width | vs `parquet-rs` (SF=1) | vs `parquet-rs` (SF=10) | vs `polars-parquet` |
| ------------- | ----------- | ----- | ---------------------- | ----------------------- | ------------------- |
| `l_orderkey`  | INT64       | mixed | ~at parity / +7%       | ~at parity              | 75-99% faster       |
| `l_suppkey`   | INT64       | bw=14 | **+16% faster**        | **+48% faster (~2×)**   | 81-97% faster       |
| `l_shipdate`  | INT32       | bw=12 | **+11% faster**        | **+46% faster (~2×)**   | 85-99% faster       |
| `l_returnflag`| BYTE_ARRAY  | bw=2  | **+12% faster**        | **+13% faster**         | 83-99% faster       |

`bench_decode` is the harness — `cargo run --release --example bench_decode`
with `TPCH_DATA_DIR` pointing at a TPC-H lineitem.parquet.

**SIMD bit-unpackers (NEON, AArch64)** — every kernel hits the
~78 GB/s output ceiling on M-series. Specialised widths:

- bw=12: dates (`l_shipdate`, `l_commitdate`, `l_receiptdate`).
- bw=14: keys (`l_suppkey` 100%) — 16× the scalar baseline.
- bw=17: prices/keys (`l_extendedprice`, `l_partkey`, `l_orderkey`).

Each width has both a raw-indices kernel (for index-only consumers)
and a fused-gather lookup kernel that interleaves dict gather with
the unpack, eliminating an intermediate `Vec<u32>`.

**Scalar fallback** — const-generic per-bit-width unpacker covers
every bit-width from 1 to 32 on every target. The `unpack_chunks`
hot-loop uses raw pointer writes (capacity reserved by the caller),
so the scalar path is also free of `Vec::push`'s capacity check on
every value. Benefits from LLVM auto-vectorisation on widths that
align well.

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

Oracle tests against `parquet-rs` cover both directions on every
codec and every type. ~430 tests across 40+ test binaries. The
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

v1.0 cut criteria are met: every Parquet shape we read or write is
covered, predicate pushdown lights up end-to-end, and
TPC-H lineitem decode beats `parquet-rs` and `polars-parquet`. The
remaining roadmap items are performance polish, not correctness or
interop gaps:

| Item                                                                  | Status   |
| --------------------------------------------------------------------- | -------- |
| Additional NEON kernels for mid-range widths (1, 4, 5, 8, 18, 20, 21) | Open — performance polish; bw=12/14/17 already cover the dominant TPC-H widths |
| Smarter RLE encoder (run-coalescing on writes)                        | Open — current single-bit-packed-run output is spec-compliant; benchmark before optimising |
| Per-column encoding choice on `write_table_to_path_*`                 | Open — single-column dict entry points exist; multi-column needs a `ColumnEncoding` opt-in |
| Bloom-filter writer                                                   | Open — decoder ships; writer is the symmetric work |

**Not on the v1.0 path**: encrypted modules (out of scope), async /
object-store integration (the `io` crate is sync over `Read` —
async is its own design choice), x86 SIMD targets (scalar fallback
works on every target; NEON is the only hand-tuned path).

## Build

```
cargo build --release
cargo test
```

Requires AArch64 for the NEON kernels; on other targets the scalar
fallback is used and tests still pass.

## License

Apache-2.0.
