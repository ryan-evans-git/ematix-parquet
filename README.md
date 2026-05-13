# ematix-parquet

A hand-rolled Apache Parquet implementation in Rust — both read and
write — optimised for AArch64 (NEON) and analytical query workloads.
The goal is a self-contained crate that competes with `parquet-rs` on
the shapes where most analytical queries actually live.

Sibling project to [ematix-flow](https://github.com/ryan-evans-git/ematix-flow);
designed to be a clean drop-in for high-throughput columnar scans and
a write path that produces files the rest of the ecosystem can read.

## Status

`v0.0.x` — pre-1.0, API still moving. The standard read+write path is
feature-complete for every Parquet shape covered below; the missing
items are listed explicitly in the roadmap. Downstream consumers
should pin by SHA.

## What's implemented

### Read path

**Physical types** — INT32, INT64, INT96, FLOAT, DOUBLE, BOOLEAN,
BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY. Every Parquet primitive.

**Encodings** — PLAIN, PLAIN_DICTIONARY, RLE_DICTIONARY (the
RLE + bit-pack hybrid for dictionary indices and definition /
repetition levels), DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,
DELTA_BYTE_ARRAY. BYTE_STREAM_SPLIT is still open.

**Compression** — Snappy (hand-rolled fast path + `snap` fallback),
Zstd, Gzip, Brotli, LZ4_RAW, Uncompressed. The complete set of
mainstream codecs.

**Metadata** — Thrift compact-protocol parse of file footer, row
groups, column chunks, page headers, schema (logical + converted
types), statistics (min / max / null count). Page index + column
index parsed; pruning is wired but only fires when stats are tight
enough to skip pages.

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

**SIMD bit-unpackers (NEON, AArch64)**

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

Oracle tests against `parquet-rs` cover both directions on every
codec and every type. ~200 tests across 30+ test binaries. The
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

Aiming for these in v1.0+:

| Item                                                               | Status   |
| ------------------------------------------------------------------ | -------- |
| **Dictionary encoding on writes**                                  | Open     |
| **Multi row-group writes** (writer currently emits one row group)  | Open     |
| **Statistics on writes** (min / max / null_count on column chunks) | Open     |
| **DataPageV2 read + write**                                        | Open     |
| BYTE_STREAM_SPLIT encoding (rare in practice)                       | Open     |
| Bloom-filter decoder                                                | Open     |
| Page-index pruning wired into the read façade (parsed today)        | Open     |
| NEON kernels for mid-range widths (1, 4, 5, 8, 18, 20, 21)          | Open     |
| INT96 / FLBA dispatch in the high-level façade (low-level done)     | Open     |

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
