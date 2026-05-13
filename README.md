# ematix-parquet

A hand-rolled Parquet decoder in Rust, built for AArch64 (NEON) and
focused on **fused predicate evaluation** — the decode loop produces a
match-bitmap in the same pass that unpacks bit-packed values, instead
of materialising every value and filtering afterwards.

Sibling project to [ematix-flow](https://github.com/ryan-evans-git/ematix-flow);
designed to be a clean drop-in alternative to `parquet-rs` for
analytical workloads where most pages are predicate-filtered before
they reach a downstream aggregator.

## Status

Experimental. API is moving. Pinned by SHA from downstream consumers
until the bridge surface settles.

Coverage today:

- **Reader**: file footer + row-group + column-chunk + page-header parse
  (Thrift compact protocol, hand-written varint readers).
- **Decoders**: PLAIN, RLE/bit-packed hybrid (definition + repetition
  levels and dictionary indices), DELTA_BINARY_PACKED,
  DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY.
- **Compression**: Snappy, Zstd, Uncompressed.
- **Bit-unpackers**: NEON kernels for bw=12 (date-shaped columns) and
  bw=17 (price/key-shaped columns), with a generic scalar fallback for
  all other widths.
- **Predicate-fused decode**: `decode_rle_dictionary_predicate_bitmap_bw12`
  performs RLE-walk + dict-index gather + dictionary value comparison +
  bitmap pack in a single NEON loop.
- **Sparse gather**: `gather_dict_at_bitmap_into<T>` reads only the
  dictionary entries selected by an upstream bitmap, with 8-row
  bitmap-byte skip for cold selectivity.
- **Page index / column index**: parsed; pruning is wired but
  workload-dependent (uniform-distribution columns won't benefit).

Not implemented:

- Writers. This is a read path only.
- DELTA_LENGTH_BYTE_ARRAY writes, BYTE_STREAM_SPLIT, BROTLI/LZ4_RAW,
  encrypted modules, Bloom filters.
- Non-NEON SIMD targets (x86 AVX2/AVX-512). Scalar fallback works on
  any target.

## Crates

The workspace publishes three crates with sharp boundaries:

| Crate                    | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `ematix-parquet-format`  | Thrift compact-protocol decoder + Parquet metadata types |
| `ematix-parquet-io`      | File / page-header reading on top of `std::io::Read`   |
| `ematix-parquet-codec`   | Column decoders, bit-unpackers, compression, NEON kernels |

`format` has no dependencies beyond `std`. `io` depends on `format`.
`codec` depends on `format` (for types) but not `io` (so it can be
fed bytes from any source — mmap, network, in-memory test buffer).

## Using it

Right now, as a git dependency:

```toml
[dependencies]
ematix-parquet-codec = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-format = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
ematix-parquet-io = { git = "https://github.com/ryan-evans-git/ematix-parquet.git", rev = "<sha>" }
```

Pin to a specific SHA — the API is not yet stable. crates.io
publishing is planned once the surface settles.

The canonical integration example is `ematix-flow`'s
[`ematix_parquet_bridge.rs`](https://github.com/ryan-evans-git/ematix-flow/blob/main/crates/ematix-flow-core/src/ematix_parquet_bridge.rs),
which produces Arrow `RecordBatch`es from ematix-parquet column chunks
without going through `parquet-rs`'s `ParquetRecordBatchReader`.

## Why another Parquet library

`parquet-rs` is mature, complete, and the right default. ematix-parquet
exists for one specific shape of workload: **predicate-heavy analytical
scans on AArch64 where the predicate is cheap relative to dictionary
materialisation**. The Q14 lever in ematix-flow ran:

- parquet-rs (decode → arrow → filter): ~31 ms
- ematix-parquet (fused decode-and-filter, sparse gather): ~14.6 ms
  end-to-end on TPC-H SF=1 lineitem (~6M rows, ~1.26% selectivity)

The win comes from never materialising the 98.74% of values that the
predicate is going to throw out. There's no magic — it's just that a
fused loop avoids two passes over the data and lets the bitmap pack
inline with the unpack.

## Build

```
cargo build --release
cargo test
```

Requires AArch64 for the NEON kernels; on other targets the scalar
fallback is used.

## License

Apache-2.0.
