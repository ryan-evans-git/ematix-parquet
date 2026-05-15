# ematix-parquet — current plan

Tracks the work between v0.1.1 and v1.0. Phases use the existing `Π.N` convention
(Π.1 read façade, Π.2 write path, Π.3 codec/type completeness — all shipped).

## Phase map

| Phase | Theme                                                                | Status   |
| ----- | -------------------------------------------------------------------- | -------- |
| Π.4   | Write-side production parity (stats, multi-RG, dictionary encoding)  | ✓ Done   |
| Π.5   | Read-side predicate pushdown (page-index pruning, façade dispatch)   | ✓ Done   |
| Π.6   | DataPageV2 (read + write) + Bloom-filter decoder                     | ✓ Done   |
| Π.7   | BYTE_STREAM_SPLIT (NEON kernel coverage deferred to v1.1)            | ✓ Done   |

The v1.0 cut criteria are met. Remaining open item — NEON kernels for
mid-range widths — is performance polish, not a correctness or interop
gap; deferred to v1.1.

---

## Π.4 — Write-side production parity (active)

The reader handles every Parquet shape. The writer only emits PLAIN-encoded
single-row-group files with no statistics. Π.4 closes that gap so files we
produce are first-class for every downstream consumer.

### Π.4a — Statistics on writes ✓ DONE

**Goal:** every column chunk we emit carries `min`, `max`, and `null_count`
in `ColumnMetaData.statistics`, so downstream predicate pushdown actually
fires on our files.

**Shipped** in this iteration:
- `Statistics` Thrift writer in `metadata_writer::encode_statistics` —
  emits all eight fields (deprecated `min`/`max` + modern
  `min_value`/`max_value` + `is_min/max_value_exact` flags + `null_count`).
- Per-type stats accumulators (`stats_i32`, `stats_i64`, `stats_f64`,
  `stats_bool`, `stats_byte_array`) in `codec/src/write.rs`. f64 path
  excludes NaN and applies the +0.0/-0.0 normalisation the spec
  requires.
- Stats land on both the page header and the column-chunk metadata,
  matching what parquet-rs emits.
- 12 new oracle tests in `tests/write_statistics_oracle.rs` — parquet-rs
  reads back the stats we emit on every type; the multi-column writer
  computes per-column stats independently; pushdown decisions on
  out-of-range predicates use our stats correctly.

**Touches:**
- `crates/ematix-parquet-codec/src/write.rs` — accumulate stats while encoding each column.
- `crates/ematix-parquet-format/src/metadata_writer.rs` — emit the `Statistics` Thrift struct on `ColumnMetaData`.
- `crates/ematix-parquet-format/src/metadata.rs` — confirm `Statistics` shape matches what the spec/parquet-rs expects on the wire.

**Per-type stat semantics** (matches Parquet spec):
- INT32, INT64, FLOAT, DOUBLE — `min`/`max` are the type's native byte
  representation (LE). Skip NaN for floats.
- BOOLEAN — `min`/`max` over `false < true`.
- BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY — unsigned lexicographic.
- INT96 — leave stats off (spec says no stats for INT96).
- `null_count` — always emit (we know it; no nulls today, but plumbing matters).

**Acceptance:**
1. Round-trip oracle: write a column, read with `parquet-rs`, assert
   `column_chunk.metadata().statistics()` returns the expected min/max/null_count.
2. Predicate pushdown lights up: write a sorted i64 column, use
   `parquet-rs` `RowFilter` with a range predicate, assert pruned row count.
3. Empty column still emits `null_count = 0` and no min/max.

**Estimate:** 2-3 days.

### Π.4b — Multi row-group writes ✓ DONE

**Goal:** caller can choose row-group boundaries. Stats are computed per
row group so reader-side pruning can drop individual groups.

**Shipped** in this iteration:
- New public entry points `write_table_to_path_with_row_group_size` and
  `write_table_with_row_group_size` in `codec/src/write.rs`. Existing
  `write_table_to_path` / `write_table` are thin wrappers that pass
  `usize::MAX` (preserves single-RG behaviour — no breaking change).
- `ColumnData::slice(range)` for zero-copy sub-range views; each row
  group encodes from a sliced column.
- Row-group walk yields per-RG `PreparedColumn` descriptors with
  per-RG stats and per-RG `num_values`. Empty-input contract (one
  empty row group) preserved.
- `RowGroup.ordinal` populated on the wire.
- 9 oracle tests in `tests/write_multi_row_group_oracle.rs` — 10×100
  shape, uneven last RG, end-to-end value round-trip via parquet-rs,
  per-RG stats tight on sorted column, our reader walks every RG,
  oversized `row_group_size` collapses to one RG, multi-column +
  multi-RG per-column stats, default entry point still single-RG,
  `row_group_size = 0` rejected.

**Deferred to Π.4c (or later):**
- Auto-default for row group size based on byte budget — current
  default stays at `usize::MAX` (one big RG) for backward compatibility.
  Callers opt into multi-RG explicitly. Adding a byte-budget default
  pairs naturally with dictionary encoding work in Π.4c.
- Per-type single-column entry points still emit a single row group;
  the multi-RG knob is on `write_table_*` only. Wrapping single-column
  in a multi-RG path is a one-line refactor when a caller needs it.

### Π.4c — Dictionary encoding on writes ✓ DONE

**Goal:** writer can emit `PLAIN_DICTIONARY` (data page) + dictionary
page for low-cardinality columns.

**Shipped** in this iteration:
- RLE/bit-pack hybrid encoder in `codec/src/rle.rs`:
  `encode_rle_bit_packed_single_run` + `min_bit_width_for_dict`.
  MVP emits a single bit-packed run; the decoder consumes it
  identically to the multi-run form parquet-rs emits. 11 unit
  tests round-trip via the existing decoder at every bit width.
- Dictionary builders per type
  (`build_dict_i64`/`build_dict_i32`/`build_dict_f64`/`build_dict_byte_array`)
  preserve first-occurrence order. f64 keys on bit pattern so NaN
  collapses sanely.
- Single-column dict entry points per type:
  `write_i64_column_dict_to_path` / `_i32_` / `_f64_` /
  `_byte_array_`. Each emits a Dictionary page + a
  `PLAIN_DICTIONARY` data page and sets
  `ColumnMetaData.dictionary_page_offset` + encodings list.
- 9 oracle tests in `tests/write_dict_oracle.rs` — round-trip via
  parquet-rs on every type, round-trip via our own reader, file
  metadata advertises PLAIN_DICTIONARY, dict file shrinks to
  <half-size of plain for 10 distinct strings × 10k rows, and the
  single-distinct-value (bit_width=0) edge case round-trips.

**Deferred to follow-ups:**
- Multi-column dict (`write_table_*` with per-column encoding choice).
  Adding a `ColumnEncoding` enum on the multi-column writer is a
  clean follow-up; single-column dict is sufficient to prove the
  encoder works and unblocks the common low-cardinality use case.
- Smarter encoder that coalesces repeated indices into RLE runs.
  The current single-bit-packed-run encoder is spec-compliant; RLE
  coalescing is a size optimisation worth a focused benchmark
  before designing.
- Auto-detection of when to dict-encode (caller decides for now via
  the entry-point name).
- Bool dict (only two values; the bit-packed PLAIN encoding is
  already optimal).

---

## Π.5 — Read-side predicate pushdown ✓ DONE

### Π.5a — page-index pruning in the read façade ✓

- `read_column_i32_with_range` / `read_column_i64_with_range` in
  `codec/src/read.rs`. They load the column index for `(rg, col)`,
  compute a per-data-page mask via `select_pages_overlapping_*`, and
  skip pruned pages without decompressing or decoding them.
- Falls back to a full chunk read when no column index is present.
- 5 oracle tests in `tests/read_with_range_oracle.rs`: superset
  recovery via caller-side filter, pruning actually drops pages,
  predicate above max drops most pages, i64 round-trip, fallback
  works on files we wrote (which don't carry a column index).

### Π.5b — INT96 + FLBA façade dispatch ✓

- `read_column_int96` returns `Vec<plain::Int96>`.
- `read_column_flba` returns `Vec<Vec<u8>>`; `type_length` is read
  from the schema element.
- 3 oracle tests in `tests/read_int96_flba_facade_oracle.rs`:
  INT96 round-trip via parquet-rs, 16-byte FLBA (UUID width) round-
  trip, 5-byte FLBA (DECIMAL width) round-trip.

---

## Π.6 — DataPageV2 + Bloom filter ✓ DONE

### Π.6a — DataPageV2 read ✓

- New `data_page_view` helper in `codec/src/read.rs` abstracts V1 vs
  V2 page-body layout differences. For V1 the whole body is one
  compressed unit; for V2 the rep+def prefixes are uncompressed and
  only the values portion is decompressed (gated by `is_compressed`).
- All four façade entry points (`read_column_i32`/`_i64`/`_f64`/
  `_byte_array` plus `_int96`/`_flba`) now dispatch both V1 and V2.
- 3 oracle tests in `tests/read_data_page_v2_oracle.rs`: i32
  uncompressed, i64 Snappy, byte_array uncompressed — all written
  via parquet-rs with `WriterVersion::PARQUET_2_0`.

### Π.6b — DataPageV2 write ✓

- `format/src/metadata_writer.rs`: `encode_data_page_header_v2`
  emits all eight V2 fields. Two roundtrip unit tests (minimal +
  uncompressed-with-stats).
- `codec/src/write.rs`: `PageVersion` enum + `write_table_to_path_v2`
  entry point. Internal `write_table_inner` is parameterized so V1
  and V2 share a single code path (existing single-RG and multi-RG
  entry points stay V1).
- 4 oracle tests in `tests/write_data_page_v2_oracle.rs`: i64
  round-trip via parquet-rs, byte_array Snappy round-trip,
  multi-RG self-roundtrip via our reader, page-type sanity check
  on the wire.

### Π.6c — Bloom-filter decoder ✓

- `format/src/metadata.rs`: `BloomFilterHeader` + tagged-union
  enums (`BloomFilterAlgorithm`, `BloomFilterHash`,
  `BloomFilterCompression`) + `read_bloom_filter_header`.
- `codec/src/bloom.rs`: `SplitBlockBloomFilter` view over a borrowed
  bitset, `contains_hash` / `contains_bytes`. XXHash64 (seed 0)
  implemented inline (no new deps); validated against canonical
  xxhsum reference vectors.
- 3 oracle tests in `tests/bloom_oracle.rs` against parquet-rs-written
  bloom filters: i64 + byte_array — every present value reports
  present (no false negatives), false-positive rate stays under
  5% on out-of-set probes.

---

## Π.7 — Polish ✓ DONE (NEON kernel work deferred to v1.1)

### BYTE_STREAM_SPLIT encoding ✓

- `codec/src/byte_stream_split.rs`: `decode_byte_stream_split_f32`/
  `_f64` + symmetric encoders. 7 unit tests round-trip f32 and f64
  through the encoder→decoder path including extremes (NaN, ±0.0,
  MIN/MAX, EPSILON) and reject malformed byte counts.
- 2 oracle tests in `tests/byte_stream_split_oracle.rs`: parquet-rs
  writes a BYTE_STREAM_SPLIT-encoded file, our decoder reads the
  page body and asserts bit-exact recovery via `to_bits()`.

### Deferred to v1.1: NEON kernels for widths 1, 4, 5, 8, 18, 20, 21

Each width is its own focused NEON-engineering effort against a
benchmark — the existing bw=12 and bw=17 kernels are 561 lines of
hand-tuned NEON between them. The scalar const-generic fallback is
correct for every width and is auto-vectorised by LLVM at many of
them; absent profiling that points at a specific width as the
bottleneck, more NEON kernels are speculative work. Concrete
trigger to revisit: a benchmark on a real workload that shows a
specific width dominating decode time on AArch64.

---

## v1.0 cut criteria — MET

All four phases (Π.4, Π.5, Π.6, Π.7) are shipped:
- We write everything mainstream consumers expect (stats, multi-RG,
  dict, V1 + V2 page formats).
- Read-side predicate pushdown is wired (page-index, bloom).
- We read DataPageV2 (parquet-mr ≥ 1.13 / Spark 3.x default).
- BYTE_STREAM_SPLIT round-trips bit-exact.

Test coverage: 427 tests across the workspace, every passing.

The remaining roadmap item — NEON kernels for additional widths —
is an open performance project, not a correctness gap.
