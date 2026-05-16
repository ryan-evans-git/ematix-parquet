# ematix-parquet — current plan

Tracks the work between v0.1.1 and v1.0. Phases use the existing `Π.N` convention
(Π.1 read façade, Π.2 write path, Π.3 codec/type completeness — all shipped).

## Phase map

| Phase | Theme                                                                | Status   |
| ----- | -------------------------------------------------------------------- | -------- |
| Π.4   | Write-side production parity (stats, multi-RG, dictionary encoding)  | ✓ Done   |
| Π.5   | Read-side predicate pushdown (page-index pruning, façade dispatch)   | ✓ Done   |
| Π.6   | DataPageV2 (read + write) + Bloom-filter decoder                     | ✓ Done   |
| Π.7   | BYTE_STREAM_SPLIT + first NEON wave (bw=14, gather opt, bw=17 lookup) | ✓ Done   |
| Π.8   | Performance polish: byte_array zero-copy, more NEON widths, RLE coalesce | ✓ Done |
| Π.9   | Photon-inspired: decode-into-buffer, predicate fusion, batched API   | ✓ Done   |

v1.0 cut criteria are already met (correctness, interop, beats
parquet-rs end-to-end on TPC-H lineitem). Π.8 is concrete performance
work to widen the lead — measured by `bench_decode` against parquet-rs
and polars on TPC-H lineitem at SF=1 and SF=10.

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

### Π.7 deferred items (subsequent work)

Picked up in Π.8b: bw=15, bw=16, bw=18 NEON kernels — all at
memory bandwidth.

Still open and explicitly out of scope: NEON kernels for widths
1, 4, 5, 8, 20, 21. The scalar const-generic fallback hits
~7-9 GB/s output for these — fast enough that the gather (not the
unpack) dominates on the columns where they appear in TPC-H
lineitem. Concrete trigger to revisit: a real-workload benchmark
that shows one of these widths dominating decode time.

---

## v1.0 cut criteria — MET

All four phases (Π.4, Π.5, Π.6, Π.7) are shipped:
- We write everything mainstream consumers expect (stats, multi-RG,
  dict, V1 + V2 page formats).
- Read-side predicate pushdown is wired (page-index, bloom).
- We read DataPageV2 (parquet-mr ≥ 1.13 / Spark 3.x default).
- BYTE_STREAM_SPLIT round-trips bit-exact.

Test coverage: 427 tests across the workspace, every passing.

---

## Π.8 — Performance polish ✓ DONE

The v1.0 cut criteria are met. Π.8 is concrete performance work to
widen the lead over `parquet-rs` and `polars-parquet`. Each sub-phase
ships behind a benchmark — no speculative SIMD or "should be faster"
claims. The harness is `bench_decode` against TPC-H lineitem at SF=1
and SF=10, with `parquet-rs` and `polars-parquet` as the cross-check.

### Sequencing

I'm reordering from how the items were originally listed, by
expected end-to-end ROI on the TPC-H lineitem benchmark:

| Order | Sub-phase | Why first / Why later |
| ----- | --------- | --------------------- |
| 1 | Π.8a — byte_array zero-copy gather | Biggest single read-perf win available. l_returnflag is 43-83 ms today, dominated by 1M tiny mallocs from `dict[i].clone()`. Estimated 5-20× speedup → drops byte_array decode from ~50 ms to ~3-10 ms. |
| 2 | Π.8b — Mid-range NEON widths (bw=15, 16, 18) | Mechanical work that mirrors bw=14/17. Modest end-to-end wins (5-10% on l_orderkey, l_partkey, l_extendedprice). Validates the NEON pattern is repeatable and doesn't require novel design. |
| 3 | Π.8c — Smarter RLE encoder (run-coalescing on writes) | Write-side only. File-size reduction on highly-repetitive dict-encoded columns. Measured by output bytes vs `parquet-rs`-written equivalent. Modest (5-15% size reduction on l_returnflag-class output). |

Small-width NEON (bw=1, 2, 3, 4, 6) is **explicitly deferred** —
it's not on the v1.1 path. The microbench shows scalar bw=1-6 at
0.47-0.61 ns/val (~7-9 GB/s output), which is already fast enough
that on l_returnflag the gather, not the unpack, dominates.

### Π.8a — byte_array zero-copy gather ✓ DONE

**Shipped:** new `read_column_byte_array_offsets(file, rg, col) ->
Result<(Vec<u8>, Vec<u32>)>` returning Arrow-style flat bytes +
N+1 offsets. The dict-gather hot loop validates indices in a
pre-pass then uses `ptr::copy_nonoverlapping` with raw pointer
writes — no per-row bounds check, no `Vec::push` capacity check,
no per-row allocator call.

**Bench result.** l_returnflag (1M rows × 1-byte values, dict-encoded)
went from 43 ms → **4.4 ms** — a 10× speedup, 91% faster than
parquet-rs, 60× faster than polars-parquet. The hypothesis that
1M `dict[i].clone()` mallocs were the bottleneck was correct.

6 oracle tests cover PLAIN + dict round-trips, agreement with the
existing `Vec<Vec<u8>>` entry point, single-byte (l_returnflag) shape,
empty column, variable-length values, and a parquet-rs-written file
cross-check.

### Π.8b — Mid-range NEON widths: bw=15, 16, 18 ✓ DONE

**Shipped:** three new kernel pairs (`unpack_indices_into_neon_bw{15,16,18}`
+ `unpack_lookup_into_neon_bw{15,16,18}`), wired into both
`unpack_indices_into` and `unpack_lookup_into` dispatchers in
`bitpack.rs`. Each kernel hits the ~76-96 GB/s output bandwidth
ceiling on M-series.

**Microbench.** Width / scalar / NEON / speedup:
- bw=15: 0.55 ms / 0.05 ms / **~10×**
- bw=16: 0.58 ms / 0.04 ms / **~14×** (byte-aligned, fastest)
- bw=18: 0.58 ms / 0.05 ms / **~10×**

**End-to-end.** No regression on the lineitem matrix; modest wins
on l_orderkey (bit-unpack was no longer the bottleneck after Π.7).
The kernels are in place for any workload that has these widths
dominating.

**Implementation note.** First attempt used
`vextq_u8(v0, vld1q_u8(src + 8), 7)` to construct a "bytes 7..23"
view for the high lanes. That LOOKS contiguous but actually returns
`v0[7..16] + v1[0..7]` = bytes 7..15 + bytes 8..14 (REPEATED). The
fix is `vld1q_u8(src + 7)` — direct load. Spotted by a microbench
assertion failure on bw=18 lane 7 (which needs byte 16/17
contributions to survive the mask).

### Π.8c — Smarter RLE encoder ✓ DONE

**Shipped:** `rle::encode_rle_bit_packed` with run-coalescing.
`write_single_column_dict` now uses it instead of the single-run
encoder. The reference single-run encoder stays in place for tests.

**Algorithm.** Walks the index stream identifying maximal value-runs.
Runs of length ≥ 8 emit RLE; shorter runs accumulate in a bit-pack
buffer that flushes (only the final flush is allowed to be
zero-padded; intermediate flushes borrow from the upcoming RLE run
to align to a multiple of 8).

**Size oracle results.**
- 100K rows × 10000-element runs (10 distinct values, bit_width=4):
  on-disk file < 4 KB. Bit-pack alone would produce ~50KB index
  stream; RLE coalescing collapses it to ~30 bytes.
- 30K cycling input (3 distinct, no runs ≥ 8): no regression vs
  single-run encoder.
- 50K mixed shape (80%-hot value with cold bursts): file < 2 KB.

**Implementation note — alignment via borrow.** A naive "pad with
zeros and emit" approach for intermediate flushes leaks padding
into the value stream because the decoder only stops at the global
`num_values` cap, not at end-of-run markers. The borrow-and-realign
trick avoids this: when an RLE-worthy run interrupts a non-aligned
bit-pack accumulator, take `8 - pending_mod` values from the run
to align the bit-pack flush, then RLE the residual. Spotted by a
unit test (`smart_ends_with_long_run`) that decoded one extra
zero where it shouldn't have.

### Out of scope for Π.8

- **Smaller-width NEON** (bw=1, 2, 3, 4, 6) — scalar already at
  ~7-9 GB/s output; adding NEON would help in narrow benchmarks but
  not on the lineitem decode matrix. Revisit if a real workload
  shows a small-width column dominating decode time.
- **Bloom-filter writer** — symmetric to the decoder, but no real
  reader will pay attention to a bloom filter on output until we have
  a story for who needs us to write one.
- **Per-column encoding choice on `write_table_*`** — would let
  callers mix PLAIN and dict per column. Useful but invasive (new
  `WriteOptions` shape). Land it when a real consumer asks.

---

## Π.9 — Photon-inspired performance work (active)

The Databricks Photon paper (SIGMOD 2022) and engineering blog
posts describe a vectorized C++ query engine. Most of its design
sits at the *engine* layer (what ematix-flow plays), but a coherent
subset — how individual column scans are organized — lives at our
layer. Π.9 lifts that subset.

**What we already have that mirrors Photon:** bytes+offsets layout
(Π.8a), page-index pruning (Π.5a), bloom filters (Π.6c), fused
predicate decode (existing bw=12 kernel), sparse gather, raw-pointer
hot loops, smart RLE writes (Π.8c). Those are explicit in the
Photon paper too.

**What's worth lifting:**

### Π.9a — Decode-into-caller-buffer API (active)

**Goal.** Eliminate the per-call `Vec` allocation in the read façade
by letting callers pass an `&mut Vec<T>` they reuse. Mirrors how
Photon writes directly into Arrow buffers.

**Touches.** `crates/ematix-parquet-codec/src/read.rs`. For each
existing `read_column_*` entry point, add an `_into` variant that
takes the output buffer; refactor the existing function to be a
thin wrapper that allocates and calls the `_into` version. Same
treatment for `read_column_byte_array_offsets`.

**Acceptance.**
1. `read_column_*_into(file, rg, col, &mut Vec<T>)` for every type
   we expose today (i32, i64, f64, byte_array, int96, flba +
   `byte_array_offsets`).
2. `read_column_*_with_range_into` for the page-pruning variants.
3. Calling `_into` twice with the same buffer overwrites on the
   second call (clear-then-fill semantics).
4. Existing `read_column_*` tests still pass (the wrappers must be
   byte-identical to the old direct implementations).
5. Oracle test: 10 successive reads of the same chunk into the same
   buffer produce 10 identical results, with no growth past the
   first allocation.
6. `bench_decode` adds an `ours_into` row to demonstrate the
   steady-state savings on a hot read path.

**Estimate.** 1 day. Pure refactor + thin wrappers + tests.

### Π.9b — Predicate fusion for more widths and types ✓ DONE

**Shipped:** width-generic predicate-fused decoder that mirrors the
bw=12 path across every NEON-specialized width (14, 15, 16, 17, 18)
plus a scalar fallback for non-specialized widths.

**Touches:**
- `bitpack_neon::decode_predicate_bitmap_neon_bw{14,15,16,17,18}` —
  per-width fused kernels. Each reuses the existing
  `unpack_neon_bw{N}_into_staging` helper and packs 8 dict-mask
  lookups into one bitmap byte per block via the shared
  `pack_predicate_byte` helper.
- `dict::decode_rle_dictionary_predicate_bitmap` — width-generic
  entry point. Reads bit_width from `body[0]`, validates
  `dict_mask.len() ≥ 1 << bit_width`, dispatches per-width.
- `dict::build_dict_predicate_mask` — helper that builds a
  `dict_mask` of the right size from a decoded dictionary and a
  predicate closure.
- `decode_rle_dictionary_predicate_bitmap_bw12` becomes a thin
  back-compat wrapper.

**Bench (microbench, 1M values, ~1% selectivity, M-series):**
| bw | baseline (unpack→Vec<u32>→bitmap) | fused | speedup |
| -- | --------------------------------- | ----- | ------- |
| 14 | 1.098 ms | 0.189 ms | **5.81×** |
| 15 | 1.103 ms | 0.190 ms | **5.82×** |
| 16 | 1.112 ms | 0.177 ms | **6.29×** |
| 17 | 1.210 ms | 0.207 ms | **5.84×** |
| 18 | 1.231 ms | 0.329 ms | **3.74×** |

All well above the ≥ 2× acceptance bar. 15 new oracle tests in
`tests/dict_predicate_bitmap_widths_oracle.rs` cover every width,
the scalar fallback (bw=13), unaligned tails, all-zero/all-one
masks, bw=0 degenerate, and `build_dict_predicate_mask` error
cases.

### Π.9c — Streaming / batched decode API ✓ DONE

**Shipped:** `read_column_*_batches(file, rg, col, batch_size)`
returning `ColumnBatchIter<T>` which implements
`Iterator<Item = Result<Vec<T>>>`. Walks a column chunk page-by-
page and emits batches of (mostly) `batch_size` rows; final batch
may be shorter.

**Implementation (`read::ColumnBatchIter<T, F>`):**
- Chunk bytes (compressed) live in the iterator (single allocation
  per chunk, matches non-streaming reads).
- Page walker is reconstructed per `fill_one_page` call against
  `&chunk_bytes[walker_pos..]` (avoids self-referential struct).
- Dict pages decode into a persistent `dict` field; subsequent
  RLE_DICTIONARY data pages decode against it.
- Decoded values land in `carry`; emitting a batch slices off
  `batch_size` and advances `carry_pos`. Compacts when fully drained.

**Memory bound.** `carry` peaks at `max(batch_size, page_values)`
plus the chunk_bytes buffer. The strict-batch-size bound isn't
achievable because parquet pages can be larger than batch_size
(typical: 20K values per page); we decode one full page at a time
and emit batches from it.

**Public entry points:** `read_column_{i64,i32,f64}_batches`.
BYTE_ARRAY batched API deferred — `Vec<u8>: !Copy` clashes with
the `decode_rle_dictionary_into<T: Copy>` bound, and the dict-
encoded path needs a separate index-then-gather-then-clone
strategy. Land when a real consumer asks.

**8 oracle tests in `tests/read_batches_oracle.rs`:** concat parity
across batch sizes 1..50_000, non-final batches always exactly
`batch_size`, `batch_size = 0` rejected, oversized batch → one
batch with everything, iterator stays `None` after exhaustion,
dict-encoded i64 streams correctly, multi-row-group: each RG
streams independently.

### Out of scope for Π.9

- **NEON prefetching in the dict gather** — `pld` instruction one
  cache line ahead. Worthwhile when dict > L1, but a focused
  benchmark first; defer until measured. Could be a quick Π.9d.
- **u8 dict indices when bw ≤ 8** — saves Vec<u32> overhead in
  the index stream. Small wins on l_returnflag-class columns;
  defer until a benchmark says it matters.
- **BYTE_ARRAY batched API** — needs `T: Clone` (not `Copy`) and
  a separate gather strategy on dict pages. Add when a consumer asks.
- **Custom LLVM codegen** — Photon does this; we don't have the
  infra and our const-generic monomorphisation gets ~80% of the
  benefit at zero infra cost.
- **NUMA awareness, work-stealing** — server concerns, not for a
  library.
- **Adaptive runtime dispatch on observed selectivity** — needs
  profiling infra; hard to make pay off in a per-call library.
