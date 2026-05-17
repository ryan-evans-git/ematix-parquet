# ematix-parquet — current plan

Tracks the work between v0.1.1 and v2.0. Phases use the existing `Π.N` convention
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
| Π.10  | Late-materialization read façade (`read_column_*_masked_into`)       | ✓ Done (codec; engine in ematix-flow)  |
| Π.11  | Async / object-store integration (S3 / GCS / Azure)                  | ✓ Done (a–d, f; e deferred to v0.4.1) |
| Π.12  | x86 SIMD parity (AVX2 / AVX-512 kernels mirroring NEON)              | ✓ Done   |
| Π.13  | Parquet Modular Encryption (read + write)                            | ✓ Done   |
| Π.14  | Adaptive runtime dispatch on observed selectivity                    | ✓ Done   |
| Π.15  | NUMA awareness and work-stealing for multi-RG parallel decode        | ✓ Done   |
| Π.16  | Custom LLVM codegen for hot decode paths (Photon-style)              | Speculative |

v1.0 cut criteria are already met (correctness, interop, beats
parquet-rs end-to-end on TPC-H lineitem). v0.1–v0.2 shipped the
Apple-Silicon-first, sync-IO codec; Π.10 (the next ship, sized
~1.5 weeks because every building block already exists) closes a
measured 2 ms Q14 gap to Polars via late-materialization. v0.4 and
beyond broaden the platform footprint (Π.11 async, Π.12 x86,
Π.13 encryption) and push the performance ceiling further on
workloads where const-generic monomorphization stops being enough
(Π.14–Π.16).

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

### Out of scope for Π.9 (deferred to later phases or follow-ups)

- **NEON prefetching in the dict gather** — `pld` instruction one
  cache line ahead. Worthwhile when dict > L1; defer until measured.
  Could be a quick Π.9d follow-up.
- **u8 dict indices when bw ≤ 8** — saves Vec<u32> overhead in
  the index stream. Small wins on l_returnflag-class columns;
  defer until a benchmark says it matters.
- **BYTE_ARRAY batched API** — needs `T: Clone` (not `Copy`) and
  a separate gather strategy on dict pages. Add when a consumer asks.

The bigger structural items (async I/O, x86 SIMD, encryption,
adaptive dispatch, NUMA, custom codegen) graduated from "out of
scope" to **Π.11 through Π.16** — see below.

---

## Π.10 — Late-materialization read façade (active)

**Goal.** Close a measured ~2 ms Q14 gap to Polars by adding
top-level "decode a column with a row-mask" entry points. Today
ematix-parquet decodes every column in full, then applies the
filter — Polars decodes the filter column first, builds a mask,
then sparse-decodes only matching rows in the other columns
(99% of the decode work skipped at Q14's ~1.4% selectivity).

See `docs/plans/PI-10-late-mat-design.md` for the full design.

### Π.10a — read_column_*_masked_into for scalar types ✓ DONE

**Shipped.**
- `read::read_column_{i32,i64,f64}_masked_into(file, rg, col, &mask,
  &mut out)` — appends matched values to caller's buffer (does NOT
  clear). Mask is a packed bitmap (`&[u8]`, 1 bit per chunk row).
- `read::build_packed_mask(num_rows, |i| pred(i)) -> Vec<u8>` — helper.
- `plain::plain_sparse_decode_{i32,i64,f64}_into` — 8-row block-skip
  + per-lane gather. Mirrors `gather_dict_at_bitmap_into`.
- Internal `decode_chunk_row_masked_into` walker: per-page popcount-
  skip; dict pages dispatch to `gather_dict_at_bitmap_into`;
  PLAIN pages use the new sparse-decode primitives.
- 12 oracle tests in `tests/read_masked_oracle.rs`:
  selectivity sweep (0/0.1/1/10/50/100%) per type × encoding;
  append semantics; multi-page mask transitions; per-page popcount-
  skip edge case; undersized/empty/full mask edges.

### Π.10b — read_column_byte_array_masked_into + offsets variant ✓ DONE

**Shipped.**
- `read::read_column_byte_array_masked_into(.., &mut Vec<Vec<u8>>)` —
  one allocation per matched value.
- `read::read_column_byte_array_offsets_masked_into(.., &mut bytes,
  &mut offsets)` — Arrow-style flat-bytes + N+1 offsets, zero
  malloc per row. Multi-call concatenation: continues offsets from
  the previous trailing value (doesn't re-push leading 0).
- `plain::plain_sparse_decode_byte_array_into` + offsets variant —
  walks length-prefixes sequentially (variable-length forces this).
- Relaxed `dict::gather_dict_at_bitmap_into` bound from `T: Copy` →
  `T: Clone`. Strictly more permissive (Copy: Clone); for existing
  Copy callers `.clone()` inlines to a trivial copy — no perf cost.
  Enables `Vec<u8>` to flow through the dict path.
- 8 oracle tests in `tests/read_byte_array_masked_oracle.rs`:
  selectivity sweep × shapes; multi-call concatenation; edge cases.

### Π.10c — Q14 bench + ematix-flow bridge integration ✓ codec-side DONE

**Shipped (codec side).**
- `examples/bench_q14_late_mat.rs` — end-to-end Q14 bench using
  the new façade. Compares baseline (4× full decode + filter)
  vs late-mat (decode shipdate → mask → 3× masked decode).
- Measured on TPC-H lineitem SF=1, row-group 0 (~6M rows,
  ~84K matches @ 1.4% selectivity):
  - **baseline (4× full decode + filter)**: 14.03 ms median
  - **late-mat (façade _masked_into)**:    13.26 ms median
  - **5.5% faster (1.06×)**

**Why "only" 5.5% at the codec layer.** The chunk-bytes I/O +
Snappy decompression cost is the same in both paths (we
decompress every page either way; per-page popcount-skip can't
help at uniform selectivity because Q14's matches are spread
across every page). The win is bounded by per-row output
materialization savings: ~98.6% × 3 columns × 8 bytes ≈ ~480 KB
fewer output writes. Higher-selectivity workloads see bigger
codec-layer wins; the larger Q14 wins live at the engine layer
(Arrow construction skipped on filtered rows).

**Remaining for v0.3.0 tag.**
- ematix-flow bridge integration (in the **ematix-flow** repo,
  not this one). Bridge orders the decode behind a `with_late_mat`
  builder flag on `FastParquetTableProvider` for A/B testing.
- Full TPC-H sweep in ematix-flow for 0 regressions.
- Tag v0.3.0 from `main` after the codec-side PRs (#4, #5, #6)
  merge and the ematix-flow integration confirms end-to-end Q14
  ≤ 13.0 ms.

**Acceptance status.**
1. ✓ Bit-identical equivalence vs full-decode-then-filter across
   types × encodings × selectivities (20 oracle tests, all green).
2. ⏳ End-to-end Q14 ≤ 13.0 ms — measured in ematix-flow once the
   bridge integration lands.
3. ⏳ 0 regressions on TPC-H sweep — same.

---

## Π.11 — Async / object-store integration ✓ DONE

Ships as **v0.4.0**. New crate `ematix-parquet-async` exposes
`AsyncParquetFile` over any `object_store::ObjectStore` plus
async siblings for every scalar + byte_array read façade entry
point, including streaming `Stream<Item = Result<Vec<T>>>`.

Sync `ematix-parquet-io` is unchanged and dep-free. The async
crate isolates tokio + object_store.

Full design in `docs/plans/PI-11-async-design.md`.

### Π.11a — AsyncParquetFile primitive ✓ DONE

- `crates/ematix-parquet-async/src/file.rs`: `AsyncParquetFile`
  with `open`, `metadata()`, `read_range()`. Cold-open via the
  suffix-range trick (≤ 2 round trips); per-call `read_range`
  issues one GET, returns `bytes::Bytes` zero-copy where the
  store supports it.
- Pinned to `object_store 0.11`; bump-and-fix chore every 6-12
  months. Re-exports the `ObjectStore` trait so consumers don't
  double-import.
- 7 oracle tests + 1 doc-test against the `LocalFileSystem`
  ObjectStore for parity with sync `ParquetFile`.

### Π.11b — Async façade for scalar types ✓ DONE

- `crates/ematix-parquet-async/src/read.rs`:
  `read_column_{i32,i64,f64}_async{,_into}` — six entries.
- Internal `decode_chunk_into` mirrors the codec's sync chunk-
  walker. Async surface is one `.await` per chunk; after that
  the page walk + decode is sync over in-memory bytes.
- 8 oracle tests vs sync `read_column_*`.

### Π.11c — Async façade for byte_array ✓ DONE

- `read_column_byte_array_async{,_into}` →  `Vec<Vec<u8>>`.
- `read_column_byte_array_offsets_async{,_into}` →
  Arrow-style flat `(bytes, offsets)`. Multi-call concatenation
  continues offsets from the previous trailing value.
- 5 oracle tests.

### Π.11d — Async streaming ✓ DONE

- `read_column_{i32,i64,f64}_async_stream(file, rg, col, batch_size)`
  → `impl Stream<Item = Result<Vec<T>>>`.
- Built on `async_stream::try_stream!` + `futures-core`.
  Internally: one async GET for the chunk, then yields owned
  `Vec<T>` in `batch_size`-sized slices until exhausted.
- 6 oracle tests (concat parity, batch-size sweep, edge cases).

### Π.11f — Bench + docs + release wiring ✓ DONE

- `examples/bench_decode_async.rs`: timed sync vs async i64
  decode on TPC-H lineitem (LocalFileSystem ObjectStore).
  Measured on SF=1 `l_suppkey` (1M rows, dict bw=14):
  sync 0.696 ms vs async 0.762 ms = **1.09× (+9.4%)**.
  The 9% gap is honest `object_store` + `tokio::spawn_blocking`
  overhead on local FS. For cloud workloads, network latency
  dwarfs this; for raw local-file throughput, sync stays
  faster.
- README — 4-crate layout, Async section in Performance.
- `release.yml` + `docs/RELEASING.md` — 4-crate publish order
  (`format → io → codec → async`).

### Π.11e — S3 integration tests (deferred to v0.4.1)

Needs `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars
and a public-read TPC-H S3 bucket. Doesn't gate v0.4.0; ship
as a follow-up once a nightly CI workflow with AWS secrets is
wired up.

---

## Π.12 — x86 SIMD parity ✓ DONE

**Goal.** Hand-tuned AVX2 bit-unpack + dict-lookup kernels mirroring
the NEON wave (bw=12, 14, 15, 16, 17, 18). Keeps the SIMD lead on
Linux x86 deployments (the majority of analytical infra) — where the
scalar fallback would lose the 10× advantage that matters most for
production throughput.

**Shipped** across six sub-phases:

| Sub-phase | Width | PR | Shape |
| --- | :---: | :---: | --- |
| Π.12-Phase0 | dispatch + CI matrix | #14 | runtime `is_x86_feature_detected!("avx2")`, ubuntu-latest in CI matrix alongside macos-14 |
| Π.12a | 16 | (rolled into Phase0) | `_mm256_cvtepu16_epi32` widen, byte-aligned shape |
| Π.12b | 14 | #16 | single `_mm_loadu_si128`, two halves, `_mm_srlv_epi32`, mask 0x3FFF |
| Π.12c | 15 | #17 | two halves at +0/+7, asymmetric shifts `[0,7,6,5]`/`[4,3,2,1]`, mask 0x7FFF |
| Π.12d | 17 | #21 | two halves at +0/+8, shared shuffle, symmetric shifts `[0..3]`/`[4..7]`, mask 0x1FFFF |
| Π.12e | 18 | #22 | two halves at +0/+7, different shuffles, symmetric shifts `[0,2,4,6]`, mask 0x3FFFF |
| Π.12f | 12 | #23 | single load, two halves from one v0, shifts `[0,4,0,4]`, mask 0xFFF |

Plus during the phase, three correctness/hygiene PRs landed:

- #18 — `gather_dict_at_bitmap_into` misaligned-bitmap-offset fix (real correctness bug in the sparse-gather fast path)
- #19 — workspace-wide `cargo fmt --all`
- #20 — workspace-wide clippy cleanup (workspace `[lints.clippy]` policy, `is_multiple_of` rollback for MSRV 1.80)

**Touches** (final state).
- `crates/ematix-parquet-codec/src/bitpack_avx2.rs` — six pub fns × 2 (indices + lookup) + matching `_unchecked` and `_into_staging` helpers, all `#[cfg(target_arch = "x86_64")]`.
- `crates/ematix-parquet-codec/src/bitpack.rs` — both `unpack_indices_into` and `unpack_lookup_into` dispatch all six widths to AVX2 on x86_64 with runtime feature detection.
- `crates/ematix-parquet-codec/tests/bitpack_avx2_unit.rs` — 5 unit tests per width (aligned / tail / max-values / lookup-vs-dict / cross-check vs scalar dispatcher).
- `.github/workflows/ci.yml` — matrix over `[macos-14, ubuntu-latest]`; AVX2 kernels actually execute on the linux runner.

**Acceptance** — landed:

1. ✓ AVX2 kernels for bw=12/14/15/16/17/18, both `unpack_indices_into` and `unpack_lookup_into` paths.
2. ✓ Correctness validated bit-for-bit against the scalar dispatcher (30 x86-gated unit tests).
3. ✓ Scalar fallback unchanged on hosts without AVX2 (cfg-gated module).
4. ✓ aarch64 NEON path untouched; 526 tests pass on every supported target.

**Predicate-fused AVX2** intentionally deferred to a later phase — the
fused NEON kernels (Π.9b) shipped first because Q14 selectivity drives
the design; the AVX2 mirror is mechanical once we measure the
selectivity profile on a real x86 host and confirm the same shape pays
off there.

**Released as v0.5.0.**

---

## Π.13 — Parquet Modular Encryption ✓ DONE

See `PI-13-encryption-design.md` for the design and full sub-phase
breakdown. Shipped as v0.6.0.

**Goal.** Read and write files using Parquet Modular Encryption (PME)
— the spec-defined per-column-chunk AES-GCM encryption used in
regulated industries (finance, healthcare, government).

**Shipped.**
- **Π.13a** — encryption metadata Thrift extensions (read + write)
  in `ematix-parquet-format`: `EncryptionAlgorithm`,
  `AesGcmV1` / `AesGcmCtrV1`, `ColumnCryptoMetaData`,
  `FileCryptoMetaData`.
- **Π.13b** — new `ematix-parquet-crypto` crate: AES-GCM primitives
  (`seal` / `open` via `aes-gcm`), AAD construction
  (`build_module_aad`), `Key`, `NonceSource` trait + `OsRng`-backed
  default, `KeyRetriever` trait. Validated against NIST AES-GCM
  test vectors.
- **Π.13c** — per-page decrypt primitive (`decrypt_module`) and
  `ColumnDecryptContext`. Wired through the read façade behind
  `--features encryption`.
- **Π.13d** — encrypted-footer (PARE magic) read path: trailer
  parsing, `decrypt_footer`, end-to-end read of files produced by
  the upstream Rust Parquet writer.
- **Π.13e** — plaintext-footer (PAR1) write path: per-page seal,
  28-byte trailing footer signature, façade entry point
  `write_i32_column_to_path_encrypted`. Round-trips through the
  upstream Rust Parquet reader.
- **Π.13f** — encrypted-footer (PARE magic) write path: emits the
  `FileCryptoMetaData` trailer + encrypted-FileMetaData wire frame.
  `examples/key_rotation.rs` demos end-to-end key rotation
  (write under OLD_KEY → decrypt + decode → re-write under NEW_KEY
  → confirm OLD_KEY no longer decrypts).
- **Π.13g** — release wiring: 5-crate publish order in
  `release.yml` (`format → io → crypto → codec → async`), CI
  matrix exercises `--features encryption` on every PR, version
  bump 0.5.0 → 0.6.0.

**Acceptance — all met.**
1. ✓ Round-trip oracle: the upstream Rust Parquet writer emits a
   PME-encrypted file; we read it back and recover every value.
2. ✓ Inverse: we write PME-encrypted files; the upstream Rust
   Parquet reader reads them back identically.
3. ✓ Both encryption modes (encrypted footer, plaintext footer)
   covered by oracle tests on read + write.
4. ✓ Key-rotation flow demonstrated by
   `examples/key_rotation.rs` (runnable from the workspace).
5. ✓ Crypto code path is `#[cfg(feature = "encryption")]`-gated on
   `ematix-parquet-codec`; default build pulls no crypto deps
   (verified by `cargo tree`).

**Out of scope (followups).** `AES_GCM_CTR_V1` algorithm,
encrypted page-index / encrypted bloom filter, encrypted async
reads exercised end-to-end, KMS integrations, footer-only mode
where data pages stay plaintext.

**Released as v0.6.0.**

---

## Π.14 — Adaptive runtime dispatch on observed selectivity ✓ DONE

Shipped as v0.8.0. The fused-predicate path (Π.9b) is 2-3× faster
than materialise-then-filter at low selectivity (bitmap output, no
gather); the materialise path is competitive or faster at high
selectivity (most rows pass anyway, bitmap-pack adds overhead).
The codec now picks per-chunk based on selectivity observed from
the first N pages.

**Shipped.**
- **Π.14a** — `adaptive` module primitive: `Dispatch`,
  `PageProbe`, `AdaptiveDictPredicate` (threshold + probe budget),
  `popcount_bitmap_prefix`, `probe_page_fused`.
- **Π.14b** — `run_adaptive_dict_chunk<T: Copy>` multi-page runner:
  probe-phase decode via fused → aggregate selectivity → commit
  to `Fused` (bitmap) or `Materialized` (values) → emit phase. On
  `Materialized` the probed pages are re-decoded (bounded cost,
  ≤ probe_pages × page size). Optional per-chunk telemetry
  callback `Fn(SelectivityProbe)`.
- **Π.14c** — `examples/bench_adaptive_dispatch.rs`: 7-point
  selectivity sweep on TPC-H SF1 lineitem `l_shipdate` (1M rows,
  bw=12). Tunes `DEFAULT_THRESHOLD = 0.10`, baked into the
  docstring as in-code reference.
- **Π.14d** — extended oracle (`tests/adaptive_chunk_runner_extended.rs`):
  width coverage (bw ∈ {14, 16, 18}), `probe_pages` edge cases,
  custom threshold override, mid-chunk selectivity-shift dispatch.
- **Π.14e** — per-type façade: `read_column_{i32,i64,f64}_predicate_adaptive`.
  Predicate is evaluated against dict entries (≤ dict.len()
  invocations per chunk); returns `AdaptiveChunkOutput<T>`.
- **Π.14f** — release wiring: version 0.7.0 → 0.8.0; README +
  CURRENT.md updates.

**Acceptance — met / known caveats.**
1. ✓ Π.14c bench numbers committed in code. Adaptive is within
   ~10% of the better static path at every sweep point except
   the 50% mid-range (16% slower than f+gather there — a
   workload-specific cache/allocator artifact where the static
   matr curve dips below static f+gather; the runner cannot
   know that without an actual run-and-compare). 4 of 7 sweep
   points are within 5%; all are correct.
2. ✓ Q14-shape (~1%): adaptive 0.28 ms vs static fused 0.27 ms —
   <5% overhead from the per-chunk probe.
3. ✓ Telemetry callback fires once per chunk with the right
   `SelectivityProbe` (tested at both runner-level and façade-
   level).
4. ✓ Adaptive Fused output is byte-identical to the static
   page-by-page reference across every width tested.

**Constraints.**
- Dict-only entry points (chunks with PLAIN data pages return
  `InvalidInput` — caller falls back to `read_column_*_masked_into`).
- BYTE_ARRAY not covered (T: Copy doesn't hold) — follow-up if
  asked.
- Bitmap-consuming callers (filter chain, COUNT aggregator) should
  stay on the static `decode_rle_dictionary_predicate_bitmap`
  entry point — fused always wins for that output shape and the
  adaptive runner adds dispatch overhead.

**Released as v0.8.0.**

---

## Π.15 — NUMA awareness and work-stealing parallel decode ✓ DONE

Shipped as v0.9.0. Multi-row-group files are embarrassingly
parallel (each RG is independent); Π.15 lifts the threading + NUMA
awareness into the codec so consumers don't have to roll it.

**Shipped.**
- **Π.15a** — `parallel::read_columns_parallel<T, F>(file, &targets,
  opts, decode_one)` over rayon. Generic over caller closure so the
  same primitive handles homogeneous + heterogeneous workloads.
  Output preserves input order; first error short-circuits. New
  `parallel` feature on `ematix-parquet-codec` (rayon optional;
  default builds stay rayon-free).
- **Π.15b** — `CancellationToken` (AtomicBool, `Arc`-cloneable).
  Cooperative: checked at target boundaries. Cancelled targets
  surface as new `CodecError::Cancelled`; in-flight decodes run to
  completion.
- **Π.15c** — `parallel::numa` (Linux-only via `cfg`).
  `NumaTopology::detect` via sysfs, `pin_current_thread_to_node`
  via `sched_setaffinity`, `build_numa_pinned_pool` (rayon pool
  with round-robin worker pinning). New `libc` dep is
  Linux-target-gated. macOS / Windows compile with the `numa`
  module absent.
- **Π.15d** — `current_node()` (via `getcpu(2)`) and
  `alloc_local_buffer(size)` (4 KiB-stride first-touch). Combined
  with Π.15c worker pinning, chunk-bytes buffers land on the
  correct NUMA node via Linux's first-touch policy — no `libnuma`
  C dep, no `mbind(2)`.
- **Π.15e** — `examples/bench_parallel_scaling.rs`. Synthetic 50-RG
  Snappy-compressed i64 file, sweeps thread counts, reports
  speedup + efficiency vs sequential. Linux variant also exercises
  the NUMA-pinned pool. Local M-series (14 cores, single NUMA
  node) peaks at 1.82× speedup at N=8.
- **Π.15f** — release wiring: 0.8.0 → 0.9.0; README + plan doc
  updates; CI matrix exercises `--features
  ematix-parquet-codec/parallel` on both runners.

**Acceptance — met locally; multi-socket validation deferred.**
1. **Linear-to-socket-count scaling on 2-socket Linux** —
   instrumented (Π.15e bench) but not yet executed on a multi-
   socket box (blocked on AWS infra, tracked alongside Π.14g).
   Bench harness will drop in unchanged when that's provisioned.
   Local single-node host hits the `ParquetFile.file: Mutex<File>`
   serialization bottleneck at ~1.8× peak — documented in the
   bench's module-level docstring; switching to `pread`-based
   unlocked I/O is a future optimisation, not part of Π.15.
2. ✓ Single-socket / single-NUMA-node hosts: 585 / 585 tests pass,
   no regression vs sequential reads.
3. ✓ Cancellation token: 3 unit tests + 1 mid-flight integration
   test confirm cooperative cancellation surfaces `Cancelled` on
   queued targets without leaking allocations.

**Constraints.**
- NUMA module is `cfg(target_os = "linux")` — macOS / Windows builds
  see no NUMA symbols. Portable callers stay on
  `read_columns_parallel`; NUMA-aware callers gate their own usage.
- Cancellation is at target boundaries only — not inside a single
  (rg, col) decode. Fine-grained cancellation would need plumbing
  into per-type readers; deferred.
- Real linear scaling needs `ParquetFile` `pread`-based I/O (separate
  optimisation, not part of Π.15).

**Released as v0.9.0.**

---

## v0.9.1 patch — opportunistic items

A small additive bundle ahead of the next big phase. No API breaks,
no behaviour changes for existing callers.

- **u8 dict-indices reader** —
  `read::DictPreservedColumnU8 { dict_bytes, dict_offsets,
   indices: Vec<u8> }` and `read_column_byte_array_dict_preserved_u8`
  + `..._u8_into`. Saves 3 bytes/row on dict-encoded columns with
  ≤ 256 unique values (most TPC-H string columns). Unlocks Arrow
  `DictionaryArray<UInt8, T>` materialisation in ematix-flow with
  a 4× smaller indices buffer.
- **BYTE_ARRAY adaptive façade** — closes the v0.8 gap.
  `read::read_column_byte_array_predicate_adaptive` (+
  `AdaptiveByteArrayChunkOutput` / `AdaptiveByteArrayOutputKind`).
  Same `DEFAULT_THRESHOLD = 0.10` contract as i32/i64/f64; output
  is bitmap (Fused) or Arrow-style `(bytes, offsets)`
  (Materialized).
- **Split-Block Bloom Filter builder** —
  `bloom::SplitBlockBloomFilterBuilder` (+ `insert_hash`,
  `insert_bytes`, `into_bytes`) and `bloom::optimal_num_blocks`.
  Symmetric to the Π.6c decoder; round-trips byte-stable through
  `SplitBlockBloomFilter::from_bytes`. Full writer-integration
  (emitting bloom filters into a parquet file's body + setting
  `ColumnMetaData.bloom_filter_offset`) is a deferred follow-up
  that needs format-crate metadata-writer work too.

**Released as v0.9.1.**

---

## v0.9.2 patch — bloom-filter writer end-to-end

Closes the bloom-filter story: v0.9.1 shipped the in-memory
builder; v0.9.2 wires it through the codec write path so emitted
Parquet files carry consultable bloom filters that downstream
readers — including the upstream Rust Parquet reader — discover
via ColumnMetaData and apply automatically.

- **Format crate**: `metadata_writer::encode_column_meta_data`
  now writes `bloom_filter_offset` (field 14, i64) and
  `bloom_filter_length` (field 15, i32) when set. Previously
  both panicked.
- **Codec writer**: new public entries
  `write::write_{i32,i64,f64,byte_array}_column_dict_with_bloom_to_path(path,
  name, values, codec, target_fpp)`. Builds an SBBF over the
  column's distinct values (sized via `optimal_num_blocks`),
  emits it inline with the column chunk.
- **Interop**: 4 round-trip tests confirm parquet-rs reads our
  bloom filters via `ReaderProperties::set_read_bloom_filter(true)`
  + `RowGroupReader::get_column_bloom_filter`, and every distinct
  value reports present.

Hash contract follows the spec: XXHash64 seed=0 of the value's
PLAIN-encoded bytes (LE for scalar, raw bytes for byte_array
without length prefix).

Scope notes (deferred):
- Multi-column / multi-row-group bloom writes.
- Bloom on plain (non-dict) write paths.

**Released as v0.9.2.**

---

## Π.16 — Custom LLVM codegen for hot decode paths (speculative)

**Goal.** Photon (Databricks) generates per-query LLVM IR for hot
decode loops, fusing predicate, dictionary lookup, projection, and
output materialization into one tight kernel. Our const-generic
monomorphization gets ~80% of this for free at compile time, but
the remaining 20% requires runtime codegen for shapes we can't
know at crate-build time (predicate trees, complex dict shapes).

**Touches.**
- New crate `ematix-parquet-codegen` depending on `inkwell` (LLVM
  bindings) or `cranelift` (faster JIT, simpler IR).
- Runtime IR generation for fused (predicate × bit_width × dict
  type × downstream operation) shapes that aren't in the const-
  generic table.
- Codegen cache keyed on shape; warm cache on second call to the
  same shape is free.
- Fallback to the existing const-generic path when codegen is
  disabled.

**Acceptance.**
1. Codegen path is feature-gated; default build doesn't pull LLVM.
2. End-to-end: a complex predicate (e.g. `(a > 5 AND b < 10) OR c LIKE 'x%'`)
   decoded via codegen runs ≥ 2× faster than the equivalent
   composed from the const-generic kernels.
3. Cold-cache codegen latency ≤ 100 ms for a typical Q14-shape
   predicate; warm-cache calls are zero-overhead vs the codegen'd
   kernel running directly.

**Estimate.** 6-12 weeks. LLVM/cranelift integration is non-trivial
and demands serious bench discipline to confirm wins are real.

**Why last (and speculative).** This is the most ambitious item and
the least likely to pay off vs. its engineering cost. The const-
generic table covers every shape we've seen in TPC-H lineitem and
in real consumers' workloads. Codegen is justified only if a
specific workload demonstrates the const-generic table is the
bottleneck. Marked **speculative**: may never ship.

---

## What's still open below the phase line

These are smaller items that don't merit a full phase but will be
picked up opportunistically:

- **NEON prefetching in dict gather** (Π.9d) — `pld` instruction.
- **u8 dict indices when bw ≤ 8** — saves Vec<u32> overhead.
- **BYTE_ARRAY batched API** — needs `T: Clone`-based gather.
- **Per-column encoding choice on `write_table_*`** — requires a
  `WriteOptions { encoding_per_column }` shape; land when a real
  consumer asks for mixed-encoding tables.
- **Bloom-filter writer** — symmetric to the decoder; defer until
  a downstream reader will actually consult it.
- **NEON kernels for small widths (1, 4, 5, 8, 20, 21)** — scalar
  is at ~7-9 GB/s output; gather dominates on these columns.
  Revisit only if a workload demands.
- **DELTA_BINARY_PACKED u64-output unpacker** (TODO in `delta.rs`).
