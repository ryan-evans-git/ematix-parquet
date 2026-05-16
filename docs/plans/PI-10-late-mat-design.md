# Π.10 — Late-materialization read façade: design

**Status:** design draft, pre-commitment. Ships as **v0.3.0**.
Async / object-store (formerly Π.10) becomes Π.11 and ships
as v0.4.0; see `PI-11-async-design.md`.

## Problem statement

The TPC-H Q14 end-to-end benchmark in ematix-flow currently hits
**14.6 ms**; Polars hits **12.53 ms**. That ~2 ms gap is
kernel-vs-kernel inside ematix-parquet's scope — both engines
decompress the same Snappy bytes and walk the same RLE-dict
index streams. The gap is a structural one.

Decomposing ematix-parquet's 14.6 ms on Q14:

| Stage | Approx | Why |
| --- | --- | --- |
| 4× lineitem column decode (partkey, extprice, discount, shipdate) | ~10-11 ms | Dominant cost; Snappy + RLE-dict unpack + Arrow build × 4 cols × ~298 pages each |
| Filter mask construction (shipdate) | ~0.4 ms | Π.9b NEON-fused bw=12, already 2.1× parquet-rs |
| Part decode + promo-bitmap | ~1 ms | Small dimension table, dense |
| Join + agg | ~0.5 ms | Direct-indexed Vec<bool> |
| Page-walk / scratch / setup overhead | ~1-2 ms | Per-page allocation, dispatch, headers |

The structural difference vs Polars is **late materialization
within the chunk**. Polars's path:

```
1. decode l_shipdate            (full 6M rows)
2. build filter mask from shipdate
3. for partkey, extendedprice, discount:
     sparse-decode only mask-true rows (~84K)
   → 99% of decode work on three columns SKIPPED
```

Today's ematix-parquet path:

```
1. decode l_partkey       (full 6M rows)
2. decode l_extendedprice (full 6M rows)
3. decode l_discount      (full 6M rows)
4. decode l_shipdate      (full 6M rows)
5. build filter mask from shipdate, apply to everything
   → 99% of the decode work in steps 1-3 was thrown away
```

Q14's shipdate window is 30 days out of ~7 years — selectivity
~1.4%. Polars decodes ~84K rows × 3 columns instead of ~6M × 3.
That accounts for almost exactly the 2 ms gap.

## Why this is shovel-ready

Every building block already lives in `ematix-parquet-codec`.
We just don't expose a top-level "decode column with a row-mask"
entry point that composes them.

| Building block | Status | Source |
| --- | --- | --- |
| `gather_dict_at_bitmap_into<T>` — decode only at `bitmap[i]==1` rows; 8-row block-skip when the bitmap byte is 0 | ✓ | `dict.rs:447` |
| `decode_rle_dictionary_predicate_bitmap` — RLE-dict → bitmap, NEON-fused per width | ✓ shipped Π.9b | `dict.rs:225` |
| `decode_*_into(&mut Vec<T>)` — caller-buffer reuse | ✓ shipped Π.9a | `read.rs` (every read entry) |
| `PageWalker` + `data_page_view` — page iteration + V1/V2 abstraction | ✓ | `read.rs:705` |
| `read_column_*_masked_into` — top-level: walk pages, popcount mask per page, dispatch to dict-sparse or PLAIN-sparse | **missing** | this phase |
| PLAIN sparse-decode (fixed-width) — read at `mask[i]==1` offsets | **missing** | this phase |
| PLAIN sparse-decode (byte_array) — variable-length, gather by length-prefix walk | **missing** | this phase |

The decision to reject Photon-style late-materialization in
`docs/project_late_materialization_next.md` (May 2026) was about
parquet-rs's `with_row_filter` API hitting **0.98×** on Q14. That
benchmark was specifically against parquet-rs's implementation,
not against composing our own primitives. Our existing
`gather_dict_at_bitmap_into` already exists precisely to make this
work pay off — it just needs to be wired up from the façade.

## Design decisions

### D1 — One new entry point per scalar type

```rust
// crate: ematix-parquet-codec
// module: read (the high-level façade)

pub fn read_column_i64_masked_into(
    file: &ParquetFile,
    row_group: usize,
    column: usize,
    mask: &[u8],          // packed bitmap, 1 bit per row in the chunk
    out: &mut Vec<i64>,   // append-only; caller clears if reusing
) -> Result<()>;
```

Same shape for `_i32_`, `_f64_`, `_byte_array_`, plus
`_byte_array_offsets_` (which appends to `(bytes, offsets)`).

Inputs:
- `mask.len() * 8 ≥ chunk.num_values` — caller asserts the mask
  covers the whole chunk (extra bits past num_values are ignored).
- `out` is appended to in row order, in mask-set positions only.
- Final `out.len() == popcount(mask[..num_values_in_bits])`.

Why "append" and not "clear-then-fill" like the existing `_into`
variants: Q14-shape consumers run this once per row-group, then
move on. They typically build a contiguous output across multiple
row-groups via a single Vec; clearing per-call would force them
to track per-call boundaries themselves. Append semantics are
strictly more flexible — caller calls `out.clear()` if they want.

### D2 — Per-page popcount skip

For each data page, compute `popcount(mask[page_row_start ..
page_row_start + page_num_values])`. If zero, skip the page
entirely — no decompression, no decode. Cost: O(page_num_values
/ 8) bytes scanned; pays off when whole pages are dead.

For Q14 specifically, shipdate is uniformly distributed within
each page (per the page-index probe in
`docs/project_late_materialization_next.md`), so we expect
~0 fully-dead pages. The popcount-skip is correct but doesn't
help Q14 directly — it pays off on workloads with date-sorted
columns or column-store partitions. **Ship it anyway**: cheap
to add, free at runtime when not exercised.

### D3 — Dict-encoded path → `gather_dict_at_bitmap_into`

The hot path. For each dict-encoded data page where the
per-page mask is non-empty:

```rust
let bitmap_offset = page_row_start;  // global row index of page row 0
gather_dict_at_bitmap_into(
    info.values,
    info.num_values,
    mask,
    bitmap_offset,
    &dict,
    out,
)?;
```

This already does the right thing per page — walks RLE/bit-packed
indices, computes `popcount` per 8-row bitmap byte, only decodes
the 8-row blocks where the bitmap has at least one set bit.
At Q14 selectivity (~1.4%), that's ~7-12 NEON ops per 8 rows
instead of 8 × dict-lookup + 8 × store.

### D4 — PLAIN path → new sparse-decode primitives

For PLAIN-encoded fixed-width pages (INT32/INT64/F64), add:

```rust
fn plain_sparse_decode_i64_into(
    bytes: &[u8],          // page values
    num_values: usize,
    mask: &[u8],
    bitmap_offset: usize,
    out: &mut Vec<i64>,
) -> Result<()>;
```

Implementation per type:

```rust
// Fixed-width: for each set bit in mask[bitmap_offset..bitmap_offset + num_values],
// read 8 bytes at offset (row_index * 8) and push to out.
for row in 0..num_values {
    let bit_pos = bitmap_offset + row;
    let bit = (mask[bit_pos / 8] >> (bit_pos % 8)) & 1;
    if bit == 1 {
        let v = u64::from_le_bytes(bytes[row * 8..row * 8 + 8].try_into().unwrap()) as i64;
        out.push(v);
    }
}
```

The naive form above is fine for v0.3.0; an 8-row block-skip
variant (mirror of `gather_dict_at_bitmap_into`'s scan) is a
fast follow-up if a benchmark needs it.

For PLAIN byte_array (variable-length), the offsets force a
sequential walk:

```rust
fn plain_sparse_decode_byte_array_offsets_into(
    bytes: &[u8], num_values: usize, mask: &[u8], bitmap_offset: usize,
    out_bytes: &mut Vec<u8>, out_offsets: &mut Vec<u32>,
) -> Result<()>;
```

Walk all values length-prefix-first; copy only mask-set values'
bytes into `out_bytes` and push the running offset. Variable-
length means we can't skip ahead without parsing.

### D5 — Mask format: aligned to chunk rows

`mask: &[u8]` is a packed bitmap, bit `i` of byte `k` = row
`8k + i` in the chunk's row 0..num_values address space.

- Caller is responsible for building the mask. Typical pattern:
  ```rust
  let mut shipdate = Vec::new();
  read_column_i32_into(&file, rg, shipdate_col, &mut shipdate)?;
  let mut mask = build_mask(&shipdate, |&d| d >= LO && d < HI);  // user code
  ```
  Or for dict-encoded filter columns, the existing Π.9b path:
  ```rust
  let dict_mask = build_dict_predicate_mask(&dict, bit_width, pred)?;
  decode_rle_dictionary_predicate_bitmap(body, n, &dict_mask, &mut mask)?;
  ```

- The bitmap covers the whole chunk; this phase does not provide
  multi-chunk mask helpers (callers iterate row-groups themselves).

### D6 — Public surface: read façade only

The masked-decode entry points live in `ematix_parquet_codec::read`
alongside the existing `read_column_*` and `read_column_*_into`
families. No new crate. No new module.

### D7 — Acceptance: bit-identical to full-decode-then-filter

Oracle tests (every type × every page encoding):

```rust
// Reference path
let full = read_column_<T>(&file, rg, col)?;
let want: Vec<T> = full.iter().zip(mask_bits)
    .filter(|(_, m)| *m == 1).map(|(v, _)| *v).collect();

// New path
let mut got = Vec::new();
read_column_<T>_masked_into(&file, rg, col, &mask, &mut got)?;

assert_eq!(got, want);
```

Across:
- Types: i32, i64, f64, byte_array (Vec<u8> + offsets variant)
- Encodings: PLAIN, RLE_DICTIONARY (dict + bit-packed run)
- Mask selectivities: 0%, 0.1%, 1%, 10%, 50%, 90%, 100%
- Page boundaries: mask transitions across page edges

### D8 — Acceptance: end-to-end Q14 bench

`bench_q14_late_mat` example added under
`crates/ematix-parquet-codec/examples/`:

```
== Q14 — full decode → filter (baseline) ==
  l_partkey + l_extendedprice + l_discount + l_shipdate decode :  ~10.5 ms
  apply mask + agg                                              :  ~ 1.0 ms
  total                                                          :  ~11.5 ms

== Q14 — late-materialization (this phase) ==
  l_shipdate decode + mask build                                :  ~ 1.0 ms
  l_partkey + l_extendedprice + l_discount  masked-decode      :  ~ 1.5 ms
  agg                                                            :  ~ 0.3 ms
  total                                                          :  ~ 2.8 ms
```

End-to-end Q14 in ematix-flow target: **≤ 13.0 ms** (beats Polars
12.53 with breathing room). Stretch: **≤ 11.0 ms** if the
3-column masked-decode lands closer to 1 ms total.

If Q14 lands at 12.5-13 ms, we're at parity with Polars on the
benchmark that matters; further wins require matching kernel
constants (decode-loop μ-arch tuning, ~5-10% upside) or a
different lever.

## Sub-phases

| # | Sub-phase | Scope | Estimate |
| --- | --- | --- | --- |
| Π.10a | `read_column_*_masked_into` for i32/i64/f64 | Façade entry points; dict-encoded uses existing `gather_dict_at_bitmap_into`; PLAIN gets new `plain_sparse_decode_<T>_into` (3 types × ~30 LOC each). Per-page popcount skip. Oracle tests across encodings × selectivities. | 3-4 days |
| Π.10b | `read_column_byte_array_masked_into` + offsets variant | The variable-length case. Walk length-prefixes sequentially; copy only mask-set values. Dict-encoded path reuses `gather_dict_at_bitmap_into<Vec<u8>>` (already generic on T). PLAIN gets a new sparse-decode that does the length walk. Oracle tests. | 2-3 days |
| Π.10c | Q14 bench + bridge integration in ematix-flow | `bench_q14_late_mat` example in this repo. In ematix-flow: bridge orders the decode (filter col → mask → other cols), behind a `with_late_mat` builder flag for A/B. Full TPC-H sweep for 0 regressions. | 3-4 days |
| **Total** | | | **8-11 days** |

Realistically ~1.5 calendar weeks single-developer.

## What lands as v0.3.0

Tagging v0.3.0 once Π.10a–Π.10c are merged. The three crates
(`ematix-parquet-{format,io,codec}`) bump in sync to 0.3.0;
inter-crate version pins move to `version = "0.3"`. No new crates
in this release — async (Π.11) is what adds `ematix-parquet-async`.

The ematix-flow integration (Π.10c) ships in ematix-flow's own
release; this repo's v0.3.0 just exposes the API.

## Risks + open questions

1. **Mask format: packed vs Vec<bool>.** Packed (`&[u8]`) matches
   what `gather_dict_at_bitmap_into` already takes and what
   Π.9b's `decode_rle_dictionary_predicate_bitmap` emits — no
   conversion. `Vec<bool>` would be ergonomic but force every
   caller to convert. **Decision: packed.** Document the bit-
   layout convention up-front; provide a `build_packed_mask`
   helper that takes a closure `|i| -> bool` for callers who
   don't already have a packed bitmap.

2. **Selectivity threshold for adaptive dispatch.** At what
   selectivity does masked-decode lose to full-decode-then-filter?
   The fused-vs-baseline bench (Π.9b) saw 3.7-6.3× wins at ~1%;
   at ≥ 50% selectivity the baseline likely catches up (no
   bitmap-pack overhead, no popcount cost). **Don't auto-dispatch
   in this phase** — let the caller choose. Π.13 (adaptive
   runtime dispatch, already on the roadmap) covers this.

3. **PLAIN sparse-decode without block-skip.** The naive PLAIN
   path reads every row's mask bit individually. At 1% selectivity
   that's 99 wasted bit-reads per emit; the cache impact is
   minimal but it's not optimal. **Acceptable for v0.3.0** — the
   PLAIN path is much less hot than the dict path on TPC-H
   lineitem (no PLAIN-encoded columns in lineitem's dominant
   shape). Add an 8-row block-skip variant in a follow-up if a
   bench demands it.

4. **Mask covers row-group, not row-group-aligned bit boundary.**
   Caller passes the mask as `&[u8]`; if the row-group has 6M
   rows, the mask is `ceil(6M / 8) = 750000` bytes. Caller is
   responsible for ensuring the mask is sized correctly; we
   assert in debug mode and return `InvalidInput` in release.

5. **Multi-row-group masks.** Q14 spans multiple row-groups; the
   caller has to slice the mask per row-group. Out of scope for
   this phase — that's wiring at the engine layer (ematix-flow).
   We expose row-group-scoped APIs; the caller iterates.

6. **What about DataPageV2?** The existing `data_page_view`
   already abstracts V1 vs V2; masked-decode reuses that. No
   special case needed.

7. **`Vec<u8>: !Copy` for byte_array dict path.** The existing
   `gather_dict_at_bitmap_into<T: Copy>` doesn't accept
   `Vec<u8>`. For byte_array we need a sibling
   `gather_dict_at_bitmap_into_clone<T: Clone>` that pushes
   clones instead of copies. Trivially derivable from the
   existing function; ~30 LOC.

## Confirmed decisions

Before sub-phase Π.10a starts, two design choices to confirm:

- **Q1.** Mask format = packed `&[u8]`. (Recommended; aligns
  with `gather_dict_at_bitmap_into` and Π.9b's bitmap output.)
- **Q2.** Don't auto-dispatch between masked-decode and full-
  decode-then-filter based on selectivity in this phase.
  Caller picks; revisit in Π.13.

Both default to the choices made in this doc. Adjust if either
needs to flip before implementation.
