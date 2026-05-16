# Π.10 — Async / object-store integration: design

**Status:** design draft, pre-commitment. Once approved, the
sub-phases land as their own PRs and a v0.3.0 tag closes the phase.

## Problem statement

Real analytical workloads run on object storage — S3, GCS, Azure
Blob, R2 — not local disk. Today `ematix-parquet-io::ParquetFile`
is sync `std::io::Read + Seek`, which forces consumers into one of
three bad shapes:

1. **Download the whole file first**, then `ParquetFile::open` on
   the local path. Pays the entire file's download latency before
   the first byte decodes. Defeats every range-pruning win we have.
2. **Wrap a sync byte-range cache** around the object store. Works
   but blocks a worker thread for every range fetch — wasteful at
   any scale.
3. **Don't use ematix-parquet** and fall back to parquet-rs's async
   path. The status quo for any cloud workload today.

Π.10 closes this gap by adding an async read path that issues
range-aware fetches against any `object_store::ObjectStore`,
streams pages as they arrive, and preserves the parser-side
performance work from Π.1–Π.9.

**Out of scope** for Π.10: async writes (still sync `Write`),
async metadata parsing as a stream (footer + per-row-group
metadata still parse from a fully-materialised byte buffer —
the parse is microseconds, not the bottleneck).

## Design decisions

### D1 — Crate boundary: one new crate `ematix-parquet-async`

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Feature flag on `ematix-parquet-codec`** (`features = ["async"]`) | Single crate to depend on. | `tokio` + `object_store` (~80 transitive deps) appear in every codec build's dep graph; cargo's feature unification means downstream consumers can't really opt out once anything in the workspace enables `async`. |
| **B. Two new crates** (`-io-async` + `-async`) | Mirrors the sync split. | Cross-crate deps multiply; user-facing surface gets noisier (3 imports). |
| **C. One new crate** `ematix-parquet-async` ✅ | Clean line: sync stays small + dep-free; async pulls tokio/object_store only when requested. | New crate to publish + version-bump alongside the others. |

**Decision: C.** New crate `ematix-parquet-async` depends on
`ematix-parquet-format` + `ematix-parquet-codec` (re-uses every
byte-slice decoder) plus `tokio` + `object_store`. The sync io
crate is unchanged.

### D2 — Runtime: tokio-only

`object_store` is tokio-bound and is the entire Rust ecosystem's
de facto async object-storage trait. Wrapping it just to expose
a runtime-agnostic `futures::AsyncRead` surface would be pointless
yak-shaving — the underlying ops all schedule on tokio anyway.

**Decision: tokio.** Pin to `tokio = "1"`. For callers stuck on
async-std / smol / monoio, document `tokio::runtime::Handle::block_on`
or `tokio::task::spawn_blocking` as the bridge.

### D3 — ObjectStore: depend on `object_store` directly

The `object_store` crate (from Apache Arrow) provides
`LocalFileSystem`, `AmazonS3`, `GoogleCloudStorage`,
`MicrosoftAzure`, `Http`, and `InMemory`. We accept any
`Arc<dyn ObjectStore>` and expose it in our API.

Trade-off: `object_store` is pre-1.0 and has had breaking changes
between minor versions. Mitigation:
- Pin to a specific minor version in Cargo.toml (currently
  `object_store = "0.11"` as of writing — confirm at impl time).
- Re-export `object_store::ObjectStore` from
  `ematix_parquet_async::store` so users don't import it twice.
- Bumping object_store is a tracked chore — accept the cost.

### D4 — Cold-open: ≤2 round trips

Following parquet-rs convention:

```
1. GET last 8 KB with Range: bytes=-8192
   → returns body + Content-Range header (gives file size)
2. Parse footer:
   - bytes[len-8 .. len-4] = "PAR1"      (magic)
   - bytes[len-12 .. len-8] = footer_len (u32 LE)
3. If footer_len ≤ 8KB - 8: done (one round trip).
   Else: GET range [size - footer_len - 8, size]   (second round trip)
4. Parse Thrift FileMetaData from the footer bytes.
```

Best case: 1 GET. Worst case: 2 GETs (for files with > 8 KB footers,
typically only files with many columns × many row groups).

### D5 — Per-column read: 1 round trip per column chunk

For each `(row_group, column)` read:
- Compute the byte range from `ColumnMetaData.{data_page_offset,
  dictionary_page_offset, total_compressed_size}`.
- Issue one GET for that range.
- Hand the bytes to the existing sync page walker + codec.

No streaming during decode — we buffer the whole column chunk and
walk pages from memory. Chunks are typically 1-50 MB; streaming
pages individually would add latency without saving meaningful
memory.

For **multi-column reads** (`read_columns_async(&[(rg, col), ...])`),
use `ObjectStore::get_ranges(path, &[Range])` which issues all
ranges in one HTTP request via `Range: bytes=` multi-spec. Cuts
round trips when reading many columns from the same file.

### D6 — Public API shape

```rust
// Crate: ematix-parquet-async

pub struct AsyncParquetFile {
    store: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    size: u64,
    metadata: FileMetaData,
}

impl AsyncParquetFile {
    /// Open by issuing ≤2 GETs to fetch the footer + parse metadata.
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        path: object_store::path::Path,
    ) -> Result<Self>;

    /// Direct metadata access (already parsed during open).
    pub fn metadata(&self) -> &FileMetaData;
    pub fn size(&self) -> u64;

    /// Lower-level: fetch an arbitrary byte range.
    pub async fn read_range(&self, offset: u64, len: u64) -> Result<Bytes>;
}

// ---- Read façade: mirrors `ematix_parquet_codec::read` ----

pub async fn read_column_i64_async(
    file: &AsyncParquetFile, rg: usize, col: usize,
) -> Result<Vec<i64>>;

pub async fn read_column_i64_async_into(
    file: &AsyncParquetFile, rg: usize, col: usize, out: &mut Vec<i64>,
) -> Result<()>;

// Same shape for i32, f64, byte_array, int96, flba,
// byte_array_offsets, and the *_with_range variants.

// ---- Streaming: mirrors `ColumnBatchIter` ----

pub fn read_column_i64_async_stream(
    file: &AsyncParquetFile, rg: usize, col: usize, batch_size: usize,
) -> impl Stream<Item = Result<Vec<i64>>>;
```

API conventions, copied from the sync side:
- Every allocating entry point has an `_into(&mut Vec<T>)` sibling
  for buffer reuse.
- Stream variant yields batches of `batch_size` (final may be shorter).
- Error type is the same `CodecError`; `object_store::Error` is
  mapped via a new `CodecError::ObjectStore(String)` variant.

### D7 — Prefetch / pipelining: deferred to Π.10.1

A `prefetch_columns(&[(rg, col)])` API that issues parallel GETs
into a per-file cache is appealing — caller pipelines decode and
fetch. But:
- For single-column reads, `futures::join!` over the existing
  `read_column_*_async` already pipelines fine.
- For multi-column reads from the same file, `get_ranges` already
  coalesces.
- A real cache adds complexity (eviction policy, key lifetime,
  thread-safety) that's only justified once a workload measurably
  benefits.

**Decision: ship Π.10 without prefetch.** Add as Π.10.1 if a
real bench shows pipelining is the bottleneck.

### D8 — Cancellation

`tokio::select!` works naturally — async reads abort on the next
`.await` point. The `Stream` from `_async_stream` is cancel-safe
because it owns its own `AsyncParquetFile` reference and chunk
bytes.

No explicit `CancellationToken` API — the standard tokio
cancellation pattern (drop the future) is enough.

### D9 — Tests

| Layer | Strategy |
| --- | --- |
| Unit | `InMemory` ObjectStore + bytes written by parquet-rs. Replay every existing sync oracle against the async path. |
| Integration (LocalFileSystem) | `LocalFileSystem` ObjectStore against TPC-H lineitem.parquet. Cross-check value-by-value against the sync read. |
| Integration (S3, nightly CI) | Behind feature flag `s3-it`; requires `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env vars. Runs in scheduled nightly workflow, not on every PR. |
| Bench | `bench_decode_async` example: decode TPC-H lineitem from (a) LocalFileSystem vs sync, (b) S3 (skipped without creds). |

All async oracle tests use `tokio::test` with the multi-thread
runtime so async behaviour matches what consumers see.

### D10 — Performance acceptance

Two distinct targets because they measure different things:

1. **Async overhead** (parser-bound): on `LocalFileSystem`
   ObjectStore, async i64 decode of a TPC-H lineitem column is
   **within 5%** of the sync path. Confirms tokio scheduling
   isn't eating into our parser wins.

2. **Cloud throughput** (network-bound): on S3 (same region),
   async decode hits the network bandwidth ceiling. Concrete
   bar: ≥70% of the EC2 instance's reported NIC throughput on a
   10-column lineitem read with 16-way `futures::join!`.

If (1) misses by >5%, that's a parser-overhead bug we have to
fix before shipping. (2) is documentation of what users should
expect, not a gating bar.

## Sub-phases

| # | Sub-phase | Scope | Estimate |
| --- | --- | --- | --- |
| Π.10a | `AsyncParquetFile` primitive | `open`, `metadata()`, `read_range`. ObjectStore wiring, footer parse (≤2 RT). `InMemory` + `LocalFileSystem` oracle tests. | 3-4 days |
| Π.10b | Async read façade — scalar types | `read_column_{i32,i64,f64,int96,flba}_async{,_into}` + the `_with_range_*` variants. Replay every sync oracle against async. | 3-4 days |
| Π.10c | Async read façade — byte_array | `read_column_byte_array_async{,_into,_offsets,_offsets_into}_async{,_into}`. Mirror the sync byte_array contract. | 2-3 days |
| Π.10d | Async streaming batch iterator | `read_column_*_async_stream(batch_size)` returning `impl Stream<Item = Result<Vec<T>>>`. Mirror `ColumnBatchIter`'s shape. | 3-4 days |
| Π.10e | S3 integration tests | New CI workflow `nightly.yml` running the `s3-it` test set against a public-read TPC-H bucket. Skipped on PRs. | 1-2 days |
| Π.10f | Bench + docs | `bench_decode_async` example. README "Reading from S3" section. Update `docs/RELEASING.md` for the new crate. | 2 days |
| **Total** | | | **14-19 days** |

Realistically 3-4 calendar weeks including review + churn on
the object_store API.

## What lands as v0.3.0

Tagging v0.3.0 once Π.10a–Π.10f are merged on `main`. The four
existing crates bump in sync to 0.3.0. The new crate ships at
0.3.0 too (matches the workspace version; the version-pin
discipline holds — inter-crate deps are `version = "0.3"`).

## Risks + open questions

1. **`object_store` minor-version churn.** Mitigation: pin
   tightly, accept the maintenance cost. Question: do we want to
   re-export the trait so users don't double-import? **Lean yes.**

2. **Tokio runtime requirement.** Consumers who don't use tokio
   today need to add it. Question: should we document a `smol`
   bridge, or just say "tokio-only"? **Lean tokio-only**; smol
   users can `block_on` or use a tokio compatibility layer.

3. **Range-coalescing across columns.** `get_ranges` works for
   contiguous ranges in one file; what about a multi-file query?
   Out of scope — that's a query-engine concern, not a codec one.

4. **Auth / credentials.** Pushed entirely to `object_store` —
   we never see credentials. Question: do we want a convenience
   `AsyncParquetFile::open_s3(bucket, key)` that constructs the
   `AmazonS3` builder from env vars? **Lean no** — keeps our crate
   credential-policy-free, callers wire up their own ObjectStore
   exactly once.

5. **Sync API consistency for "with_range" variants.** Sync has
   `read_column_i64_with_range(file, rg, col, lo, hi)` for page-
   index pruning. Async needs the same. The page-index parse +
   range fetch is the same logic — only the GET is async. Easy.

6. **What if a user wants async + the existing fused-predicate
   path?** They can: the byte-slice decoders (incl. predicate-
   fused) all live in `ematix-parquet-codec` and are sync. Async
   just fetches bytes and hands them to those. No new decoder
   surface needed.

## Confirmed decisions

Before sub-phase Π.10a starts, two questions were raised in the
draft and confirmed by the maintainer:

- **Q1 → confirmed.** New crate `ematix-parquet-async` (D1.C).
  The workspace grows to four crates. Sync stays small and
  dep-free; the async crate isolates tokio + object_store.
- **Q2 → confirmed.** Pin object_store tight (~0.11 or whatever
  is current at impl time). Accept a periodic bump-and-fix chore
  every 6-12 months as object_store releases minor versions.
  Re-export the trait from `ematix_parquet_async::store` so
  consumers don't double-import.
