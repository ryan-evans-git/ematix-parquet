# Π.13 — Parquet Modular Encryption: design

**Status:** design draft, pre-commitment. Once approved, the
sub-phases land as their own PRs and a v0.6.0 tag closes the phase.

## Problem statement

The Parquet Modular Encryption (PME) spec
(`apache/parquet-format/Encryption.md`) defines per-column-chunk
AES-GCM encryption for column data, dictionary pages, page indexes,
bloom filters, and (optionally) the footer itself. PME is a hard
requirement for any deployment in finance / healthcare / government,
and is supported by parquet-mr, parquet-rs, and Spark out of the box.

Without PME, ematix-parquet cannot be used in any environment that
requires encrypted-at-rest columnar data — the file simply will not
decode, regardless of how fast we are on plaintext.

This doc breaks the phase into seven sub-phases shaped the same way
Π.12 was (each PR-sized: ~1-3 hours implementation + tests,
independently shippable, every commit green on CI). v0.6.0 ships
after Π.13g.

## Scope

In scope for v0.6.0:

- Read + write of both PME modes: **encrypted footer** and
  **plaintext footer**.
- `AES_GCM_V1` algorithm. (See D4 on `AES_GCM_CTR_V1`.)
- Per-column-chunk keys + a separate footer key, retrieved via a
  caller-provided `trait`.
- AAD prefix + AAD module-id suffix bound into the GCM tag exactly
  as the spec requires.
- Encrypted dictionary pages, data pages V1, data pages V2 (the
  three page types our writer emits today).
- Oracle parity with parquet-rs in both directions.

Explicitly **out of scope** for v0.6.0:

- KMS integrations (AWS KMS, GCP KMS, Vault). Caller's problem; we
  receive raw key bytes and return decrypted bytes.
- Key-rotation tooling beyond a working example showing the rewrite
  pattern. (A standalone `ematix-parquet-keytool` CLI is a possible
  v0.7.x follow-up — not part of Π.13.)
- Encrypted page indexes + encrypted bloom filters. The reader and
  writer in v0.6.0 ignore page index / bloom on encrypted columns
  (falls back to full-chunk read). Tracked as Π.13-followup.
- `AES_GCM_CTR_V1` (see D4).
- Async path. Encryption layers below the I/O surface; the async
  crate gets it for free once the codec supports it, but a separate
  oracle test against an encrypted-file-on-S3 is deferred.

## Design decisions

### D1 — Crate boundary: one new crate `ematix-parquet-crypto`

Same shape as Π.11 (`-async`). The rest of the workspace stays
crypto-free; only consumers that explicitly add the new crate (and
flip `--features encryption` on the codec) pull AES-GCM into their
dep graph.

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Inline in `-codec`** under a `crypto` cfg | Single crate to publish. | Mixes lifecycle: every crypto bump rev's the codec. Hard to audit. |
| **B. New crate `ematix-parquet-crypto`** ✅ | Tight surface, auditable on its own, separate publish cadence, easy to feature-gate out. | One more crate to version + publish. |
| **C. Two crates** (`-crypto` + `-crypto-codec`) | Mirrors `format`/`codec` split. | Over-engineered for ~500 LOC of AES-GCM + AAD plumbing. |

**Decision: B.** `ematix-parquet-crypto` exposes the AES-GCM
primitives, AAD construction, the `KeyRetriever` trait, and nothing
else. `ematix-parquet-codec` depends on it under
`#[cfg(feature = "encryption")]` and wires it into the read/write
façades. Default codec builds never see crypto.

### D2 — Crypto library: `aes-gcm` (RustCrypto), pure-Rust

| Library | MSRV | FFI? | Notes |
| --- | --- | --- | --- |
| **`aes-gcm`** (RustCrypto) ✅ | 1.65 | No | Pure-Rust. AES-NI on x86 via `aes` crate's runtime detection; ARM Crypto Extensions on AArch64 via the same. Audited. Used by `rustls`, `quinn`, etc. |
| `ring` | 1.66 | Yes (BoringSSL via cc) | Fastest, but FFI. Build-system pain on cross-compile + alpine. |
| `openssl` | varies | Yes (system libssl) | Distro dep, version skew, not on our path. |

**Decision: `aes-gcm = "0.10"` + `aes = "0.8"`.** Both comfortably
support MSRV 1.80 and have no FFI deps. Matches the repo's stated
preference for pure-Rust crypto and pure-Rust compression
(`lz4_flex`, `flate2`, `brotli` in `codec/Cargo.toml` follow the
same rule). Pin tight in `crypto/Cargo.toml` with a per-line comment
explaining why each dep is chosen, matching codec's style.

Benchmark hook: a `bench_aes_gcm` example in `-crypto` measures
encrypt/decrypt throughput for typical page sizes (256 KB) on both
M-series and x86 (Linux CI). On a 2024 M3, `aes-gcm` is ~4 GB/s on
the ARM Crypto Extensions path — never the bottleneck (Snappy
decompression already runs ~3 GB/s, AES is faster).

### D3 — `KeyRetriever` trait + caller-owned key material

```rust
// crates/ematix-parquet-crypto/src/key.rs
pub trait KeyRetriever {
    /// Resolve a key by its caller-defined metadata. The bytes in
    /// `key_metadata` come straight from the file's
    /// `key_metadata` Thrift field (opaque to us — could be a
    /// JSON KMS reference, a base64-encoded blob, anything).
    fn retrieve(&self, key_metadata: &[u8]) -> Result<Key, KeyError>;
}

pub struct Key(pub [u8; 16]);   // AES-128
// We also support [u8; 24] / [u8; 32] — D8.
```

The trait is intentionally minimal. KMS integrations (AWS, GCP,
Vault) belong in downstream crates; we just need bytes. A test
helper `StaticKeys(HashMap<Vec<u8>, Key>)` ships in
`-crypto`'s dev-dep section for oracle tests + caller examples.

### D4 — `AES_GCM_V1` only; `AES_GCM_CTR_V1` deferred

`AES_GCM_CTR_V1` exists for environments where pure-CTR (no
authentication tag on data pages) is acceptable for performance.
The spec mandates both modes to be readable. **For v0.6.0 we only
implement `AES_GCM_V1`** (full GCM on every encrypted module) and
**error cleanly** with `UnsupportedAlgorithm` on
`AES_GCM_CTR_V1` files. This is in line with our "ship slim, expand
on demand" stance: every benchmark we have on M-series and x86
shows GCM is not the bottleneck.

Rationale:
- parquet-rs's encryption path defaults to `AES_GCM_V1`. The
  practical fleet of PME files in the wild is overwhelmingly GCM.
- `AES_GCM_CTR_V1` adds a second AAD construction rule + a fallback
  decryption mode that is non-trivial to validate against
  parquet-rs (their reference test fixtures are GCM-only).
- A v0.6.x point release can add it if a real consumer asks.

Tracked as Π.13-followup.

### D5 — AAD construction is in `-crypto`, not the codec

The AAD (Additional Authenticated Data) for each encrypted module is:

```
AAD = aad_prefix || aad_suffix
aad_suffix = file_aad || module_type || row_group_ordinal
              || column_ordinal || page_ordinal
```

Where `aad_prefix` is caller-supplied (per the spec — optional but
recommended for cross-file binding) and the suffix bytes are
encoding-defined. We expose a typed builder:

```rust
pub fn build_module_aad(
    aad_prefix: Option<&[u8]>,
    aad_file_unique: &[u8],
    module: ModuleType,    // Footer | DataPage | DictPage | DataPageHeader | DictPageHeader | ColumnMetaData | OffsetIndex | ColumnIndex | BloomFilterHeader | BloomFilterBitset
    rg_ordinal: i16,
    col_ordinal: i16,
    page_ordinal: Option<i32>,
) -> Vec<u8>
```

The codec hands the builder a typed `ModuleType` enum + ordinals;
encoding the byte layout (little-endian widths, when to elide
page_ordinal, etc.) is the crypto crate's job. This keeps AAD
quirks out of the read/write hot path and isolates them in a
testable surface — a unit test in `-crypto` validates every
module-type's AAD byte layout against published spec test vectors.

### D6 — Wire shape of an encrypted page

Per spec, an encrypted module (page, page header, footer) is:

```
[length: u32 LE] [nonce: 12 bytes] [ciphertext: N bytes] [tag: 16 bytes]
```

Where `length = 12 + N + 16` (the on-disk byte count after the
length prefix).

- The **page header** itself can be encrypted as a separate module
  preceding the page body. In encrypted-footer mode the spec is
  explicit: page headers are encrypted with the same column key.
- The **page body** is encrypted as one module per page (a single
  GCM seal over the whole compressed body).
- For DataPageV2, only the compressed-values portion is encrypted;
  rep/def levels stay plaintext (matching V2's `is_compressed`
  semantics — rep/def levels are never compressed, never encrypted).

We model these as `EncryptedModule<'a> { nonce: &[u8; 12],
ciphertext_and_tag: &'a [u8] }` views in `-crypto` and let the codec
slice the on-disk bytes to construct them. Decrypt allocates one
output buffer (sized = ciphertext length); future Π.13-perf could
reuse a caller-owned scratch buffer like our `_into` decode paths.

### D7 — Two PME modes are one writer, two flags

| Mode | Footer | Per-column |
| --- | --- | --- |
| **Encrypted footer** | Footer is one big AES-GCM module; the file ends with `[encrypted footer][PARE]` instead of `[plaintext footer][PAR1]`. | All encrypted columns are AES-GCM. |
| **Plaintext footer** | Footer parses normally as Thrift; carries the new `EncryptionAlgorithm` field and a per-column-chunk `ColumnCryptoMetaData` pointing each encrypted column at its key metadata. | Encrypted columns are AES-GCM; unencrypted columns are plain. |

The reader auto-detects from the trailer magic (`PAR1` vs `PARE`).
The writer takes a `WriteEncryption` config:

```rust
pub struct WriteEncryption {
    pub mode: EncryptionMode,             // EncryptedFooter | PlaintextFooter
    pub footer_key: Option<Key>,          // required iff EncryptedFooter
    pub footer_key_metadata: Vec<u8>,     // opaque KMS reference, included in trailer
    pub column_keys: HashMap<Vec<String>, (Key, Vec<u8>)>,  // schema path → (key, key_metadata)
    pub aad_prefix: Option<Vec<u8>>,
    pub aad_file_unique: [u8; 16],        // spec-recommended random per file
}
```

Columns absent from `column_keys` are written plaintext (PME spec
allows this — a "selectively encrypted" file).

### D8 — Key sizes: 128/192/256-bit, runtime-selected

`Key` is an enum:

```rust
pub enum Key {
    Aes128([u8; 16]),
    Aes192([u8; 24]),
    Aes256([u8; 32]),
}
```

The wire format doesn't carry key size — the retriever returns
whatever size matches the KMS-stored key, and `aes-gcm` dispatches
on the type at the seal/open call site. Oracle tests cover all
three sizes; performance is within 5% across sizes on AArch64
Crypto Extensions.

### D9 — Footer trailer wire format

| Mode | Trailer (last 8 bytes) |
| --- | --- |
| Plaintext (today) | `[footer_length: u32 LE] PAR1` |
| Plaintext footer + encrypted columns | `[footer_length: u32 LE] PAR1` (same; encryption metadata lives inside the footer) |
| Encrypted footer | `[encrypted_footer_length: u32 LE] PARE` |

The `ParquetFile` open path in `-io` already reads the last 8 bytes
to bootstrap. Π.13d extends that to recognize `PARE` and dispatch
the footer-decryption path. The plaintext-footer variant doesn't
change the trailer at all — only the parsed metadata changes shape.

### D10 — Feature-gating exactly

```toml
# crates/ematix-parquet-codec/Cargo.toml
[features]
default = []
encryption = ["dep:ematix-parquet-crypto"]
```

```rust
// crates/ematix-parquet-codec/src/read.rs
#[cfg(feature = "encryption")]
mod encrypted;     // pulled in only with the feature

// at every dispatch site that handles a potentially-encrypted chunk:
#[cfg(feature = "encryption")]
if column_chunk.crypto_metadata.is_some() {
    return encrypted::decode_chunk(..);
}
#[cfg(not(feature = "encryption"))]
if column_chunk.crypto_metadata.is_some() {
    return Err(ReadError::EncryptionFeatureRequired);
}
```

Building `ematix-parquet-codec` with default features pulls zero
crypto code (and zero crypto deps). Building with
`--features encryption` pulls `ematix-parquet-crypto` and lights
up the dispatch sites. CI matrix covers both shapes.

## Sub-phase breakdown

Seven sub-phases. Each is one PR; each compiles, tests, and ships
incrementally. **Π.13g cuts v0.6.0.** Estimates assume a working
session of ~2 focused hours unless noted.

| Sub-phase | Theme | Estimate | Blocks |
| --- | --- | --- | --- |
| Π.13a | Thrift extensions for encryption metadata (read-only) | 3-4 h | Π.13c, Π.13d |
| Π.13b | New `-crypto` crate: primitives + AAD + key trait | 3-4 h | Π.13c, Π.13d, Π.13e, Π.13f |
| Π.13c | Read path: plaintext-footer mode | 3-4 h | Π.13d (validates AAD wiring) |
| Π.13d | Read path: encrypted-footer mode | 3 h | Π.13e (writer needs read-back oracle) |
| Π.13e | Write path: plaintext-footer mode | 4-5 h | Π.13f |
| Π.13f | Write path: encrypted-footer mode + key-rotation example | 3-4 h | Π.13g |
| Π.13g | Release wiring: feature flag CI, README, v0.6.0 tag | 2 h | — |

**Total:** ~21-26 hours of implementation + tests. Calendar: 2-3
weeks of part-time work with PR review.

---

### Π.13a — Thrift extensions in `ematix-parquet-format` (read-only)

**Goal.** Parse `FileMetaData.encryption_algorithm` (field 8),
`FileMetaData.footer_signing_key_metadata` (field 9),
`ColumnChunk.crypto_metadata` (field 8),
`ColumnChunk.encrypted_column_metadata` (field 9), and the
`FileCryptoMetaData` standalone struct used in encrypted-footer
mode. No decrypt yet — just parse and round-trip the bytes.

**Touches.**
- `crates/ematix-parquet-format/src/metadata.rs`:
  - New types: `EncryptionAlgorithm` (union of `AesGcmV1` and
    `AesGcmCtrV1`), `AesGcmV1 { aad_prefix, aad_file_unique,
    supply_aad_prefix }`, `AesGcmCtrV1` (same shape), `FileCryptoMetaData
    { encryption_algorithm, key_metadata }`,
    `ColumnCryptoMetaData` (union of `EncryptionWithFooterKey`
    + `EncryptionWithColumnKey { path_in_schema, key_metadata }`).
  - Reader fns: `read_encryption_algorithm`, `read_aes_gcm_v1`,
    `read_aes_gcm_ctr_v1`, `read_file_crypto_metadata`,
    `read_column_crypto_metadata`.
  - Extend `FileMetaData` with `encryption_algorithm: Option<EncryptionAlgorithm<'a>>`
    + `footer_signing_key_metadata: Option<&'a [u8]>`. Wire fields 8/9 in
    `read_file_metadata` instead of erroring `UnknownStructField`.
  - Extend `ColumnChunk` with `crypto_metadata: Option<ColumnCryptoMetaData<'a>>`
    + `encrypted_column_metadata: Option<&'a [u8]>`. Same: field 8/9
    in `read_column_chunk`.
- `crates/ematix-parquet-format/src/metadata_writer.rs`:
  - Symmetric writer fns (used by Π.13e/f, but landing them now
    keeps `metadata.rs` and `metadata_writer.rs` in lockstep).
- `crates/ematix-parquet-format/tests/encryption_metadata_oracle.rs`
  (NEW): write a tiny encrypted-footer file via parquet-rs (with
  `EncryptionConfiguration`), read the bytes back, decode our
  `FileCryptoMetaData` struct from the trailer, assert field-by-field
  parity with what parquet-rs's footer parser produces. Cover both
  algorithm variants on the read side (parquet-rs can emit
  AesGcmCtrV1 for the metadata test even though we won't decrypt
  it).

**Acceptance.**
1. `read_file_metadata` on a plaintext-footer file (parquet-rs PME
   write) returns `encryption_algorithm = Some(AesGcmV1 { .. })` with
   the right `aad_file_unique` bytes.
2. `read_column_chunk` on each encrypted column returns
   `crypto_metadata = Some(EncryptionWithColumnKey { .. })`.
3. New file `tests/encryption_metadata_oracle.rs` has ≥ 4 tests:
   plaintext-footer PME (AesGcmV1), plaintext-footer PME with
   per-column keys, encrypted-footer trailer parse
   (`read_file_crypto_metadata` directly), missing-field error
   surfaces clean.
4. Existing tests stay green — additions are backward-compatible
   (new fields default to `None`).

**Blocks Π.13c + Π.13d** (both need the metadata parsed before any
decrypt can dispatch).

---

### Π.13b — New `ematix-parquet-crypto` crate

**Goal.** Stand up the crate, get the AES-GCM primitives + AAD
builder + `KeyRetriever` trait under unit-test coverage against
NIST test vectors. No coupling to the codec yet.

**Touches.**
- `Cargo.toml` (workspace): add `crates/ematix-parquet-crypto` to
  `members`.
- `crates/ematix-parquet-crypto/Cargo.toml` (NEW): pin
  `aes-gcm = "0.10"` and `aes = "0.8"` with per-line rationale
  comments matching codec's style.
- `crates/ematix-parquet-crypto/src/lib.rs` (NEW):
  - `mod aead;` — `seal(key, nonce, aad, plaintext) -> Vec<u8>` and
    `open(key, nonce, aad, ciphertext_and_tag) -> Result<Vec<u8>>`.
    Both branch on the `Key` enum's variant.
  - `mod aad;` — `ModuleType` enum, `build_module_aad` per D5.
  - `mod key;` — `KeyRetriever` trait, `Key` enum, `StaticKeys`
    test helper.
  - `mod nonce;` — `RandomNonceSource` (uses `getrandom`); test
    helper `FixedNonceSource` for deterministic oracle tests.
  - `mod error;` — `CryptoError` enum (`KeyNotFound`,
    `AuthenticationFailed`, `UnsupportedAlgorithm`, `MalformedNonce`,
    `ShortCiphertext`).
- `crates/ematix-parquet-crypto/tests/aead_vectors.rs` (NEW):
  decrypt the NIST GCM test vectors (publicly available; embed the
  small subset that covers 128/192/256-bit keys, empty AAD, and
  long AAD).
- `crates/ematix-parquet-crypto/tests/aad_layout.rs` (NEW):
  validate `build_module_aad` byte layout against hand-rolled
  reference bytes for each `ModuleType` × ordinal combination.
- `crates/ematix-parquet-crypto/examples/bench_aes_gcm.rs` (NEW):
  throughput sweep at 4/64/256/1024 KB. Single-line report;
  used to confirm AES is not the bottleneck before each later phase.

**Acceptance.**
1. `cargo test -p ematix-parquet-crypto` green: NIST vectors round-
   trip (encrypt → decrypt → original); AAD layout matches reference
   for each module type.
2. Wrong-tag and wrong-AAD inputs return `AuthenticationFailed`.
3. `bench_aes_gcm` reports ≥ 1 GB/s on the CI runner for 256-KB
   pages.
4. Crate doc-test demonstrates `StaticKeys` → `seal` → `open` flow.
5. Workspace build (`cargo build --workspace`) stays green; new
   crate appears in `cargo metadata`.

**Blocks Π.13c, Π.13d, Π.13e, Π.13f.**

---

### Π.13c — Read path: plaintext-footer mode

**Goal.** Decrypt encrypted column chunks in a file whose footer is
plaintext. Smallest readable PME milestone — the footer parses
normally (Π.13a), so we only need to wire the per-page decrypt and
the AAD construction for data/dict/page-header modules.

**Touches.**
- `crates/ematix-parquet-codec/Cargo.toml`: add `encryption` feature
  per D10, with `dep:ematix-parquet-crypto` (path + version pin).
- `crates/ematix-parquet-codec/src/encrypted.rs` (NEW, cfg-gated):
  - `pub struct EncryptedColumnContext { key, aad_prefix,
    aad_file_unique, rg_ordinal, col_ordinal }`.
  - `fn decrypt_page_header(bytes, ctx, page_ordinal) -> Result<Vec<u8>>`
    — returns the plaintext header bytes; caller parses via
    `read_page_header` as today.
  - `fn decrypt_page_body(bytes, ctx, module: ModuleType,
    page_ordinal) -> Result<Vec<u8>>`.
- `crates/ematix-parquet-codec/src/read.rs`:
  - In `decode_chunk_into` (the per-chunk walker), when
    `column_chunk.crypto_metadata` is `Some`, resolve key via the
    caller-supplied retriever, build `EncryptedColumnContext`,
    and dispatch through `encrypted::decrypt_page_*` before the
    existing decompress + decode path.
  - Add a `ReadOptions` struct with optional `key_retriever:
    Option<Arc<dyn KeyRetriever>>`. New `read_column_*_encrypted`
    convenience entry points + an extended `read_column_*_with_options`
    that accepts the struct (we already need this shape for Π.14
    adaptive dispatch — kill two birds).
- `crates/ematix-parquet-codec/tests/encryption_plaintext_footer_oracle.rs`
  (NEW): parquet-rs writes a 3-column file with plaintext footer
  + per-column encryption (one column encrypted, two plaintext);
  our reader reads all three identically to parquet-rs's decoded
  output. Cover i32, i64, byte_array. Cover both V1 and V2 data
  pages (the V2 case validates that rep/def stay plaintext).

**Acceptance.**
1. Round-trip oracle: 6 tests in
   `encryption_plaintext_footer_oracle.rs` covering (i32, i64,
   byte_array) × (DataPageV1, DataPageV2).
2. Wrong key → `CryptoError::AuthenticationFailed` (not a panic).
3. Missing retriever on an encrypted column → clean
   `ReadError::KeyRetrieverRequired`.
4. Unencrypted columns on the same file decode without invoking
   the retriever (verified via a counting test-double retriever).
5. `cargo build --workspace` (default features) does not pull
   `ematix-parquet-crypto`. CI matrix adds a job for
   `--features encryption`.

**Blocks Π.13d** for AAD-wiring confidence (encrypted-footer mode
reuses the same AAD builder + the same per-page decrypt; if it's
wrong here, it's wrong there too).

---

### Π.13d — Read path: encrypted-footer mode

**Goal.** Recognize `PARE` magic, decrypt the footer module, then
proceed identically to Π.13c for the column data.

**Touches.**
- `crates/ematix-parquet-io/src/file.rs`: extend the bootstrap that
  reads the last 8 bytes to recognize `PARE`. When `PARE`, set a
  flag on `ParquetFile` (`footer_encrypted: bool`); the caller is
  responsible for passing a footer key on the first metadata access.
  - Add `ParquetFile::open_with_footer_key(path, key)` or extend
    the existing `open` to return a "needs decryption" sentinel that
    the codec resolves. **Decision: new method.** Keeps `open` infallible
    on plaintext files.
- `crates/ematix-parquet-format/src/metadata.rs`: nothing new (Π.13a
  shipped the `FileCryptoMetaData` parser). The encrypted-footer
  trailer layout is `[FileCryptoMetaData Thrift][encrypted FileMetaData][len: u32 LE][PARE]`.
  Add a top-level `read_encrypted_footer_trailer(bytes) ->
  (FileCryptoMetaData<'a>, &'a [u8])` that returns the parsed
  crypto-metadata + the still-encrypted footer slice.
- `crates/ematix-parquet-codec/src/encrypted.rs`: `decrypt_footer(
  encrypted_bytes, footer_key, aad_prefix, aad_file_unique) -> Vec<u8>`.
- `crates/ematix-parquet-codec/src/read.rs`: when `ParquetFile`
  reports `footer_encrypted`, route through `decrypt_footer` before
  `read_file_metadata`. The decrypted bytes are the plaintext footer
  the rest of the pipeline expects.
- `crates/ematix-parquet-codec/tests/encryption_encrypted_footer_oracle.rs`
  (NEW): parquet-rs writes an encrypted-footer file; we read with
  the right footer key. Wrong footer key → AuthenticationFailed.
  Missing footer key on a `PARE` file → `MissingFooterKey`.

**Acceptance.**
1. 4 tests in `encryption_encrypted_footer_oracle.rs`: full-footer-
   encrypted + per-column-key file; same with footer-key-only (all
   columns use the footer key); wrong footer key; missing footer
   key.
2. `PAR1` files continue to open without a footer key — no
   regression on the existing oracle tests.
3. The `FileCryptoMetaData` carried in the trailer round-trips: its
   `aad_prefix` and `aad_file_unique` thread through to per-page
   decryption identically to the plaintext-footer case.

**Blocks Π.13e** (the writer's plaintext-footer oracle test needs
to read back via our own reader as well as parquet-rs; if Π.13d's
encrypted-footer reader is wrong, Π.13f's writer test breaks too).

---

### Π.13e — Write path: plaintext-footer mode

**Goal.** Emit a file with plaintext footer + per-column encryption.
First write milestone; the smaller of the two write surfaces (no
footer encryption, no `PARE` magic — just the per-column-chunk
machinery + the new Thrift fields on the footer).

**Touches.**
- `crates/ematix-parquet-format/src/metadata_writer.rs`: writer
  fns for `EncryptionAlgorithm`, `FileCryptoMetaData`,
  `ColumnCryptoMetaData`, plus the new optional fields on
  `FileMetaData` (field 8/9) and `ColumnChunk` (field 8/9). Most of
  this was sketched in Π.13a; this is the second half (actually
  emitting it).
- `crates/ematix-parquet-codec/src/write.rs`: `WriteEncryption`
  config per D7; new entry point `write_table_to_path_encrypted`
  (or extend `write_table_with_options`). For each column with a
  key:
  - For each page: build the AAD module (module type +
    rg_ordinal + col_ordinal + page_ordinal); seal the compressed
    body; emit `[length: u32 LE][nonce: 12B][ciphertext+tag]` instead
    of the raw compressed body.
  - Encrypt the data page header symmetrically (separate AAD module
    type) and emit length-prefixed.
  - Skip page-index + bloom emission on encrypted columns (spec
    technically requires their own encryption keys; punt).
- `crates/ematix-parquet-codec/tests/encryption_write_plaintext_footer_oracle.rs`
  (NEW): round-trip oracle in both directions.
  - **Outbound**: we write, parquet-rs reads. Cover i32, i64,
    byte_array; one column per file (per-column-key shape), then
    multi-column with mixed encrypted + plaintext.
  - **Inbound**: we write, we read back via Π.13c. Establishes the
    cross-reader oracle is symmetric.

**Acceptance.**
1. 8 tests in `encryption_write_plaintext_footer_oracle.rs`: 3
   per-type × {parquet-rs read, our read} = 6; plus mixed
   encrypted/plaintext multi-column; plus DataPageV2 encrypted.
2. File-format validity: `parquet-tools` (or
   `parquet::file::reader::SerializedFileReader`) opens the file
   without error and lists the encryption algorithm + per-column
   crypto metadata identically to a parquet-rs-written equivalent.
3. Unencrypted-column path on a file with `WriteEncryption` (some
   columns absent from `column_keys`) produces byte-identical
   plaintext column chunks to a non-encrypted write of the same
   data.
4. Writes WITHOUT `WriteEncryption` (i.e. existing callers) are
   byte-identical to today.

**Blocks Π.13f** (encrypted-footer write reuses the per-page
encryption pipeline).

---

### Π.13f — Write path: encrypted-footer mode + key-rotation example

**Goal.** Emit `PARE`-trailed files (footer is itself encrypted).
Plus document and exemplify key rotation as a read-decrypt-write
flow.

**Touches.**
- `crates/ematix-parquet-codec/src/write.rs`: when
  `WriteEncryption.mode == EncryptedFooter`, after the row groups
  are emitted, build the plaintext footer Thrift bytes as today,
  then GCM-seal them with the footer key, then emit
  `[FileCryptoMetaData Thrift][sealed bytes][len: u32 LE][PARE]`
  instead of `[footer][len][PAR1]`.
- `crates/ematix-parquet-format/src/metadata_writer.rs`: helper
  `encode_file_crypto_metadata(algorithm, key_metadata) -> Vec<u8>`.
- `crates/ematix-parquet-codec/examples/key_rotation.rs` (NEW):
  end-to-end example. Reads a file with old keys (Π.13c/d), writes
  the same columns under new keys (Π.13e/f), confirms the new file
  decodes with new keys and not old keys. Documented in the
  example's header comment as the canonical rotation recipe.
- `crates/ematix-parquet-codec/tests/encryption_write_encrypted_footer_oracle.rs`
  (NEW): 4 tests mirroring Π.13e — outbound (parquet-rs reads),
  inbound (we read via Π.13d), key-rotation flow round-trips.
- `docs/ENCRYPTION.md` (NEW, short): consumer-facing usage doc.
  Lists the two modes; shows `KeyRetriever` + `WriteEncryption`
  shapes; points at the rotation example. ~80 lines.

**Acceptance.**
1. 4 tests in `encryption_write_encrypted_footer_oracle.rs` green.
2. `key_rotation` example runs to completion under
   `cargo run --release --example key_rotation`.
3. parquet-rs reads our encrypted-footer files (with the right
   footer key) and recovers every value bit-identically.
4. A test that writes encrypted-footer then opens with `ParquetFile::open`
   (no footer key) returns `MissingFooterKey` from the codec layer,
   not a panic.

**Blocks Π.13g.**

---

### Π.13g — Release wiring: feature-flag CI, README, v0.6.0 tag

**Goal.** Ship v0.6.0.

**Touches.**
- `.github/workflows/ci.yml`: add `cargo test --workspace --features
  encryption` to the matrix. Also keep the default-features job
  green (proves the feature gate works and the crypto deps don't
  leak into the default build).
- `.github/workflows/release.yml`: extend the publish order to
  `format → io → crypto → codec → async`. Add `sleep 45` between
  crypto and codec per the existing pattern.
- `README.md`: new "Encryption" section under Features, ~30 lines.
  Mention both modes, the `KeyRetriever` shape, the feature flag,
  and the rotation example. Link `docs/ENCRYPTION.md`.
- `docs/RELEASING.md`: update 4-crate publish order → 5-crate.
- `Cargo.toml` (workspace): bump version to `0.6.0`.
- `docs/plans/CURRENT.md`: mark Π.13 ✓ DONE with the standard
  Shipped / Touches / Acceptance block. Move the Π.13 detail block
  into the "Shipped" position; move Π.14 to "active".

**Acceptance.**
1. `cargo test --workspace --all-targets` green (default features).
2. `cargo test --workspace --features encryption --all-targets`
   green.
3. `cargo build --workspace` produces no `aes-gcm` or `aes` in the
   default-feature dep tree (verified by `cargo tree`).
4. `cargo publish --dry-run -p ematix-parquet-crypto` succeeds.
5. v0.6.0 tag pushed; release workflow publishes all 5 crates.

---

## Open questions / assumptions to validate

| # | Question | Default if not answered |
| --- | --- | --- |
| Q1 | Does the `aes-gcm` crate's ARM Crypto Extensions path actually fire on M-series in our default build profile, or do we need an explicit `RUSTFLAGS="-Ctarget-cpu=apple-m1"` knob? | Verify in Π.13b's bench example. If it doesn't auto-detect, document the flag in `docs/ENCRYPTION.md`; do not gate on this. |
| Q2 | Should `ReadOptions` accept a `Box<dyn KeyRetriever>` or an `Arc<dyn KeyRetriever>`? | `Arc` — read paths fan out to RG-parallel decode (Π.15 someday); shared ownership is the right default. |
| Q3 | Encrypted page-index + bloom filter on read: silently skip, or error? | Silently skip with a warning logged via the existing error type's `Display`. Full support is a tracked Π.13-followup; erroring would block consumers who use neither. |
| Q4 | Does writing an encrypted file imply stats are encrypted too? | Per spec, `ColumnMetaData` (which contains `Statistics`) is the encrypted unit in encrypted-footer mode — yes. In plaintext-footer mode, `encrypted_column_metadata` on the `ColumnChunk` is the encrypted Thrift-encoded `ColumnMetaData`, so stats are encrypted iff the column is. The writer must emit stats inside the encrypted blob, not in the plaintext shell. |
| Q5 | What happens if a caller decrypts with the wrong footer key and the resulting bytes are *valid Thrift* by accident? | GCM authentication tag detects this with 2^-128 probability; we error before Thrift even sees it. No additional plumbing. |
| Q6 | Do we need a separate `signing` mode (plaintext data + HMAC tag)? | No — not in the spec. PME is encrypt-and-authenticate; there is no auth-only mode. |

## Risks

- **Spec ambiguity on AAD ordering for DataPageV2.** The spec text is
  clear that only the values portion is encrypted, but doesn't explicitly
  list the AAD construction for V2 vs V1 separately. We validate
  against parquet-rs's emitted bytes during Π.13c (read-side oracle
  catches mismatches immediately).
- **parquet-rs PME test fixture surface.** parquet-rs's encryption
  support has been GA for a year; fixtures exist but the API surface
  has churned. Pin to a known parquet-rs version in dev-deps and
  document the pin (matches the existing pinning pattern for `parquet
  = "58"`).
- **Crypto-crate version drift.** `aes-gcm` 0.10 → 0.11 will be
  breaking; track and budget a bump every 12-18 months like
  `object_store`.

## Out-of-scope follow-ups (not part of Π.13)

- `AES_GCM_CTR_V1` algorithm support (D4).
- Encrypted page-index + encrypted bloom filter.
- Encrypted async reads — should "just work" through `ematix-parquet-async`
  once the codec supports them, but a separate oracle test
  exercising an encrypted file over `LocalFileSystem` ObjectStore is
  worth a half-day in v0.6.1.
- KMS integrations: caller's problem; we provide the trait.
- `ematix-parquet-keytool` CLI for rotation, key-list, etc.
- Footer-only encryption mode where data pages stay plaintext (the
  spec allows it; it's a niche shape; defer until requested).
