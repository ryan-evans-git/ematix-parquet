//! Sidecar-index manifest: the [`IndexManifest`] type and its JSON
//! wire format, plus the [`SourceFingerprint`] that ties a sidecar to
//! a specific state of its source `.parquet` file.
//!
//! ## Wire format
//!
//! The manifest is JSON, stored in the sidecar's footer
//! `KeyValueMetadata` under the key [`MANIFEST_KEY`] (currently
//! `"ematix_index_manifest_v1"`). JSON instead of Thrift because:
//!
//! - It's the same shape spark / iceberg / hudi already use for
//!   manifests, so external tools can read it without a thrift dep.
//! - The codec doesn't pull `serde` today; a 200-line hand-rolled
//!   encoder/decoder for this fixed shape is cheaper than the
//!   `serde + serde_json` dep weight.
//! - The on-disk size of the manifest is tiny (sub-1 KB per index).
//!   Compactness doesn't matter.
//!
//! Forward compat: if a future sidecar carries
//! `ematix_index_manifest_v2`, the v1 reader returns
//! [`ManifestError::UnsupportedVersion`] and refuses to answer
//! lookups. Sidecar producers MUST bump the key suffix on any
//! breaking change.
//!
//! ## Fingerprint
//!
//! [`SourceFingerprint`] captures four scalars from the source file's
//! footer: byte length, CRC32 of the raw footer bytes, total
//! `num_rows`, and `num_row_groups`. Together these are enough to
//! detect any rewrite that would shift page offsets or row counts;
//! it's not a cryptographic hash and not intended as one — a
//! malicious producer could craft a collision, but the only effect
//! is the reader serving stale results to itself.

use std::fmt;

/// Top-level KV-metadata key under which the manifest JSON lives.
/// Bumped on any breaking change to the JSON shape.
pub const MANIFEST_KEY: &str = "ematix_index_manifest_v1";

/// Symbolic version string embedded in the manifest payload. Matches
/// the suffix of [`MANIFEST_KEY`]. Readers compare on the suffix, not
/// this field — the field is informational only and never
/// authoritative.
pub const MANIFEST_VERSION: &str = "v1";

/// One sidecar manifest = source fingerprint + list of indexes.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexManifest {
    pub source_fingerprint: SourceFingerprint,
    pub indexes: Vec<IndexEntry>,
}

/// Identifies a specific state of the source `.parquet` file. The
/// sidecar reader rejects on any mismatch — sidecars are tied to a
/// specific footer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceFingerprint {
    /// Byte length of the source file's serialized footer (the
    /// thrift-encoded `FileMetaData`, exclusive of the 4-byte length
    /// prefix and the trailing `PAR1` magic).
    pub footer_length: u32,
    /// CRC32 (IEEE) of the same footer bytes. Cheap collision check
    /// over the entire metadata block — picks up writes that alter
    /// page offsets, schema, statistics, or row-group counts.
    pub footer_crc32: u32,
    /// `FileMetaData.num_rows`. Belt-and-braces against a CRC
    /// collision that happens to preserve row count.
    pub num_rows: i64,
    /// `FileMetaData.row_groups.len()`.
    pub num_row_groups: u32,
}

/// One index in the manifest. `sidecar_row_group` names which row
/// group of the `.idx` parquet file holds this index's body.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexEntry {
    /// User-chosen name, e.g. `"idx_orderkey"`. Unique within the
    /// manifest. Used as the lookup key in [`crate::index`] APIs.
    pub name: String,
    /// The kind of index — drives the row-group schema and the
    /// lookup algorithm.
    pub kind: IndexKind,
    /// Index of the row group in the sidecar parquet file that
    /// holds this index's data.
    pub sidecar_row_group: u32,
}

/// What kind of index this is. Each variant pins the schema of the
/// sidecar row group and the lookup algorithm.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexKind {
    /// Sorted (B-tree-like) index on one column. Schema:
    /// `(value, target_rg, target_page, target_rowset)`, sorted by
    /// `value` ASC. Equality and range lookup.
    Sorted {
        source_column: String,
        physical_type: PhysicalType,
    },
    /// Per-source-page Bloom filter. Schema:
    /// `(source_rg, source_page, bloom_block)`, sorted by
    /// `(source_rg, source_page)`. Equality-only.
    BloomPage {
        source_column: String,
        target_fpp: f64,
    },
    /// Two-column composite, leading-prefix sorted. Schema:
    /// `(value_a, value_b, target_rg, target_page, target_rowset)`,
    /// sorted by `(value_a, value_b)`.
    CompositePrefix {
        source_columns: Vec<String>,
        physical_types: Vec<PhysicalType>,
    },
    /// Inverted index over a BYTE_ARRAY column. Schema:
    /// `(token, target_rg, target_page, target_rowset)`, sorted by
    /// `token`. The tokenizer is identified by [`Tokenizer`] so a
    /// reader can apply the matching transform to query terms.
    Inverted {
        source_column: String,
        tokenizer: Tokenizer,
    },
}

/// Subset of Parquet physical types we encode index keys over.
/// Mirrors the subset the sorted-index builder supports in MVP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalType {
    Int32,
    Int64,
    ByteArray,
}

/// Identifier of the tokenizer used to build an inverted index.
/// Readers apply the matching transform to query terms before
/// lookup. New variants are additive — older readers see
/// [`ManifestError::UnsupportedTokenizer`] if a sidecar uses one
/// they don't recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tokenizer {
    /// Split on ASCII whitespace; lowercase each token via the
    /// `to_ascii_lowercase` rule. No Unicode normalization.
    WhitespaceLowercaseV1,
}

impl Tokenizer {
    /// Apply this tokenizer to `value`, returning the produced
    /// tokens in source order. Tokens are owned `Vec<u8>` because
    /// most non-trivial tokenizers transform bytes (lowercasing,
    /// stemming, …) and can't borrow.
    ///
    /// Builder calls this for every source row and dedupes per row
    /// before bucketing. Reader calls this on the query string to
    /// normalize it the same way before [`crate::index::ParquetIndex::lookup_token`].
    /// Same tokenizer in → same tokens out, or the index is
    /// corrupt by construction.
    pub fn tokenize(self, value: &[u8]) -> Vec<Vec<u8>> {
        match self {
            Self::WhitespaceLowercaseV1 => whitespace_lowercase_v1(value),
        }
    }
}

/// Split `value` on ASCII whitespace, drop empties, lowercase each
/// token via [`u8::to_ascii_lowercase`]. No Unicode normalization,
/// no stemming, no stop-words. Deliberately tiny — the v1 token
/// shape that bigger English-language tokenizers can fall back to.
fn whitespace_lowercase_v1(value: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    for &b in value {
        if b.is_ascii_whitespace() {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
        } else {
            buf.push(b.to_ascii_lowercase());
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    #[test]
    fn whitespace_lowercase_v1_basic() {
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"Hello World");
        assert_eq!(toks, vec![b"hello".to_vec(), b"world".to_vec()]);
    }

    #[test]
    fn whitespace_lowercase_v1_collapses_runs_of_whitespace() {
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"  foo  \t\n  bar  ");
        assert_eq!(toks, vec![b"foo".to_vec(), b"bar".to_vec()]);
    }

    #[test]
    fn whitespace_lowercase_v1_preserves_inner_punctuation() {
        // No stemming, no punctuation stripping in v1 — that lives
        // in a future v2.
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"don't u.s.a.");
        assert_eq!(toks, vec![b"don't".to_vec(), b"u.s.a.".to_vec()]);
    }

    #[test]
    fn whitespace_lowercase_v1_empty_input() {
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"");
        assert!(toks.is_empty());
    }

    #[test]
    fn whitespace_lowercase_v1_only_whitespace() {
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"   \t  \n ");
        assert!(toks.is_empty());
    }

    #[test]
    fn whitespace_lowercase_v1_unicode_bytes_passed_through_unchanged() {
        // ASCII-only lowercasing — non-ASCII bytes (everything ≥ 0x80,
        // i.e. every byte of a multi-byte UTF-8 codepoint) survive
        // verbatim. "Über" begins with `Ü` which is `0xC3 0x9C` in
        // UTF-8 — neither byte is ASCII, so no fold happens.
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize("Über café".as_bytes());
        assert_eq!(toks.len(), 2);
        // Input "Über" stays bit-identical: the Ü's two bytes are
        // both non-ASCII (0xC3 0x9C), and the trailing `ber` is
        // already lowercase ASCII.
        assert_eq!(toks[0], "Über".as_bytes().to_vec());
        // "café" similarly survives — every byte is either lowercase
        // ASCII or part of a multi-byte codepoint.
        assert_eq!(toks[1], "café".as_bytes().to_vec());
    }

    #[test]
    fn whitespace_lowercase_v1_mixed_case_ascii_only_folds() {
        // Pure ASCII confirms the lowercasing is doing work.
        let toks = Tokenizer::WhitespaceLowercaseV1.tokenize(b"HELLO World FoO");
        assert_eq!(
            toks,
            vec![b"hello".to_vec(), b"world".to_vec(), b"foo".to_vec()]
        );
    }
}

// ============================================================
// Errors
// ============================================================

/// Anything the manifest layer can fail on. Kept narrow — the
/// codec's `CodecError` wraps these at module boundaries.
#[derive(Debug)]
pub enum ManifestError {
    /// `MANIFEST_KEY` was missing from the sidecar footer KV.
    Missing,
    /// `MANIFEST_KEY` was present but the JSON payload couldn't be
    /// parsed.
    Malformed(String),
    /// Manifest's `version` string was something other than [`MANIFEST_VERSION`].
    UnsupportedVersion(String),
    /// Inverted index referenced a tokenizer this reader build
    /// doesn't know about. Indicates either a forward-compat upgrade
    /// or a typo in a producer.
    UnsupportedTokenizer(String),
    /// Computed [`SourceFingerprint`] didn't match the one in the
    /// sidecar's manifest — source file has been rewritten.
    SourceFingerprintMismatch {
        expected: SourceFingerprint,
        actual: SourceFingerprint,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(f, "sidecar manifest key `{MANIFEST_KEY}` missing"),
            Self::Malformed(msg) => write!(f, "sidecar manifest malformed: {msg}"),
            Self::UnsupportedVersion(v) => {
                write!(f, "sidecar manifest version `{v}` unsupported (expected `{MANIFEST_VERSION}`)")
            }
            Self::UnsupportedTokenizer(t) => {
                write!(f, "sidecar manifest references unknown tokenizer `{t}`")
            }
            Self::SourceFingerprintMismatch { expected, actual } => write!(
                f,
                "sidecar fingerprint mismatch: source file changed (expected footer_len={}, crc32={:#x}, num_rows={}, n_rg={}; got footer_len={}, crc32={:#x}, num_rows={}, n_rg={})",
                expected.footer_length,
                expected.footer_crc32,
                expected.num_rows,
                expected.num_row_groups,
                actual.footer_length,
                actual.footer_crc32,
                actual.num_rows,
                actual.num_row_groups,
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

// ============================================================
// JSON encode
// ============================================================

impl IndexManifest {
    /// Serialize to the JSON form stored under [`MANIFEST_KEY`].
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("{\"version\":\"");
        s.push_str(MANIFEST_VERSION);
        s.push_str("\",\"source_fingerprint\":");
        self.source_fingerprint.write_json(&mut s);
        s.push_str(",\"indexes\":[");
        for (i, e) in self.indexes.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            e.write_json(&mut s);
        }
        s.push_str("]}");
        s
    }

    /// Parse from JSON. Errors on missing keys, unknown variants, or
    /// version mismatch — see [`ManifestError`].
    pub fn from_json(s: &str) -> Result<Self, ManifestError> {
        let mut p = JsonParser::new(s);
        p.expect_obj_open()?;

        let mut version: Option<String> = None;
        let mut fp: Option<SourceFingerprint> = None;
        let mut indexes: Option<Vec<IndexEntry>> = None;

        loop {
            if p.try_obj_close()? {
                break;
            }
            let key = p.string()?;
            p.expect_colon()?;
            match key.as_str() {
                "version" => version = Some(p.string()?),
                "source_fingerprint" => fp = Some(SourceFingerprint::parse_json(&mut p)?),
                "indexes" => indexes = Some(parse_indexes(&mut p)?),
                _ => p.skip_value()?, // forward-compat: ignore unknown top-level keys
            }
            p.consume_comma_or_close()?;
        }

        let version =
            version.ok_or_else(|| ManifestError::Malformed("missing `version` key".into()))?;
        if version != MANIFEST_VERSION {
            return Err(ManifestError::UnsupportedVersion(version));
        }
        let source_fingerprint =
            fp.ok_or_else(|| ManifestError::Malformed("missing `source_fingerprint` key".into()))?;
        let indexes =
            indexes.ok_or_else(|| ManifestError::Malformed("missing `indexes` key".into()))?;
        Ok(Self {
            source_fingerprint,
            indexes,
        })
    }
}

impl SourceFingerprint {
    fn write_json(&self, s: &mut String) {
        use std::fmt::Write as _;
        write!(
            s,
            "{{\"footer_length\":{},\"footer_crc32\":{},\"num_rows\":{},\"num_row_groups\":{}}}",
            self.footer_length, self.footer_crc32, self.num_rows, self.num_row_groups
        )
        .unwrap();
    }

    fn parse_json(p: &mut JsonParser<'_>) -> Result<Self, ManifestError> {
        p.expect_obj_open()?;
        let mut footer_length: Option<u32> = None;
        let mut footer_crc32: Option<u32> = None;
        let mut num_rows: Option<i64> = None;
        let mut num_row_groups: Option<u32> = None;
        loop {
            if p.try_obj_close()? {
                break;
            }
            let key = p.string()?;
            p.expect_colon()?;
            match key.as_str() {
                "footer_length" => footer_length = Some(p.u32()?),
                "footer_crc32" => footer_crc32 = Some(p.u32()?),
                "num_rows" => num_rows = Some(p.i64()?),
                "num_row_groups" => num_row_groups = Some(p.u32()?),
                _ => p.skip_value()?,
            }
            p.consume_comma_or_close()?;
        }
        Ok(Self {
            footer_length: footer_length.ok_or_else(|| {
                ManifestError::Malformed("source_fingerprint.footer_length".into())
            })?,
            footer_crc32: footer_crc32.ok_or_else(|| {
                ManifestError::Malformed("source_fingerprint.footer_crc32".into())
            })?,
            num_rows: num_rows
                .ok_or_else(|| ManifestError::Malformed("source_fingerprint.num_rows".into()))?,
            num_row_groups: num_row_groups.ok_or_else(|| {
                ManifestError::Malformed("source_fingerprint.num_row_groups".into())
            })?,
        })
    }
}

impl IndexEntry {
    fn write_json(&self, s: &mut String) {
        use std::fmt::Write as _;
        s.push_str("{\"name\":");
        write_json_string(s, &self.name);
        s.push_str(",\"sidecar_row_group\":");
        write!(s, "{}", self.sidecar_row_group).unwrap();
        s.push(',');
        match &self.kind {
            IndexKind::Sorted {
                source_column,
                physical_type,
            } => {
                s.push_str("\"type\":\"sorted\",\"source_column\":");
                write_json_string(s, source_column);
                s.push_str(",\"physical_type\":\"");
                s.push_str(physical_type.as_str());
                s.push('"');
            }
            IndexKind::BloomPage {
                source_column,
                target_fpp,
            } => {
                s.push_str("\"type\":\"bloom_page\",\"source_column\":");
                write_json_string(s, source_column);
                s.push_str(",\"target_fpp\":");
                write!(s, "{target_fpp}").unwrap();
            }
            IndexKind::CompositePrefix {
                source_columns,
                physical_types,
            } => {
                s.push_str("\"type\":\"composite_prefix\",\"source_columns\":[");
                for (i, c) in source_columns.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    write_json_string(s, c);
                }
                s.push_str("],\"physical_types\":[");
                for (i, t) in physical_types.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push('"');
                    s.push_str(t.as_str());
                    s.push('"');
                }
                s.push(']');
            }
            IndexKind::Inverted {
                source_column,
                tokenizer,
            } => {
                s.push_str("\"type\":\"inverted\",\"source_column\":");
                write_json_string(s, source_column);
                s.push_str(",\"tokenizer\":\"");
                s.push_str(tokenizer.as_str());
                s.push('"');
            }
        }
        s.push('}');
    }
}

fn parse_indexes(p: &mut JsonParser<'_>) -> Result<Vec<IndexEntry>, ManifestError> {
    p.expect_arr_open()?;
    let mut out = Vec::new();
    loop {
        if p.try_arr_close()? {
            break;
        }
        out.push(parse_one_index(p)?);
        p.consume_comma_or_close_arr()?;
    }
    Ok(out)
}

fn parse_one_index(p: &mut JsonParser<'_>) -> Result<IndexEntry, ManifestError> {
    p.expect_obj_open()?;
    let mut name: Option<String> = None;
    let mut sidecar_row_group: Option<u32> = None;
    let mut typ: Option<String> = None;
    let mut source_column: Option<String> = None;
    let mut physical_type: Option<String> = None;
    let mut target_fpp: Option<f64> = None;
    let mut source_columns: Option<Vec<String>> = None;
    let mut physical_types: Option<Vec<String>> = None;
    let mut tokenizer: Option<String> = None;

    loop {
        if p.try_obj_close()? {
            break;
        }
        let key = p.string()?;
        p.expect_colon()?;
        match key.as_str() {
            "name" => name = Some(p.string()?),
            "sidecar_row_group" => sidecar_row_group = Some(p.u32()?),
            "type" => typ = Some(p.string()?),
            "source_column" => source_column = Some(p.string()?),
            "physical_type" => physical_type = Some(p.string()?),
            "target_fpp" => target_fpp = Some(p.f64()?),
            "source_columns" => source_columns = Some(p.string_array()?),
            "physical_types" => physical_types = Some(p.string_array()?),
            "tokenizer" => tokenizer = Some(p.string()?),
            _ => p.skip_value()?,
        }
        p.consume_comma_or_close()?;
    }

    let name = name.ok_or_else(|| ManifestError::Malformed("index.name".into()))?;
    let sidecar_row_group = sidecar_row_group
        .ok_or_else(|| ManifestError::Malformed("index.sidecar_row_group".into()))?;
    let typ = typ.ok_or_else(|| ManifestError::Malformed("index.type".into()))?;

    let kind = match typ.as_str() {
        "sorted" => IndexKind::Sorted {
            source_column: source_column
                .ok_or_else(|| ManifestError::Malformed("sorted.source_column".into()))?,
            physical_type: PhysicalType::parse(
                &physical_type
                    .ok_or_else(|| ManifestError::Malformed("sorted.physical_type".into()))?,
            )?,
        },
        "bloom_page" => IndexKind::BloomPage {
            source_column: source_column
                .ok_or_else(|| ManifestError::Malformed("bloom_page.source_column".into()))?,
            target_fpp: target_fpp
                .ok_or_else(|| ManifestError::Malformed("bloom_page.target_fpp".into()))?,
        },
        "composite_prefix" => {
            let cols = source_columns.ok_or_else(|| {
                ManifestError::Malformed("composite_prefix.source_columns".into())
            })?;
            let types_strs = physical_types.ok_or_else(|| {
                ManifestError::Malformed("composite_prefix.physical_types".into())
            })?;
            if cols.len() != types_strs.len() {
                return Err(ManifestError::Malformed(
                    "composite_prefix: source_columns and physical_types differ in length".into(),
                ));
            }
            let mut types = Vec::with_capacity(types_strs.len());
            for t in &types_strs {
                types.push(PhysicalType::parse(t)?);
            }
            IndexKind::CompositePrefix {
                source_columns: cols,
                physical_types: types,
            }
        }
        "inverted" => IndexKind::Inverted {
            source_column: source_column
                .ok_or_else(|| ManifestError::Malformed("inverted.source_column".into()))?,
            tokenizer: Tokenizer::parse(
                &tokenizer.ok_or_else(|| ManifestError::Malformed("inverted.tokenizer".into()))?,
            )?,
        },
        other => {
            return Err(ManifestError::Malformed(format!(
                "unknown index type `{other}`"
            )))
        }
    };
    Ok(IndexEntry {
        name,
        kind,
        sidecar_row_group,
    })
}

impl PhysicalType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::ByteArray => "BYTE_ARRAY",
        }
    }
    fn parse(s: &str) -> Result<Self, ManifestError> {
        match s {
            "INT32" => Ok(Self::Int32),
            "INT64" => Ok(Self::Int64),
            "BYTE_ARRAY" => Ok(Self::ByteArray),
            other => Err(ManifestError::Malformed(format!(
                "unsupported physical_type `{other}`"
            ))),
        }
    }
}

impl Tokenizer {
    fn as_str(&self) -> &'static str {
        match self {
            Self::WhitespaceLowercaseV1 => "whitespace_lowercase_v1",
        }
    }
    fn parse(s: &str) -> Result<Self, ManifestError> {
        match s {
            "whitespace_lowercase_v1" => Ok(Self::WhitespaceLowercaseV1),
            other => Err(ManifestError::UnsupportedTokenizer(other.into())),
        }
    }
}

// ============================================================
// Hand-rolled JSON parser
// ============================================================
//
// Just enough to round-trip our own output and accept extra fields
// for forward-compat. Not a general JSON parser — comments, escapes
// beyond `\"` `\\` `\n` `\r` `\t`, and unicode escapes are not
// implemented because the producer is us. If something writes a
// surprise here, the parser errors out rather than guessing.

fn write_json_string(s: &mut String, v: &str) {
    s.push('"');
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                write!(s, "\\u{:04x}", c as u32).unwrap();
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

struct JsonParser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            s: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.s.len() {
            let c = self.s[self.pos];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ManifestError> {
        self.skip_ws();
        if self.pos >= self.s.len() || self.s[self.pos] != b {
            return Err(ManifestError::Malformed(format!(
                "expected `{}` at pos {}",
                b as char, self.pos
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn expect_obj_open(&mut self) -> Result<(), ManifestError> {
        self.expect(b'{')
    }
    fn expect_arr_open(&mut self) -> Result<(), ManifestError> {
        self.expect(b'[')
    }
    fn expect_colon(&mut self) -> Result<(), ManifestError> {
        self.expect(b':')
    }

    fn try_obj_close(&mut self) -> Result<bool, ManifestError> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b'}' {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn try_arr_close(&mut self) -> Result<bool, ManifestError> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b']' {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// After a (key:value) or array element, consume a `,` to advance
    /// or do nothing if the caller's next `try_*_close` will close.
    fn consume_comma_or_close(&mut self) -> Result<(), ManifestError> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b',' {
            self.pos += 1;
        }
        Ok(())
    }
    fn consume_comma_or_close_arr(&mut self) -> Result<(), ManifestError> {
        self.consume_comma_or_close()
    }

    fn string(&mut self) -> Result<String, ManifestError> {
        self.skip_ws();
        self.expect(b'"')?;
        let start = self.pos;
        let mut out = String::new();
        while self.pos < self.s.len() {
            let c = self.s[self.pos];
            if c == b'"' {
                if out.is_empty() {
                    // fast path: no escapes seen
                    let raw = std::str::from_utf8(&self.s[start..self.pos])
                        .map_err(|_| ManifestError::Malformed("non-utf8 in string".into()))?;
                    self.pos += 1;
                    return Ok(raw.to_owned());
                }
                self.pos += 1;
                return Ok(out);
            }
            if c == b'\\' {
                // commit pending bytes
                if out.is_empty() {
                    out.push_str(
                        std::str::from_utf8(&self.s[start..self.pos])
                            .map_err(|_| ManifestError::Malformed("non-utf8 in string".into()))?,
                    );
                }
                self.pos += 1;
                if self.pos >= self.s.len() {
                    return Err(ManifestError::Malformed("trailing backslash".into()));
                }
                match self.s[self.pos] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => {
                        return Err(ManifestError::Malformed(format!(
                            "unsupported escape `\\{}`",
                            other as char
                        )))
                    }
                }
                self.pos += 1;
                continue;
            }
            // accumulate
            if !out.is_empty() {
                let chunk_start = self.pos;
                while self.pos < self.s.len()
                    && self.s[self.pos] != b'"'
                    && self.s[self.pos] != b'\\'
                {
                    self.pos += 1;
                }
                out.push_str(
                    std::str::from_utf8(&self.s[chunk_start..self.pos])
                        .map_err(|_| ManifestError::Malformed("non-utf8 in string".into()))?,
                );
                continue;
            }
            self.pos += 1;
        }
        Err(ManifestError::Malformed("unterminated string".into()))
    }

    fn string_array(&mut self) -> Result<Vec<String>, ManifestError> {
        self.expect_arr_open()?;
        let mut out = Vec::new();
        loop {
            if self.try_arr_close()? {
                break;
            }
            out.push(self.string()?);
            self.consume_comma_or_close_arr()?;
        }
        Ok(out)
    }

    fn number_str(&mut self) -> Result<&'a str, ManifestError> {
        self.skip_ws();
        let start = self.pos;
        if self.pos < self.s.len() && (self.s[self.pos] == b'-' || self.s[self.pos] == b'+') {
            self.pos += 1;
        }
        while self.pos < self.s.len() {
            let c = self.s[self.pos];
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'+' || c == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(ManifestError::Malformed(format!(
                "expected number at pos {}",
                start
            )));
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| ManifestError::Malformed("non-utf8 in number".into()))
    }

    fn u32(&mut self) -> Result<u32, ManifestError> {
        let n = self.number_str()?;
        n.parse()
            .map_err(|_| ManifestError::Malformed(format!("bad u32 `{n}`")))
    }
    fn i64(&mut self) -> Result<i64, ManifestError> {
        let n = self.number_str()?;
        n.parse()
            .map_err(|_| ManifestError::Malformed(format!("bad i64 `{n}`")))
    }
    fn f64(&mut self) -> Result<f64, ManifestError> {
        let n = self.number_str()?;
        n.parse()
            .map_err(|_| ManifestError::Malformed(format!("bad f64 `{n}`")))
    }

    /// Forward-compat: skip a value of unknown shape (string, number,
    /// object, array, true/false/null).
    fn skip_value(&mut self) -> Result<(), ManifestError> {
        self.skip_ws();
        if self.pos >= self.s.len() {
            return Err(ManifestError::Malformed("unexpected EOF".into()));
        }
        match self.s[self.pos] {
            b'"' => {
                let _ = self.string()?;
            }
            b'{' => {
                self.expect_obj_open()?;
                loop {
                    if self.try_obj_close()? {
                        break;
                    }
                    let _ = self.string()?;
                    self.expect_colon()?;
                    self.skip_value()?;
                    self.consume_comma_or_close()?;
                }
            }
            b'[' => {
                self.expect_arr_open()?;
                loop {
                    if self.try_arr_close()? {
                        break;
                    }
                    self.skip_value()?;
                    self.consume_comma_or_close_arr()?;
                }
            }
            b't' | b'f' | b'n' => {
                // skip identifier
                while self.pos < self.s.len() && self.s[self.pos].is_ascii_alphabetic() {
                    self.pos += 1;
                }
            }
            b'-' | b'+' | b'0'..=b'9' => {
                let _ = self.number_str()?;
            }
            c => {
                return Err(ManifestError::Malformed(format!(
                    "unexpected byte `{}` at pos {}",
                    c as char, self.pos
                )))
            }
        }
        Ok(())
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fp() -> SourceFingerprint {
        SourceFingerprint {
            footer_length: 12345,
            footer_crc32: 0xDEAD_BEEF,
            num_rows: 5_000_000,
            num_row_groups: 16,
        }
    }

    fn sample_manifest() -> IndexManifest {
        IndexManifest {
            source_fingerprint: sample_fp(),
            indexes: vec![
                IndexEntry {
                    name: "idx_orderkey".into(),
                    kind: IndexKind::Sorted {
                        source_column: "l_orderkey".into(),
                        physical_type: PhysicalType::Int64,
                    },
                    sidecar_row_group: 0,
                },
                IndexEntry {
                    name: "idx_partkey_bloom".into(),
                    kind: IndexKind::BloomPage {
                        source_column: "l_partkey".into(),
                        target_fpp: 0.01,
                    },
                    sidecar_row_group: 1,
                },
                IndexEntry {
                    name: "idx_shipdate_partkey".into(),
                    kind: IndexKind::CompositePrefix {
                        source_columns: vec!["l_shipdate".into(), "l_partkey".into()],
                        physical_types: vec![PhysicalType::Int32, PhysicalType::Int64],
                    },
                    sidecar_row_group: 2,
                },
                IndexEntry {
                    name: "idx_comment_text".into(),
                    kind: IndexKind::Inverted {
                        source_column: "l_comment".into(),
                        tokenizer: Tokenizer::WhitespaceLowercaseV1,
                    },
                    sidecar_row_group: 3,
                },
            ],
        }
    }

    #[test]
    fn round_trip_all_kinds() {
        let m = sample_manifest();
        let j = m.to_json();
        let back = IndexManifest::from_json(&j).expect("parse");
        assert_eq!(m, back);
    }

    #[test]
    fn round_trip_empty() {
        let m = IndexManifest {
            source_fingerprint: sample_fp(),
            indexes: vec![],
        };
        let j = m.to_json();
        let back = IndexManifest::from_json(&j).expect("parse");
        assert_eq!(m, back);
    }

    #[test]
    fn unknown_top_level_keys_ignored() {
        // Forward-compat: a future producer may add fields the
        // current reader doesn't know. Reader must skip them, not
        // error out.
        let j = r#"{"version":"v1","future_thing":42,
                    "source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4,"future_fp_field":"ignored"},
                    "extras":{"nested":[1,2,3]},
                    "indexes":[]}"#;
        let m = IndexManifest::from_json(j).expect("parse");
        assert_eq!(m.source_fingerprint.footer_length, 1);
        assert_eq!(m.source_fingerprint.num_row_groups, 4);
        assert_eq!(m.indexes.len(), 0);
    }

    #[test]
    fn unknown_index_per_entry_field_ignored() {
        // Same forward-compat rule, but for fields inside an index
        // entry.
        let j = r#"{"version":"v1",
                    "source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4},
                    "indexes":[{"name":"x","sidecar_row_group":0,"type":"sorted","source_column":"c","physical_type":"INT64","future_field":"ok"}]}"#;
        let m = IndexManifest::from_json(j).expect("parse");
        assert_eq!(m.indexes.len(), 1);
        assert_eq!(m.indexes[0].name, "x");
    }

    #[test]
    fn unsupported_version_rejected() {
        let j = r#"{"version":"v2","source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4},"indexes":[]}"#;
        match IndexManifest::from_json(j) {
            Err(ManifestError::UnsupportedVersion(v)) => assert_eq!(v, "v2"),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_tokenizer_rejected() {
        let j = r#"{"version":"v1",
                    "source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4},
                    "indexes":[{"name":"x","sidecar_row_group":0,"type":"inverted","source_column":"c","tokenizer":"klingon_v3"}]}"#;
        match IndexManifest::from_json(j) {
            Err(ManifestError::UnsupportedTokenizer(t)) => assert_eq!(t, "klingon_v3"),
            other => panic!("expected UnsupportedTokenizer, got {other:?}"),
        }
    }

    #[test]
    fn unknown_index_type_rejected() {
        let j = r#"{"version":"v1",
                    "source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4},
                    "indexes":[{"name":"x","sidecar_row_group":0,"type":"future_index","source_column":"c"}]}"#;
        match IndexManifest::from_json(j) {
            Err(ManifestError::Malformed(msg)) => {
                assert!(msg.contains("unknown index type"), "{msg}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn missing_top_level_field_rejected() {
        // No version key.
        let j = r#"{"source_fingerprint":{"footer_length":1,"footer_crc32":2,"num_rows":3,"num_row_groups":4},"indexes":[]}"#;
        match IndexManifest::from_json(j) {
            Err(ManifestError::Malformed(msg)) => assert!(msg.contains("version")),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn strings_with_escapes() {
        let m = IndexManifest {
            source_fingerprint: sample_fp(),
            indexes: vec![IndexEntry {
                name: "weird\"name\\with\nstuff".into(),
                kind: IndexKind::Sorted {
                    source_column: "col\twith\rwhitespace".into(),
                    physical_type: PhysicalType::ByteArray,
                },
                sidecar_row_group: 7,
            }],
        };
        let j = m.to_json();
        let back = IndexManifest::from_json(&j).expect("parse");
        assert_eq!(m, back);
    }

    #[test]
    fn fingerprint_mismatch_display() {
        let err = ManifestError::SourceFingerprintMismatch {
            expected: SourceFingerprint {
                footer_length: 100,
                footer_crc32: 0xAA,
                num_rows: 1000,
                num_row_groups: 2,
            },
            actual: SourceFingerprint {
                footer_length: 200,
                footer_crc32: 0xBB,
                num_rows: 1001,
                num_row_groups: 3,
            },
        };
        let s = format!("{err}");
        assert!(s.contains("footer_len=100"), "{s}");
        assert!(s.contains("footer_len=200"), "{s}");
        assert!(s.contains("num_rows=1001"), "{s}");
    }

    #[test]
    fn physical_type_round_trips() {
        for t in [
            PhysicalType::Int32,
            PhysicalType::Int64,
            PhysicalType::ByteArray,
        ] {
            let s = t.as_str();
            assert_eq!(PhysicalType::parse(s).unwrap(), t);
        }
    }

    #[test]
    fn json_is_well_formed_and_parseable_minimum() {
        // Smallest possible manifest.
        let m = IndexManifest {
            source_fingerprint: SourceFingerprint {
                footer_length: 0,
                footer_crc32: 0,
                num_rows: 0,
                num_row_groups: 0,
            },
            indexes: vec![],
        };
        let j = m.to_json();
        // Sanity: starts with {, ends with }, has the version field.
        assert!(j.starts_with('{'));
        assert!(j.ends_with('}'));
        assert!(j.contains("\"version\":\"v1\""));
    }
}
