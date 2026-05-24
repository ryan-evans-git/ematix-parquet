//! [`EmatixDataFileExtension`] — the JSON blob appended to one
//! Iceberg `data_file` manifest entry.
//!
//! ## Wire format
//!
//! ```jsonc
//! {
//!   "version": "v1",
//!   "sidecar_relative_path": "00000-abc.parquet.idx",
//!   "summaries": [
//!     {
//!       "name": "idx_orderkey",
//!       "min_key": { "type": "i64", "value": 1 },
//!       "max_key": { "type": "i64", "value": 6000000 }
//!     },
//!     {
//!       "name": "idx_partkey_bloom",
//!       "dataset_bloom_hex": "deadbeef..."
//!     }
//!   ]
//! }
//! ```
//!
//! - `min_key` / `max_key` are tagged objects so the planner can
//!   recover the physical type without consulting any sidecar.
//! - `dataset_bloom_hex` is lowercase hex (`%02x` per byte). Hex
//!   instead of base64 to avoid pulling a crate in for one routine.
//! - Unknown top-level / summary fields are *ignored* for
//!   forward-compat. The `version` string MUST match
//!   [`EMATIX_EXTENSION_VERSION`] — producers bump the suffix on any
//!   breaking change.
//!
//! ## Where the JSON lives in Iceberg
//!
//! Iceberg's spec allows two channels for per-`data_file` metadata
//! that's opaque to the catalog:
//!
//! 1. `data_file.key_metadata`: a binary blob. Originally for
//!    encryption-related metadata but the spec doesn't restrict
//!    semantic content.
//! 2. Table-level `properties` keyed by data-file path. Higher
//!    overhead, but survives readers that strip `key_metadata`.
//!
//! Π.21b will choose the channel and wire the encode/decode call
//! sites in `iceberg-rust`. Until then, this crate just produces the
//! bytes; the channel is a configuration concern.

use std::fmt;

use crate::error::{IcebergIndexError, Result};
use crate::summary::{IndexSummary, SummaryKey};

/// The Iceberg property key (or `key_metadata` discriminator) under
/// which the JSON lives. Bumped on any breaking format change.
pub const EMATIX_EXTENSION_KEY: &str = "ematix_index_extension_v1";

/// Symbolic version string embedded in the JSON payload. Producers
/// MUST bump this in lockstep with [`EMATIX_EXTENSION_KEY`]'s suffix
/// — readers refuse non-matching versions.
pub const EMATIX_EXTENSION_VERSION: &str = "v1";

/// The complete blob attached to one Iceberg `data_file` manifest
/// entry. One per data file. Carries:
///
/// - **The path to the per-file sidecar**, relative to the data
///   file's location. Resolving it requires the data file's
///   directory; Π.21b will do that resolution against
///   `data_file.file_path`.
/// - **One [`IndexSummary`] per index** the writer chose to summarize.
///   The summary set is a strict subset of (or equal to) the
///   sidecar's `IndexManifest.indexes` set — indexes that don't
///   summarize well (e.g. inverted indexes over very high-entropy
///   text) can be skipped here without breaking correctness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmatixDataFileExtension {
    /// Path to the sidecar, relative to the data file's directory.
    /// Allowed to contain `/`; resolving is the caller's job.
    pub sidecar_relative_path: String,
    /// File-level summaries by index name. Order is preserved on
    /// round-trip; sets with the same names in different order
    /// compare unequal.
    pub summaries: Vec<IndexSummary>,
}

impl EmatixDataFileExtension {
    /// Find the summary for an index by name. `O(n)`; the per-file
    /// summary count is small (single-digit typical).
    pub fn summary(&self, name: &str) -> Option<&IndexSummary> {
        self.summaries.iter().find(|s| s.name == name)
    }

    /// Serialize to the JSON form documented at the module level.
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(128 + self.summaries.len() * 80);
        s.push_str("{\"version\":\"");
        s.push_str(EMATIX_EXTENSION_VERSION);
        s.push_str("\",\"sidecar_relative_path\":");
        write_json_string(&mut s, &self.sidecar_relative_path);
        s.push_str(",\"summaries\":[");
        for (i, summary) in self.summaries.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write_summary(&mut s, summary);
        }
        s.push_str("]}");
        s
    }

    /// Parse from JSON. Errors on missing required keys, unknown
    /// version, or shape violations. Unknown extra fields are
    /// ignored — forward-compat reserves room for additive
    /// extensions inside `v1`.
    pub fn from_json(s: &str) -> Result<Self> {
        let mut p = JsonParser::new(s);
        p.expect_obj_open()?;
        let mut version: Option<String> = None;
        let mut sidecar_relative_path: Option<String> = None;
        let mut summaries: Option<Vec<IndexSummary>> = None;
        loop {
            if p.try_obj_close()? {
                break;
            }
            let key = p.string()?;
            p.expect_colon()?;
            match key.as_str() {
                "version" => version = Some(p.string()?),
                "sidecar_relative_path" => sidecar_relative_path = Some(p.string()?),
                "summaries" => summaries = Some(parse_summaries(&mut p)?),
                _ => p.skip_value()?,
            }
            p.consume_comma_or_close()?;
        }
        let version =
            version.ok_or_else(|| IcebergIndexError::Malformed("missing `version`".into()))?;
        if version != EMATIX_EXTENSION_VERSION {
            return Err(IcebergIndexError::UnsupportedVersion(version));
        }
        let sidecar_relative_path = sidecar_relative_path.ok_or_else(|| {
            IcebergIndexError::Malformed("missing `sidecar_relative_path`".into())
        })?;
        let summaries = summaries.unwrap_or_default();
        Ok(Self {
            sidecar_relative_path,
            summaries,
        })
    }
}

// ============================================================
// Summary JSON
// ============================================================

fn write_summary(s: &mut String, summary: &IndexSummary) {
    s.push_str("{\"name\":");
    write_json_string(s, &summary.name);
    if let Some(min) = &summary.min_key {
        s.push_str(",\"min_key\":");
        write_summary_key(s, min);
    }
    if let Some(max) = &summary.max_key {
        s.push_str(",\"max_key\":");
        write_summary_key(s, max);
    }
    if let Some(blob) = &summary.dataset_bloom {
        s.push_str(",\"dataset_bloom_hex\":\"");
        write_hex(s, blob);
        s.push('"');
    }
    s.push('}');
}

fn write_summary_key(s: &mut String, key: &SummaryKey) {
    use std::fmt::Write as _;
    match key {
        SummaryKey::I64(v) => write!(s, "{{\"type\":\"i64\",\"value\":{v}}}").unwrap(),
        SummaryKey::I32(v) => write!(s, "{{\"type\":\"i32\",\"value\":{v}}}").unwrap(),
        SummaryKey::Bytes(b) => {
            s.push_str("{\"type\":\"bytes\",\"value_hex\":\"");
            write_hex(s, b);
            s.push_str("\"}");
        }
    }
}

fn parse_summaries(p: &mut JsonParser<'_>) -> Result<Vec<IndexSummary>> {
    p.expect_arr_open()?;
    let mut out = Vec::new();
    loop {
        if p.try_arr_close()? {
            break;
        }
        out.push(parse_one_summary(p)?);
        p.consume_comma_or_close()?;
    }
    Ok(out)
}

fn parse_one_summary(p: &mut JsonParser<'_>) -> Result<IndexSummary> {
    p.expect_obj_open()?;
    let mut name: Option<String> = None;
    let mut min_key: Option<SummaryKey> = None;
    let mut max_key: Option<SummaryKey> = None;
    let mut dataset_bloom: Option<Vec<u8>> = None;
    loop {
        if p.try_obj_close()? {
            break;
        }
        let key = p.string()?;
        p.expect_colon()?;
        match key.as_str() {
            "name" => name = Some(p.string()?),
            "min_key" => min_key = Some(parse_summary_key(p)?),
            "max_key" => max_key = Some(parse_summary_key(p)?),
            "dataset_bloom_hex" => {
                let hex = p.string()?;
                dataset_bloom = Some(parse_hex(&hex)?);
            }
            _ => p.skip_value()?,
        }
        p.consume_comma_or_close()?;
    }
    let name = name.ok_or_else(|| IcebergIndexError::Malformed("summary.name".into()))?;
    Ok(IndexSummary {
        name,
        min_key,
        max_key,
        dataset_bloom,
    })
}

fn parse_summary_key(p: &mut JsonParser<'_>) -> Result<SummaryKey> {
    p.expect_obj_open()?;
    let mut typ: Option<String> = None;
    let mut int_value: Option<i64> = None;
    let mut bytes_value: Option<Vec<u8>> = None;
    loop {
        if p.try_obj_close()? {
            break;
        }
        let key = p.string()?;
        p.expect_colon()?;
        match key.as_str() {
            "type" => typ = Some(p.string()?),
            "value" => int_value = Some(p.i64()?),
            "value_hex" => {
                let hex = p.string()?;
                bytes_value = Some(parse_hex(&hex)?);
            }
            _ => p.skip_value()?,
        }
        p.consume_comma_or_close()?;
    }
    let typ = typ.ok_or_else(|| IcebergIndexError::Malformed("summary_key.type".into()))?;
    match typ.as_str() {
        "i64" => {
            let v = int_value
                .ok_or_else(|| IcebergIndexError::Malformed("summary_key.value (i64)".into()))?;
            Ok(SummaryKey::I64(v))
        }
        "i32" => {
            let v = int_value
                .ok_or_else(|| IcebergIndexError::Malformed("summary_key.value (i32)".into()))?;
            let v32 = i32::try_from(v).map_err(|_| {
                IcebergIndexError::Malformed(format!("summary_key.value `{v}` out of i32 range"))
            })?;
            Ok(SummaryKey::I32(v32))
        }
        "bytes" => {
            let b = bytes_value.ok_or_else(|| {
                IcebergIndexError::Malformed("summary_key.value_hex (bytes)".into())
            })?;
            Ok(SummaryKey::Bytes(b))
        }
        other => Err(IcebergIndexError::Malformed(format!(
            "unknown summary_key.type `{other}`"
        ))),
    }
}

// ============================================================
// Hex encode/decode (no crate dep)
// ============================================================

fn write_hex(s: &mut String, bytes: &[u8]) {
    use std::fmt::Write as _;
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
}

fn parse_hex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(IcebergIndexError::Malformed(format!(
            "hex string length {} is odd",
            s.len()
        )));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(10 + c - b'a'),
        b'A'..=b'F' => Ok(10 + c - b'A'),
        _ => Err(IcebergIndexError::Malformed(format!(
            "invalid hex byte `{}`",
            c as char
        ))),
    }
}

// ============================================================
// JSON helpers
// ============================================================
//
// Same shape as the codec's `index::manifest` parser — see that file
// for the rationale. Just enough to round-trip our own output and
// accept extra fields for forward-compat. Not a general JSON parser.

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

    fn expect(&mut self, b: u8) -> Result<()> {
        self.skip_ws();
        if self.pos >= self.s.len() || self.s[self.pos] != b {
            return Err(IcebergIndexError::Malformed(format!(
                "expected `{}` at pos {}",
                b as char, self.pos
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn expect_obj_open(&mut self) -> Result<()> {
        self.expect(b'{')
    }
    fn expect_arr_open(&mut self) -> Result<()> {
        self.expect(b'[')
    }
    fn expect_colon(&mut self) -> Result<()> {
        self.expect(b':')
    }

    fn try_obj_close(&mut self) -> Result<bool> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b'}' {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn try_arr_close(&mut self) -> Result<bool> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b']' {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn consume_comma_or_close(&mut self) -> Result<()> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b',' {
            self.pos += 1;
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String> {
        self.skip_ws();
        self.expect(b'"')?;
        let start = self.pos;
        let mut out = String::new();
        while self.pos < self.s.len() {
            let c = self.s[self.pos];
            if c == b'"' {
                if out.is_empty() {
                    let raw = std::str::from_utf8(&self.s[start..self.pos])
                        .map_err(|_| IcebergIndexError::Malformed("non-utf8 in string".into()))?;
                    self.pos += 1;
                    return Ok(raw.to_owned());
                }
                self.pos += 1;
                return Ok(out);
            }
            if c == b'\\' {
                if out.is_empty() {
                    out.push_str(
                        std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| {
                            IcebergIndexError::Malformed("non-utf8 in string".into())
                        })?,
                    );
                }
                self.pos += 1;
                if self.pos >= self.s.len() {
                    return Err(IcebergIndexError::Malformed("trailing backslash".into()));
                }
                match self.s[self.pos] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => {
                        return Err(IcebergIndexError::Malformed(format!(
                            "unsupported escape `\\{}`",
                            other as char
                        )))
                    }
                }
                self.pos += 1;
                continue;
            }
            if !out.is_empty() {
                out.push(c as char);
            }
            self.pos += 1;
        }
        Err(IcebergIndexError::Malformed("unterminated string".into()))
    }

    fn i64(&mut self) -> Result<i64> {
        self.skip_ws();
        let start = self.pos;
        if self.pos < self.s.len() && self.s[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let raw = std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| IcebergIndexError::Malformed("non-utf8 in number".into()))?;
        raw.parse::<i64>()
            .map_err(|e| IcebergIndexError::Malformed(format!("i64 parse `{raw}`: {e}")))
    }

    /// Skip past one value (string, number, true/false/null, object,
    /// array). Used to ignore forward-compat extra fields.
    fn skip_value(&mut self) -> Result<()> {
        self.skip_ws();
        if self.pos >= self.s.len() {
            return Err(IcebergIndexError::Malformed("unexpected end".into()));
        }
        match self.s[self.pos] {
            b'"' => {
                let _ = self.string()?;
            }
            b'{' => {
                self.expect(b'{')?;
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
                self.expect(b'[')?;
                loop {
                    if self.try_arr_close()? {
                        break;
                    }
                    self.skip_value()?;
                    self.consume_comma_or_close()?;
                }
            }
            b't' => {
                if self.s.get(self.pos..self.pos + 4) == Some(b"true") {
                    self.pos += 4;
                } else {
                    return Err(IcebergIndexError::Malformed("expected `true`".into()));
                }
            }
            b'f' => {
                if self.s.get(self.pos..self.pos + 5) == Some(b"false") {
                    self.pos += 5;
                } else {
                    return Err(IcebergIndexError::Malformed("expected `false`".into()));
                }
            }
            b'n' => {
                if self.s.get(self.pos..self.pos + 4) == Some(b"null") {
                    self.pos += 4;
                } else {
                    return Err(IcebergIndexError::Malformed("expected `null`".into()));
                }
            }
            c if c == b'-' || c.is_ascii_digit() => {
                let _ = self.i64()?;
            }
            other => {
                return Err(IcebergIndexError::Malformed(format!(
                    "unexpected char `{}` at pos {}",
                    other as char, self.pos
                )));
            }
        }
        Ok(())
    }
}

// ============================================================
// Debug pretty-print for golden tests
// ============================================================

impl fmt::Display for EmatixDataFileExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_extension() -> EmatixDataFileExtension {
        EmatixDataFileExtension {
            sidecar_relative_path: "00000-abc.parquet.idx".into(),
            summaries: vec![
                IndexSummary::new("idx_orderkey")
                    .with_range(SummaryKey::I64(1), SummaryKey::I64(6_000_000)),
                IndexSummary {
                    name: "idx_partkey_bloom".into(),
                    min_key: None,
                    max_key: None,
                    dataset_bloom: Some(vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF]),
                },
                IndexSummary::new("idx_shipdate")
                    .with_range(SummaryKey::I32(1992 * 365), SummaryKey::I32(1998 * 365)),
                IndexSummary::new("idx_comment").with_range(
                    SummaryKey::Bytes(b"alpha".to_vec()),
                    SummaryKey::Bytes(b"zebra".to_vec()),
                ),
            ],
        }
    }

    #[test]
    fn json_round_trip_full_extension() {
        let ext = full_extension();
        let json = ext.to_json();
        let parsed = EmatixDataFileExtension::from_json(&json).unwrap();
        assert_eq!(parsed, ext);
    }

    #[test]
    fn json_round_trip_empty_summaries() {
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "sub/x.idx".into(),
            summaries: vec![],
        };
        let json = ext.to_json();
        let parsed = EmatixDataFileExtension::from_json(&json).unwrap();
        assert_eq!(parsed, ext);
    }

    #[test]
    fn json_omits_absent_optional_fields() {
        // A summary with no bounds + no bloom serializes to just {"name": ...}.
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "a.idx".into(),
            summaries: vec![IndexSummary::new("idx_x")],
        };
        let json = ext.to_json();
        assert!(json.contains("\"name\":\"idx_x\""));
        assert!(!json.contains("min_key"));
        assert!(!json.contains("max_key"));
        assert!(!json.contains("dataset_bloom_hex"));
    }

    #[test]
    fn json_unknown_version_errors() {
        let bad = r#"{"version":"v99","sidecar_relative_path":"x.idx","summaries":[]}"#;
        let err = EmatixDataFileExtension::from_json(bad).unwrap_err();
        assert!(matches!(err, IcebergIndexError::UnsupportedVersion(_)));
    }

    #[test]
    fn json_missing_version_errors() {
        let bad = r#"{"sidecar_relative_path":"x.idx","summaries":[]}"#;
        let err = EmatixDataFileExtension::from_json(bad).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn json_missing_sidecar_path_errors() {
        let bad = r#"{"version":"v1","summaries":[]}"#;
        let err = EmatixDataFileExtension::from_json(bad).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn json_unknown_top_level_field_is_ignored() {
        // Forward-compat: a producer adds `created_at_ms`, an older
        // reader still parses the rest.
        let s = r#"{"version":"v1","created_at_ms":1234567890,"sidecar_relative_path":"x.idx","summaries":[]}"#;
        let parsed = EmatixDataFileExtension::from_json(s).unwrap();
        assert_eq!(parsed.sidecar_relative_path, "x.idx");
        assert!(parsed.summaries.is_empty());
    }

    #[test]
    fn json_unknown_summary_field_is_ignored() {
        let s = r#"{"version":"v1","sidecar_relative_path":"x.idx","summaries":[{"name":"idx_x","extra":42}]}"#;
        let parsed = EmatixDataFileExtension::from_json(s).unwrap();
        assert_eq!(parsed.summaries.len(), 1);
        assert_eq!(parsed.summaries[0].name, "idx_x");
    }

    #[test]
    fn json_unknown_summary_key_type_errors() {
        // A future producer adding "f64" trips an older reader. That's
        // by design — float bounds need actual handling.
        let s = r#"{"version":"v1","sidecar_relative_path":"x.idx","summaries":[
            {"name":"idx_x","min_key":{"type":"f64","value":3}}
        ]}"#;
        let err = EmatixDataFileExtension::from_json(s).unwrap_err();
        assert!(matches!(err, IcebergIndexError::Malformed(_)));
    }

    #[test]
    fn json_unknown_summary_key_i32_out_of_range_errors() {
        let s = format!(
            r#"{{"version":"v1","sidecar_relative_path":"x.idx","summaries":[
                {{"name":"idx_x","min_key":{{"type":"i32","value":{} }} }}
            ]}}"#,
            i64::from(i32::MAX) + 1
        );
        let err = EmatixDataFileExtension::from_json(&s).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("i32 range"), "got: {msg}");
    }

    #[test]
    fn json_bytes_summary_with_odd_hex_errors() {
        let s = r#"{"version":"v1","sidecar_relative_path":"x.idx","summaries":[
            {"name":"idx_x","min_key":{"type":"bytes","value_hex":"abc"}}
        ]}"#;
        let err = EmatixDataFileExtension::from_json(s).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("odd"), "got: {msg}");
    }

    #[test]
    fn json_bytes_summary_with_invalid_hex_errors() {
        let s = r#"{"version":"v1","sidecar_relative_path":"x.idx","summaries":[
            {"name":"idx_x","min_key":{"type":"bytes","value_hex":"zz"}}
        ]}"#;
        let err = EmatixDataFileExtension::from_json(s).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid hex"), "got: {msg}");
    }

    #[test]
    fn json_escaped_strings_round_trip() {
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "weird path/with \"quotes\"\\backslash\nnewline".into(),
            summaries: vec![IndexSummary::new("idx with \"quote\"")],
        };
        let json = ext.to_json();
        let parsed = EmatixDataFileExtension::from_json(&json).unwrap();
        assert_eq!(parsed, ext);
    }

    #[test]
    fn summary_lookup_by_name() {
        let ext = full_extension();
        assert_eq!(
            ext.summary("idx_partkey_bloom").unwrap().name,
            "idx_partkey_bloom"
        );
        assert!(ext.summary("nonexistent").is_none());
    }

    #[test]
    fn negative_i64_value_round_trips() {
        let ext = EmatixDataFileExtension {
            sidecar_relative_path: "x.idx".into(),
            summaries: vec![IndexSummary::new("idx_x")
                .with_range(SummaryKey::I64(i64::MIN), SummaryKey::I64(-1))],
        };
        let json = ext.to_json();
        let parsed = EmatixDataFileExtension::from_json(&json).unwrap();
        assert_eq!(parsed, ext);
    }
}
