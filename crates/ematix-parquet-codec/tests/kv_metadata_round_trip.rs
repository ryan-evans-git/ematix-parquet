//! Round-trip test for footer `KeyValueMetadata` on the write path.
//!
//! Until this commit, `FileMetaData.key_value_metadata` panicked on
//! write. The sidecar-index builder (Π.17b) embeds the
//! `IndexManifest` JSON under the key `ematix_index_manifest_v1`, so
//! KV-metadata write support is a load-bearing prerequisite.
//!
//! These tests are independent of the index work: they only verify
//! that `WriteOptions.kv_metadata` → footer KV → re-parsed
//! `FileMetaData.key_value_metadata` is a faithful round trip.

use ematix_parquet_codec::write::{write_table_with_options_to_path, ColumnData, WriteOptions};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

#[test]
fn no_kv_metadata_means_none_on_read() {
    // Pre-existing default behaviour must not regress: writing without
    // `kv_metadata` produces a footer whose `key_value_metadata` is
    // `None` (the read-side legitimately distinguishes "absent" from
    // "present but empty" — Parquet's optional field).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_kv.parquet");
    let values: &[i64] = &[1, 2, 3];
    let cols: &[(&str, ColumnData<'_>)] = &[("v", ColumnData::I64(values))];
    write_table_with_options_to_path(&path, cols, &WriteOptions::default()).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    assert!(
        md.key_value_metadata.is_none(),
        "default WriteOptions must produce no KV metadata"
    );
}

#[test]
fn single_kv_entry_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one_kv.parquet");
    let values: &[i64] = &[10, 20, 30];
    let cols: &[(&str, ColumnData<'_>)] = &[("v", ColumnData::I64(values))];
    let kvs = [("ematix_index_manifest_v1", "{\"version\":\"v1\"}")];
    let opts = WriteOptions {
        kv_metadata: Some(&kvs),
        ..WriteOptions::default()
    };
    write_table_with_options_to_path(&path, cols, &opts).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let kvs = md.key_value_metadata.as_ref().expect("KV metadata present");
    assert_eq!(kvs.len(), 1);
    assert_eq!(kvs[0].key, b"ematix_index_manifest_v1");
    assert_eq!(
        kvs[0].value.expect("value present"),
        b"{\"version\":\"v1\"}"
    );
}

#[test]
fn multiple_kv_entries_round_trip_in_order() {
    // Sidecar usage will pin exactly one entry today, but downstream
    // tooling (statistics watermarks, schema URIs) can use many. The
    // wire format is a Thrift list, so order is preserved.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_kv.parquet");
    let values: &[i64] = &[1, 2, 3, 4, 5];
    let cols: &[(&str, ColumnData<'_>)] = &[("v", ColumnData::I64(values))];
    let kvs = [
        ("first", "alpha"),
        ("second", "beta"),
        ("third", "gamma with spaces and \"quotes\""),
    ];
    let opts = WriteOptions {
        kv_metadata: Some(&kvs),
        ..WriteOptions::default()
    };
    write_table_with_options_to_path(&path, cols, &opts).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let read_kvs = md.key_value_metadata.as_ref().expect("KV metadata present");
    assert_eq!(read_kvs.len(), 3);
    assert_eq!(read_kvs[0].key, b"first");
    assert_eq!(read_kvs[0].value.unwrap(), b"alpha");
    assert_eq!(read_kvs[1].key, b"second");
    assert_eq!(read_kvs[1].value.unwrap(), b"beta");
    assert_eq!(read_kvs[2].key, b"third");
    assert_eq!(
        read_kvs[2].value.unwrap(),
        b"gamma with spaces and \"quotes\""
    );
}

#[test]
fn kv_metadata_survives_with_compression_and_dict() {
    // Belt-and-braces: KV metadata lives in the footer, not in any
    // row group, so it must round-trip independent of every other
    // write knob. Pair with the codec + dict + bloom options to
    // verify no interaction.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv_with_opts.parquet");
    let values: Vec<i64> = (0..1_000i64).collect();
    let cols: &[(&str, ColumnData<'_>)] = &[("v", ColumnData::I64(&values))];
    let codecs = [CompressionCodec::Snappy];
    let dicts = [true];
    let kvs = [("ematix_index_manifest_v1", "{\"version\":\"v1\"}")];
    let opts = WriteOptions {
        default_codec: CompressionCodec::Snappy,
        codec_per_column: Some(&codecs),
        dict_per_column: Some(&dicts),
        kv_metadata: Some(&kvs),
        ..WriteOptions::default()
    };
    write_table_with_options_to_path(&path, cols, &opts).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let read_kvs = md.key_value_metadata.as_ref().expect("KV metadata present");
    assert_eq!(read_kvs.len(), 1);
    assert_eq!(read_kvs[0].key, b"ematix_index_manifest_v1");
    assert_eq!(read_kvs[0].value.unwrap(), b"{\"version\":\"v1\"}");
}

#[test]
fn empty_value_kv_entry_round_trips() {
    // Edge case: producers that want the *presence* of a flag without
    // a payload should be able to write `key = ""`. (Distinct from
    // value = None, which our high-level WriteOptions doesn't expose
    // today — every entry on the new API is (key, value) with a
    // non-optional value; if we need value=None later we'll add it.)
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_val.parquet");
    let values: &[i64] = &[42];
    let cols: &[(&str, ColumnData<'_>)] = &[("v", ColumnData::I64(values))];
    let kvs = [("flag.empty_value_ok", "")];
    let opts = WriteOptions {
        kv_metadata: Some(&kvs),
        ..WriteOptions::default()
    };
    write_table_with_options_to_path(&path, cols, &opts).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let md = file.metadata().unwrap();
    let read_kvs = md.key_value_metadata.as_ref().expect("KV metadata present");
    assert_eq!(read_kvs.len(), 1);
    assert_eq!(read_kvs[0].key, b"flag.empty_value_ok");
    assert_eq!(read_kvs[0].value.unwrap(), b"");
}
