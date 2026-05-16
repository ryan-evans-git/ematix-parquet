//! Oracle: `read_column_byte_array_dict_preserved` returns the
//! parquet dictionary (as flat bytes + offsets) plus a raw `Vec<u32>`
//! of per-row indices, *without* materialising per-row values.
//!
//! Σ.E3b substrate contract: the returned `(dict_bytes, dict_offsets,
//! indices)` triple is byte-identical to the values produced by
//! `read_column_byte_array_offsets` when the caller materialises the
//! row stream by index lookup. The point of this entry point is to
//! preserve the dict structure end-to-end so Arrow consumers can
//! build a `DictionaryArray<UInt32, Utf8>` instead of paying the
//! per-row gather + hash on the next operator boundary.
//!
//! Failure modes asserted:
//!   - column has no DictionaryPage  → error (caller falls back).
//!   - column has a PLAIN data page  → error (mixed-encoding column
//!     cannot be expressed as one dict + indices).

use ematix_parquet_codec::read::{
    read_column_byte_array_dict_preserved, read_column_byte_array_offsets,
};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path,
};
use ematix_parquet_io::ParquetFile;

fn materialize(dict_bytes: &[u8], dict_offsets: &[u32], indices: &[u32]) -> Vec<Vec<u8>> {
    indices
        .iter()
        .map(|&i| {
            let s = dict_offsets[i as usize] as usize;
            let e = dict_offsets[i as usize + 1] as usize;
            dict_bytes[s..e].to_vec()
        })
        .collect()
}

#[test]
fn dict_preserved_matches_offsets_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dict.parquet");

    let palette: [&[u8]; 5] = [b"alpha", b"bravo", b"charlie", b"delta", b"echo"];
    let values: Vec<&[u8]> = (0..600).map(|i| palette[i % 5]).collect();
    write_byte_array_column_dict_to_path(
        &path,
        "v",
        &values,
        ematix_parquet_format::types::CompressionCodec::Uncompressed,
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let col = read_column_byte_array_dict_preserved(&file, 0, 0).unwrap();

    // Dict carries exactly the distinct values from the palette
    // (write_byte_array_column_dict_to_path emits them in
    // first-appearance order).
    assert_eq!(col.dict_offsets.len(), palette.len() + 1);
    assert_eq!(col.indices.len(), values.len());

    // Materialised row stream is byte-identical to the offsets reader.
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    let want: Vec<Vec<u8>> = offsets
        .windows(2)
        .map(|w| bytes[w[0] as usize..w[1] as usize].to_vec())
        .collect();
    let got = materialize(&col.dict_bytes, &col.dict_offsets, &col.indices);
    assert_eq!(got, want);

    // Every index lands inside the dict.
    let dict_len = col.dict_offsets.len() - 1;
    assert!(col.indices.iter().all(|&i| (i as usize) < dict_len));
}

#[test]
fn dict_preserved_with_snappy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dict-snappy.parquet");

    let palette: [&[u8]; 3] = [b"R", b"A", b"N"];
    let values: Vec<&[u8]> = (0..1_500).map(|i| palette[i % 3]).collect();
    write_byte_array_column_dict_to_path(
        &path,
        "flag",
        &values,
        ematix_parquet_format::types::CompressionCodec::Snappy,
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let col = read_column_byte_array_dict_preserved(&file, 0, 0).unwrap();

    assert_eq!(col.indices.len(), values.len());
    let got = materialize(&col.dict_bytes, &col.dict_offsets, &col.indices);
    let want: Vec<Vec<u8>> = values.iter().map(|s| s.to_vec()).collect();
    assert_eq!(got, want);
}

#[test]
fn dict_preserved_errors_on_plain_only_column() {
    // A pure-PLAIN column has no dictionary page → caller cannot
    // build a DictionaryArray, so the entry point must surface that
    // explicitly rather than silently fabricate a dict.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.parquet");

    let values: Vec<&[u8]> = vec![b"alpha", b"bravo", b"charlie"];
    write_byte_array_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let err = read_column_byte_array_dict_preserved(&file, 0, 0).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("PLAIN") || msg.contains("dictionary"),
        "expected error to mention plain/dictionary, got: {msg}"
    );
}
