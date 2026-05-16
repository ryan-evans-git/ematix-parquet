//! Oracle: `read_column_byte_array_offsets` returns Arrow-style
//! flat bytes + offsets that reconstruct exactly the same per-row
//! values as `read_column_byte_array`.
//!
//! Π.8a contract: the zero-copy gather entry point is byte-identical
//! to the existing Vec<Vec<u8>> entry point on every supported
//! encoding (PLAIN, PLAIN_DICTIONARY) and on every input shape.

use ematix_parquet_codec::read::{read_column_byte_array, read_column_byte_array_offsets};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path,
};
use ematix_parquet_io::ParquetFile;

/// Reconstruct per-row Vec<Vec<u8>> from (bytes, offsets).
fn reconstruct(bytes: &[u8], offsets: &[u32]) -> Vec<Vec<u8>> {
    assert!(
        !offsets.is_empty(),
        "offsets must include the trailing total"
    );
    let mut out = Vec::with_capacity(offsets.len() - 1);
    for w in offsets.windows(2) {
        let s = w[0] as usize;
        let e = w[1] as usize;
        out.push(bytes[s..e].to_vec());
    }
    out
}

#[test]
fn plain_byte_array_offsets_roundtrip() {
    // Write via the PLAIN entry point → read back via the offsets API.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.parquet");

    let values: Vec<&[u8]> = vec![b"alpha", b"", b"bravo", b"charlie-delta", b"x", b"yz"];
    write_byte_array_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert_eq!(offsets.len(), values.len() + 1);
    assert_eq!(*offsets.first().unwrap(), 0);

    let reconstructed = reconstruct(&bytes, &offsets);
    let want: Vec<Vec<u8>> = values.iter().map(|s| s.to_vec()).collect();
    assert_eq!(reconstructed, want);

    // And the offsets entry point agrees with the existing entry point.
    let via_vec = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(reconstructed, via_vec);
}

#[test]
fn dict_byte_array_offsets_roundtrip() {
    // Write via the dict entry point → read back via the offsets API.
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
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert_eq!(offsets.len(), values.len() + 1);

    let reconstructed = reconstruct(&bytes, &offsets);
    let want: Vec<Vec<u8>> = values.iter().map(|s| s.to_vec()).collect();
    assert_eq!(reconstructed, want);

    let via_vec = read_column_byte_array(&file, 0, 0).unwrap();
    assert_eq!(reconstructed, via_vec);
}

#[test]
fn dict_byte_array_offsets_single_byte_values() {
    // The l_returnflag shape: 3 distinct one-byte values × 1M rows.
    // This is the workload the offsets API is optimised for.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("returnflag.parquet");

    let palette: [&[u8]; 3] = [b"A", b"R", b"N"];
    let values: Vec<&[u8]> = (0..10_000).map(|i| palette[i % 3]).collect();
    write_byte_array_column_dict_to_path(
        &path,
        "v",
        &values,
        ematix_parquet_format::types::CompressionCodec::Snappy,
    )
    .unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert_eq!(offsets.len(), values.len() + 1);
    // 1 byte per row — total bytes equals row count.
    assert_eq!(*offsets.last().unwrap() as usize, values.len());
    assert_eq!(bytes.len(), values.len());

    let reconstructed = reconstruct(&bytes, &offsets);
    let want: Vec<Vec<u8>> = values.iter().map(|s| s.to_vec()).collect();
    assert_eq!(reconstructed, want);
}

#[test]
fn empty_column_offsets() {
    // Edge case: zero rows. Offsets should be [0], no bytes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.parquet");

    let values: Vec<&[u8]> = Vec::new();
    write_byte_array_column_to_path(&path, "v", &values).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert!(bytes.is_empty());
    assert_eq!(offsets, vec![0u32]);
}

#[test]
fn variable_length_values() {
    // Values of widely varying length — exercises the offset
    // accumulator across different per-row sizes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("variable.parquet");

    let values: Vec<Vec<u8>> = (0..100)
        .map(|i| {
            let len = (i * 17) % 200; // 0..199 byte values
            (0..len)
                .map(|j| ((i + j) & 0xFF) as u8)
                .collect::<Vec<u8>>()
        })
        .collect();
    let value_refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &value_refs).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert_eq!(offsets.len(), values.len() + 1);

    let reconstructed = reconstruct(&bytes, &offsets);
    assert_eq!(reconstructed, values);
}

// ---- against parquet-rs as the cross-check ----

#[test]
fn round_trip_via_parquet_rs_written_file() {
    // parquet-rs writes a dict-encoded byte_array file; we read it
    // via the offsets API and verify the values reconstruct.
    use parquet::basic::{Compression, Repetition, Type as PhysicalType};
    use parquet::column::writer::ColumnWriter;
    use parquet::data_type::ByteArray as PqByteArray;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as SchemaType;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pq_dict.parquet");

    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("v", PhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build(),
    );

    let palette: [&[u8]; 4] = [b"foo", b"bar", b"baz", b"qux"];
    let pq_values: Vec<PqByteArray> = (0..2_000)
        .map(|i| PqByteArray::from(palette[i % 4].to_vec()))
        .collect();

    {
        let f = std::fs::File::create(&path).unwrap();
        let mut w = SerializedFileWriter::new(f, schema, props).unwrap();
        let mut rg = w.next_row_group().unwrap();
        let mut col = rg.next_column().unwrap().unwrap();
        if let ColumnWriter::ByteArrayColumnWriter(ref mut typed) = col.untyped() {
            typed.write_batch(&pq_values, None, None).unwrap();
        }
        col.close().unwrap();
        rg.close().unwrap();
        w.close().unwrap();
    }

    let file = ParquetFile::open(&path).unwrap();
    let (bytes, offsets) = read_column_byte_array_offsets(&file, 0, 0).unwrap();
    assert_eq!(offsets.len(), pq_values.len() + 1);
    let reconstructed = reconstruct(&bytes, &offsets);
    for (i, got) in reconstructed.iter().enumerate() {
        assert_eq!(got.as_slice(), pq_values[i].data(), "row {i}");
    }
}
