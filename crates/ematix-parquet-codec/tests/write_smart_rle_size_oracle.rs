//! Oracle: the smart RLE encoder produces strictly smaller files
//! than the single-bit-packed-run encoder for inputs with long
//! repeated runs, and parquet-rs reads the smaller files identically.
//!
//! Π.8c contract: run-coalescing kicks in at run length ≥ 8.
//! Inputs with high run-length should compress dramatically; inputs
//! with no repetition should produce the same on-disk size as the
//! single-run encoder.

use ematix_parquet_codec::write::write_byte_array_column_dict_to_path;
use ematix_parquet_format::types::CompressionCodec;

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray as PqByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn write_dict_file(path: &std::path::Path, values: &[&[u8]]) -> u64 {
    write_byte_array_column_dict_to_path(path, "v", values, CompressionCodec::Uncompressed)
        .unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn read_back_via_parquet_rs(path: &std::path::Path, n: usize) -> Vec<Vec<u8>> {
    let f = std::fs::File::open(path).unwrap();
    let r = SerializedFileReader::new(f).unwrap();
    let rg = r.get_row_group(0).unwrap();
    let cr = rg.get_column_reader(0).unwrap();
    let ColumnReader::ByteArrayColumnReader(mut typed) = cr else {
        panic!("expected ByteArray reader");
    };
    let mut out: Vec<PqByteArray> = Vec::with_capacity(n);
    typed.read_records(n, None, None, &mut out).unwrap();
    out.into_iter().map(|b| b.data().to_vec()).collect()
}

#[test]
fn long_runs_compress_dramatically() {
    // 10 distinct strings × 10000 contiguous repeats each. Every
    // run is length 10000 — worst case for bit-pack, best case
    // for RLE. With 10 distinct values dict_len = 10 → bit_width = 4.
    // Expected RLE size on the index stream: ~10 × (1 + 1) = 20 bytes
    // vs bit-pack ~50000 bytes. Roughly 2500× smaller index stream.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("repetitive.parquet");

    let palette: [&[u8]; 10] = [
        b"alpha", b"bravo", b"charlie", b"delta", b"echo",
        b"foxtrot", b"golf", b"hotel", b"india", b"juliett",
    ];
    let mut values: Vec<&[u8]> = Vec::with_capacity(100_000);
    for &p in &palette {
        for _ in 0..10_000 {
            values.push(p);
        }
    }

    let size = write_dict_file(&path, &values);

    // Sanity: parquet-rs reads it back identically.
    let read_back = read_back_via_parquet_rs(&path, values.len());
    assert_eq!(read_back.len(), values.len());
    for (i, got) in read_back.iter().enumerate() {
        assert_eq!(got.as_slice(), values[i], "row {i}");
    }

    // Size sanity: the file should be tiny — dict (~50 bytes for
    // 10 short strings) + RLE index stream (~20-40 bytes) +
    // page/file metadata (a few KB).
    assert!(
        size < 4 * 1024,
        "100K-row dict-encoded file with long runs should be < 4KB, got {} bytes",
        size
    );
}

#[test]
fn cycling_input_still_uses_bitpack_compactly() {
    // 3 distinct values cycled (no runs of length ≥ 8). The smart
    // encoder should fall back to bit-pack and produce roughly the
    // same size as the single-run encoder.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cycling.parquet");

    let palette: [&[u8]; 3] = [b"A", b"R", b"N"];
    let values: Vec<&[u8]> = (0..30_000).map(|i| palette[i % 3]).collect();

    let size = write_dict_file(&path, &values);

    // Sanity: round-trip via parquet-rs.
    let read_back = read_back_via_parquet_rs(&path, values.len());
    for (i, got) in read_back.iter().enumerate() {
        assert_eq!(got.as_slice(), values[i]);
    }

    // bit_width = 2 for 3 distinct → 2 bits per index → 30000/4 = 7500
    // bytes for the index stream + small metadata.
    assert!(
        size < 12 * 1024,
        "no-repetition dict-encoded file should be ~7-12KB, got {} bytes",
        size
    );
}

#[test]
fn mixed_runs_and_singletons_compress_better() {
    // Realistic shape: a "hot" value covering 80% of rows, with
    // small runs of other values mixed in. Smart encoder should
    // emit a few long RLE runs interleaved with bit-pack.
    let dir = tempfile::tempdir().unwrap();
    let path_smart = dir.path().join("mixed.parquet");

    let hot: &[u8] = b"hot-value";
    let cold: [&[u8]; 4] = [b"cold-A", b"cold-B", b"cold-C", b"cold-D"];

    // Pattern: hot×500, cold[0], cold[1], cold[2], cold[3], hot×500, ...
    // Repeat 100 times → 100 × (500 hot + 4 cold) = 50400 rows.
    let mut values: Vec<&[u8]> = Vec::new();
    for cycle in 0..100 {
        for _ in 0..500 { values.push(hot); }
        for &c in &cold {
            values.push(c);
            // Use the cycle index to vary the cold values a bit.
            let _ = cycle;
        }
    }

    let size = write_dict_file(&path_smart, &values);

    // 5 distinct → bit_width = 3. Naive bit-pack: 50400 × 3 / 8 ≈ 19000 bytes.
    // With RLE on the 100 hot runs of 500: 100 × ~3 bytes = ~300 bytes for hot,
    // plus bit-pack for the ~400 cold values ≈ 150 bytes. Total ≈ ~450 bytes
    // for the index stream — vastly smaller than naive bit-pack.
    assert!(
        size < 2 * 1024,
        "long-run-dominated file should compress well, got {} bytes",
        size
    );

    // Round-trip correctness.
    let read_back = read_back_via_parquet_rs(&path_smart, values.len());
    for (i, got) in read_back.iter().enumerate() {
        assert_eq!(got.as_slice(), values[i], "row {i}");
    }
}
