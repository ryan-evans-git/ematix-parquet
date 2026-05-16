//! Π.10b oracle: byte_array masked-decode parity vs
//! full-decode-then-filter. Mirror of `read_masked_oracle.rs` for
//! the BYTE_ARRAY type, both shapes:
//!
//! - `read_column_byte_array_masked_into(&mut Vec<Vec<u8>>)`
//! - `read_column_byte_array_offsets_masked_into(&mut Vec<u8>, &mut Vec<u32>)`
//!
//! BYTE_ARRAY can't reuse the scalar `T: Copy` chunk walker; it has
//! its own walker in read.rs. The dict path relies on the Π.9 fused
//! gather being relaxed to `T: Clone` (Vec<u8>: !Copy, : Clone).

use ematix_parquet_codec::read::{
    build_packed_mask, read_column_byte_array, read_column_byte_array_masked_into,
    read_column_byte_array_offsets, read_column_byte_array_offsets_masked_into,
};
use ematix_parquet_codec::write::{
    write_byte_array_column_dict_to_path, write_byte_array_column_to_path,
};
use ematix_parquet_format::types::CompressionCodec;
use ematix_parquet_io::ParquetFile;

fn reference(path: &std::path::Path, mask: &[u8]) -> Vec<Vec<u8>> {
    let f = ParquetFile::open(path).unwrap();
    let full = read_column_byte_array(&f, 0, 0).unwrap();
    full.iter()
        .enumerate()
        .filter(|(row, _)| (mask[row / 8] >> (row % 8)) & 1 == 1)
        .map(|(_, v)| v.clone())
        .collect()
}

fn stride_mask(n: usize, stride: usize) -> Vec<u8> {
    build_packed_mask(n, |i| i % stride == 0)
}

fn all_set_mask(n: usize) -> Vec<u8> {
    let mut m = vec![0xFFu8; n.div_ceil(8)];
    let tail = n % 8;
    if tail != 0 {
        let last = m.len() - 1;
        m[last] &= (1u8 << tail) - 1;
    }
    m
}

fn empty_mask(n: usize) -> Vec<u8> {
    vec![0u8; n.div_ceil(8)]
}

// ============================================================
// Vec<Vec<u8>> shape — PLAIN encoding
// ============================================================

#[test]
fn byte_array_plain_selectivity_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_plain.parquet");
    // Variable-length: row i = b"v" * (i % 5 + 1).
    let owned: Vec<Vec<u8>> = (0..2_000)
        .map(|i| vec![b'v'; i % 5 + 1])
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let n = owned.len();
    let masks = vec![
        ("0%", empty_mask(n)),
        ("0.1%", stride_mask(n, 1000)),
        ("1%", stride_mask(n, 100)),
        ("10%", stride_mask(n, 10)),
        ("50%", stride_mask(n, 2)),
        ("100%", all_set_mask(n)),
    ];

    let file = ParquetFile::open(&path).unwrap();
    for (label, mask) in masks {
        let want = reference(&path, &mask);
        let mut got: Vec<Vec<u8>> = Vec::new();
        read_column_byte_array_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
        assert_eq!(got, want, "byte_array PLAIN @ {label}: mismatch");
    }
}

// ============================================================
// Vec<Vec<u8>> shape — DICT encoding
// ============================================================

#[test]
fn byte_array_dict_selectivity_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_dict.parquet");
    let palette: [&[u8]; 4] = [b"alpha", b"bravo", b"charlie", b"delta"];
    let owned: Vec<Vec<u8>> = (0..2_000)
        .map(|i| palette[i % 4].to_vec())
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_dict_to_path(&path, "v", &refs, CompressionCodec::Snappy).unwrap();

    let n = owned.len();
    let masks = vec![
        ("0%", empty_mask(n)),
        ("0.1%", stride_mask(n, 1000)),
        ("1%", stride_mask(n, 100)),
        ("10%", stride_mask(n, 10)),
        ("50%", stride_mask(n, 2)),
        ("100%", all_set_mask(n)),
    ];

    let file = ParquetFile::open(&path).unwrap();
    for (label, mask) in masks {
        let want = reference(&path, &mask);
        let mut got: Vec<Vec<u8>> = Vec::new();
        read_column_byte_array_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
        assert_eq!(got, want, "byte_array dict @ {label}: mismatch");
    }
}

// ============================================================
// Offsets shape — PLAIN + DICT
// ============================================================

fn reference_offsets(
    path: &std::path::Path,
    mask: &[u8],
) -> (Vec<u8>, Vec<u32>) {
    let f = ParquetFile::open(path).unwrap();
    let (full_bytes, full_offsets) = read_column_byte_array_offsets(&f, 0, 0).unwrap();
    let mut out_bytes = Vec::new();
    let mut out_offsets = vec![0u32];
    let mut running: u32 = 0;
    let n = full_offsets.len() - 1;
    for row in 0..n {
        let bit = (mask[row / 8] >> (row % 8)) & 1;
        if bit == 1 {
            let s = full_offsets[row] as usize;
            let e = full_offsets[row + 1] as usize;
            out_bytes.extend_from_slice(&full_bytes[s..e]);
            running += (e - s) as u32;
            out_offsets.push(running);
        }
    }
    (out_bytes, out_offsets)
}

#[test]
fn byte_array_offsets_plain_matches_reference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_off_plain.parquet");
    let owned: Vec<Vec<u8>> = (0..1_000)
        .map(|i| vec![(i % 26) as u8 + b'a'; (i % 7) + 1])
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let mask = stride_mask(owned.len(), 11);
    let (want_bytes, want_offsets) = reference_offsets(&path, &mask);

    let file = ParquetFile::open(&path).unwrap();
    let mut got_bytes = Vec::new();
    let mut got_offsets = Vec::new();
    read_column_byte_array_offsets_masked_into(
        &file, 0, 0, &mask, &mut got_bytes, &mut got_offsets,
    ).unwrap();

    assert_eq!(got_bytes, want_bytes);
    assert_eq!(got_offsets, want_offsets);
}

#[test]
fn byte_array_offsets_dict_matches_reference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_off_dict.parquet");
    let palette: [&[u8]; 3] = [b"A", b"BB", b"CCC"];
    let owned: Vec<Vec<u8>> = (0..1_500)
        .map(|i| palette[i % 3].to_vec())
        .collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_dict_to_path(
        &path, "v", &refs, CompressionCodec::Uncompressed,
    ).unwrap();

    let mask = stride_mask(owned.len(), 7);
    let (want_bytes, want_offsets) = reference_offsets(&path, &mask);

    let file = ParquetFile::open(&path).unwrap();
    let mut got_bytes = Vec::new();
    let mut got_offsets = Vec::new();
    read_column_byte_array_offsets_masked_into(
        &file, 0, 0, &mask, &mut got_bytes, &mut got_offsets,
    ).unwrap();

    assert_eq!(got_bytes, want_bytes);
    assert_eq!(got_offsets, want_offsets);
}

// ============================================================
// Append semantics for both shapes
// ============================================================

#[test]
fn byte_array_masked_appends_not_clears() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_append.parquet");
    let owned: Vec<Vec<u8>> = (0..100).map(|i| vec![b'x'; (i % 3) + 1]).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let mask = stride_mask(100, 10);
    let file = ParquetFile::open(&path).unwrap();

    let mut got: Vec<Vec<u8>> = vec![b"PREFILL".to_vec()];
    read_column_byte_array_masked_into(&file, 0, 0, &mask, &mut got).unwrap();
    assert_eq!(got[0], b"PREFILL".to_vec(), "prefill must survive");
    assert_eq!(got.len(), 1 + 10, "10 matches appended onto prefill");
}

#[test]
fn byte_array_offsets_masked_concatenates_chunks() {
    // Multi-call concatenation: same out_bytes / out_offsets across
    // two reads. The second call must continue offsets from the
    // first's last value, no reset.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_concat.parquet");
    let owned: Vec<Vec<u8>> = (0..200).map(|i| vec![b'y'; (i % 4) + 1]).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let mask = stride_mask(200, 20); // 10 matches
    let file = ParquetFile::open(&path).unwrap();

    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    // Call twice with the same buffers.
    read_column_byte_array_offsets_masked_into(
        &file, 0, 0, &mask, &mut bytes, &mut offsets,
    ).unwrap();
    let after_first_n = offsets.len();
    read_column_byte_array_offsets_masked_into(
        &file, 0, 0, &mask, &mut bytes, &mut offsets,
    ).unwrap();

    assert_eq!(offsets.len(), 2 * after_first_n - 1, "second call appends N-1 new offsets onto existing N (no re-push of leading 0)");
    // Offsets must be monotonically non-decreasing.
    for w in offsets.windows(2) {
        assert!(w[1] >= w[0], "offsets must be monotonically non-decreasing");
    }
    let final_off = *offsets.last().unwrap() as usize;
    assert_eq!(bytes.len(), final_off, "bytes.len() must equal final offset");
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn byte_array_mask_too_small_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_small.parquet");
    let owned: Vec<Vec<u8>> = (0..1_000).map(|_| b"x".to_vec()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mask = vec![0u8; 50]; // need 125
    let mut out: Vec<Vec<u8>> = Vec::new();
    let r = read_column_byte_array_masked_into(&file, 0, 0, &mask, &mut out);
    assert!(r.is_err());

    let mut b = Vec::new();
    let mut o = Vec::new();
    let r = read_column_byte_array_offsets_masked_into(&file, 0, 0, &mask, &mut b, &mut o);
    assert!(r.is_err());
}

#[test]
fn byte_array_offsets_empty_mask_initial_zero_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ba_empty.parquet");
    let owned: Vec<Vec<u8>> = (0..500).map(|_| b"v".to_vec()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    write_byte_array_column_to_path(&path, "v", &refs).unwrap();

    let file = ParquetFile::open(&path).unwrap();
    let mask = empty_mask(500);
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    read_column_byte_array_offsets_masked_into(
        &file, 0, 0, &mask, &mut bytes, &mut offsets,
    ).unwrap();
    assert!(bytes.is_empty());
    assert_eq!(offsets, vec![0u32], "empty mask: only the initial 0 offset");
}
