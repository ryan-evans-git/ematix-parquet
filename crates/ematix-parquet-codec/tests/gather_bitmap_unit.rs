//! Oracle for `gather_dict_at_bitmap_into` — bitmap-driven sparse
//! gather from a dict-encoded RLE/bit-packed page body.
//!
//! Given:
//!   - body: RLE_DICTIONARY data page body (bw prefix + RLE/bit-pack stream)
//!   - num_values: total rows in this page
//!   - bitmap: row-mask covering the whole column (bit `bitmap_offset+row` = include row)
//!   - dict: PLAIN-decoded dictionary values
//!
//! Produces:
//!   - `out` appended with `dict[idx[row]]` for each row where the bitmap bit is set.
//!
//! Length of appended output == popcount(bitmap bits for this page).

use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, gather_dict_at_bitmap_into};

fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let bw_mask: u64 = if bit_width == 0 {
        0
    } else {
        (1u64 << bit_width) - 1
    };
    for (i, &v) in values.iter().enumerate() {
        let v = (v as u64) & bw_mask;
        let start_bit = i * bit_width as usize;
        let mut byte_idx = start_bit / 8;
        let mut bit_in_byte = (start_bit % 8) as u32;
        let mut remaining = v;
        let mut remaining_bits = bit_width as u32;
        while remaining_bits > 0 {
            let space = 8 - bit_in_byte;
            let take = space.min(remaining_bits);
            let chunk = (remaining & ((1u64 << take) - 1)) as u8;
            out[byte_idx] |= chunk << bit_in_byte;
            remaining >>= take;
            remaining_bits -= take;
            byte_idx += 1;
            bit_in_byte = 0;
        }
    }
    out
}

fn write_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn make_body_bitpacked(values: &[u32], bit_width: u8) -> Vec<u8> {
    let n = values.len();
    let num_groups = n.div_ceil(8);
    let mut padded = values.to_vec();
    padded.resize(num_groups * 8, 0);
    let packed = pack(&padded, bit_width);

    let mut out = vec![bit_width];
    write_uvarint(&mut out, ((num_groups as u64) << 1) | 1);
    out.extend(packed);
    out
}

fn make_body_rle(idx: u32, count: usize, bit_width: u8) -> Vec<u8> {
    let mut out = vec![bit_width];
    write_uvarint(&mut out, (count as u64) << 1);
    let value_bytes = (bit_width as usize).div_ceil(8);
    for b in 0..value_bytes {
        out.push(((idx >> (b * 8)) & 0xFF) as u8);
    }
    out
}

fn reference_gather<T: Copy>(
    body: &[u8],
    n: usize,
    bitmap: &[u8],
    bitmap_offset: usize,
    dict: &[T],
) -> Vec<T> {
    let indices = decode_rle_dictionary_indices(body, n).unwrap();
    let mut out = Vec::new();
    for (row, idx) in indices.into_iter().enumerate() {
        let bit_pos = bitmap_offset + row;
        let bit = (bitmap[bit_pos / 8] >> (bit_pos % 8)) & 1;
        if bit == 1 {
            out.push(dict[idx as usize]);
        }
    }
    out
}

#[test]
fn bw17_bitpacked_matches_reference() {
    let n: usize = 10_000;
    let mut seed: u32 = 0xC0DECAFE;
    let dict_size = 100_000u32;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed % dict_size
        })
        .collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<i64> = (0..dict_size as i64).map(|i| i * 1000).collect();

    // ~1% selectivity bitmap.
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    let mut bm_seed: u32 = 0xFEED;
    for row in 0..n {
        bm_seed = bm_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        if bm_seed % 100 == 0 {
            bitmap[row / 8] |= 1u8 << (row % 8);
        }
    }

    let mut out: Vec<i64> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 0, &dict, &mut out).unwrap();
    let expected = reference_gather(&body, n, &bitmap, 0, &dict);
    assert_eq!(out, expected);
    let expected_count: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
    assert_eq!(out.len(), expected_count);
}

#[test]
fn bw17_rle_run_with_partial_bitmap() {
    // 64 rows with idx=42, half-set bitmap.
    let body = make_body_rle(42, 64, 17);
    let dict: Vec<i32> = (0..100_000).map(|i| i * 10).collect();
    let bitmap = vec![0xAAu8; 8]; // alternating bits

    let mut out: Vec<i32> = Vec::new();
    gather_dict_at_bitmap_into(&body, 64, &bitmap, 0, &dict, &mut out).unwrap();
    let expected = reference_gather(&body, 64, &bitmap, 0, &dict);
    assert_eq!(out, expected);
    // Each 0xAA byte = 4 bits set; 8 bytes × 4 = 32 matching rows; all idx=42 → 420.
    assert_eq!(out, vec![420i32; 32]);
}

#[test]
fn bw17_with_bitmap_offset() {
    // The bitmap covers a whole row group, but this call processes
    // only one page that starts at bitmap_offset = 2048.
    let n: usize = 1024;
    let values: Vec<u32> = (0..n as u32).collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<u64> = (0..100_000).map(|i| (i as u64) * 7).collect();

    let mut bitmap = vec![0u8; 4096];
    // Set bit 2048 + 5, 2048 + 100, 2048 + 1023.
    for &row in &[5usize, 100, 1023] {
        let bit_pos = 2048 + row;
        bitmap[bit_pos / 8] |= 1u8 << (bit_pos % 8);
    }

    let mut out: Vec<u64> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 2048, &dict, &mut out).unwrap();
    assert_eq!(out, vec![5u64 * 7, 100 * 7, 1023 * 7]);
}

#[test]
fn bw12_smaller_width_still_works() {
    // Same kernel must handle smaller widths via the const-generic
    // dispatch — not all Q14 columns are bw=17.
    let n: usize = 5000;
    let mut seed: u32 = 0xDEAD;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let body = make_body_bitpacked(&values, 12);
    let dict: Vec<i32> = (0..4096).map(|i| i * 3).collect();

    let mut bitmap = vec![0u8; n.div_ceil(8)];
    for row in (0..n).step_by(7) {
        bitmap[row / 8] |= 1u8 << (row % 8);
    }

    let mut out: Vec<i32> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 0, &dict, &mut out).unwrap();
    let expected = reference_gather(&body, n, &bitmap, 0, &dict);
    assert_eq!(out, expected);
}

#[test]
fn empty_bitmap_yields_nothing() {
    let n: usize = 1000;
    let values: Vec<u32> = (0..n as u32).collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<i32> = (0..100_000).collect();
    let bitmap = vec![0u8; n.div_ceil(8)];

    let mut out: Vec<i32> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 0, &dict, &mut out).unwrap();
    assert_eq!(out, Vec::<i32>::new());
}

#[test]
fn full_bitmap_yields_everything() {
    let n: usize = 100;
    let values: Vec<u32> = (0..n as u32).collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<i32> = (0..100_000).map(|i| i * 11).collect();
    let bitmap = vec![0xFFu8; n.div_ceil(8)];

    let mut out: Vec<i32> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 0, &dict, &mut out).unwrap();
    let expected: Vec<i32> = (0..n as i32).map(|i| i * 11).collect();
    assert_eq!(out, expected);
}

#[test]
fn mixed_runs_with_sparse_bitmap() {
    // Page starts with an RLE run, then a bit-packed run. Bitmap
    // is ~5% set across the page.
    let mut body = vec![17u8];
    // RLE run: 256 of idx=99.
    write_uvarint(&mut body, 256u64 << 1);
    body.push(99u8);
    body.push(0);
    body.push(0);
    // Bit-packed run: 256 random values.
    let mut bp_values: Vec<u32> = Vec::with_capacity(256);
    let mut s: u32 = 0xBEEF;
    for _ in 0..256 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        bp_values.push(s % 100_000);
    }
    write_uvarint(&mut body, ((256u64 / 8) << 1) | 1);
    body.extend(pack(&bp_values, 17));

    let n: usize = 256 + 256;
    let dict: Vec<i64> = (0..100_000).map(|i| (i as i64) * 7 + 3).collect();
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    for row in (0..n).step_by(20) {
        bitmap[row / 8] |= 1u8 << (row % 8);
    }

    let mut out: Vec<i64> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, 0, &dict, &mut out).unwrap();
    let expected = reference_gather(&body, n, &bitmap, 0, &dict);
    assert_eq!(out, expected);
}

/// Regression: when `bitmap_offset` is not a multiple of 8, the
/// bit-packed branch's "8-row block, one bitmap-byte lookup" fast path
/// would silently read the wrong bitmap byte in release builds (the
/// in-source `debug_assert_eq!(bit_pos_base % 8, 0)` only fired in
/// debug). This happens in practice for any data page after page 0
/// whose `num_values` is not a multiple of 8 — the cumulative
/// `bitmap_offset` carried across pages goes out of alignment, and
/// the byte-wise fast path's `bitmap[bit_pos_base/8]` reads pre-shifted
/// bits. Manifested as "decoded N values, expected M" on TPC-H Q1
/// l_returnflag (Utf8, dict-encoded, byte-array column).
#[test]
fn bw_bitpacked_with_misaligned_bitmap_offset() {
    // Page of 1024 values, but call into the bitmap at a non-multiple-
    // of-8 offset (5) — simulates a real "second page after a first
    // page whose num_values % 8 == 5".
    let n: usize = 1024;
    let values: Vec<u32> = (0..n as u32).collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<u64> = (0..100_000).map(|i| (i as u64) * 7).collect();

    // Bitmap large enough to cover bitmap_offset (5) + n (1024) = 1029
    // bits. Set bits at absolute positions 5, 10, 100, 1028.
    let bitmap_offset = 5usize;
    let total_bits = bitmap_offset + n;
    let mut bitmap = vec![0u8; total_bits.div_ceil(8) + 1];
    let absolute_set: &[usize] = &[5, 10, 100, 1028];
    for &abs_pos in absolute_set {
        bitmap[abs_pos / 8] |= 1u8 << (abs_pos % 8);
    }

    let mut out: Vec<u64> = Vec::new();
    gather_dict_at_bitmap_into(&body, n, &bitmap, bitmap_offset, &dict, &mut out).unwrap();
    let expected = reference_gather(&body, n, &bitmap, bitmap_offset, &dict);
    assert_eq!(
        out,
        expected,
        "gather produced {} values, reference produced {} (bitmap_offset={bitmap_offset})",
        out.len(),
        expected.len()
    );
    let expected_count: usize = absolute_set.len();
    assert_eq!(out.len(), expected_count);
}

/// Stress: every misaligned starting offset 1..=7, against random
/// values and a random sparse bitmap — must match the reference.
#[test]
fn bw_bitpacked_all_misaligned_starting_offsets() {
    let n: usize = 2_000;
    let mut seed: u32 = 0xBEEF_0042;
    let dict_size = 50_000u32;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed % dict_size
        })
        .collect();
    let body = make_body_bitpacked(&values, 17);
    let dict: Vec<i64> = (0..dict_size as i64).map(|i| i * 11).collect();

    for offset in 1usize..=7 {
        let total_bits = offset + n + 7;
        let mut bitmap = vec![0u8; total_bits.div_ceil(8) + 1];
        let mut bm_seed: u32 = 0x1234 ^ (offset as u32);
        for row in 0..n {
            bm_seed = bm_seed.wrapping_mul(1664525).wrapping_add(1013904223);
            if bm_seed % 50 == 0 {
                let abs_pos = offset + row;
                bitmap[abs_pos / 8] |= 1u8 << (abs_pos % 8);
            }
        }
        let mut out: Vec<i64> = Vec::new();
        gather_dict_at_bitmap_into(&body, n, &bitmap, offset, &dict, &mut out).unwrap();
        let expected = reference_gather(&body, n, &bitmap, offset, &dict);
        assert_eq!(out, expected, "mismatch at bitmap_offset={offset}");
    }
}
