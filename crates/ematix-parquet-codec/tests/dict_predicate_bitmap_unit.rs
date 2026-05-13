//! Oracle for `decode_rle_dictionary_predicate_bitmap_bw12`.
//!
//! Walks an RLE/bit-packed data-page body (bw=12) and writes a packed
//! bitmap where bit `i` of byte `k` represents row `8k+i`. The bit's
//! value is `dict_mask[idx[8k+i]]` — i.e. the predicate result for
//! that row, fused into the decode pass.
//!
//! We compare against a scalar reference: decode indices via the
//! existing scalar path, gather dict_mask, pack bitmap. The new
//! function must match byte-for-byte.

#![cfg(target_arch = "aarch64")]

use ematix_parquet_codec::bitpack::unpack_indices_into;
use ematix_parquet_codec::dict::{
    decode_rle_dictionary_indices, decode_rle_dictionary_predicate_bitmap_bw12,
};

/// Build an RLE_DICTIONARY data-page body (bw=12 prefix + one large
/// bit-packed run covering all values). Returns body bytes that
/// match the on-wire format a real writer would emit for a
/// fully-bit-packed page.
fn make_body_bitpacked(values: &[u32]) -> Vec<u8> {
    // Layout: byte 0 = bit_width; bytes 1.. = RLE/bit-packed stream.
    // Bit-packed run header is a uvarint encoding (num_groups << 1) | 1,
    // where each group = 8 values.
    let n = values.len();
    let num_groups = n.div_ceil(8);
    let mut padded = values.to_vec();
    padded.resize(num_groups * 8, 0);

    let bit_packed_payload = pack(&padded, 12);

    let mut out = vec![12u8];
    let header = ((num_groups as u64) << 1) | 1;
    write_uvarint(&mut out, header);
    out.extend(bit_packed_payload);
    out
}

/// Body with one RLE run repeating a single dict index.
fn make_body_rle(idx: u32, count: usize) -> Vec<u8> {
    // RLE run header: count << 1 (low bit = 0 for RLE).
    let mut out = vec![12u8];
    let header = (count as u64) << 1;
    write_uvarint(&mut out, header);
    // Value bytes: ceil(bit_width / 8) = 2 bytes for bw=12. LE.
    out.push((idx & 0xFF) as u8);
    out.push(((idx >> 8) & 0xFF) as u8);
    out
}

fn write_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let bw_mask: u64 = (1u64 << bit_width) - 1;
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

fn padded_dict_mask(matches: &[u32]) -> Vec<u8> {
    let mut m = vec![0u8; 4096];
    for &i in matches {
        m[i as usize] = 1;
    }
    m
}

fn reference_bitmap(body: &[u8], n: usize, dict_mask: &[u8]) -> Vec<u8> {
    let indices = decode_rle_dictionary_indices(body, n).unwrap();
    let mut bitmap = vec![0u8; n.div_ceil(8)];
    for (row, idx) in indices.into_iter().enumerate() {
        let bit = dict_mask[idx as usize];
        bitmap[row / 8] |= bit << (row % 8);
    }
    bitmap
}

#[test]
fn pure_bitpacked_run_matches_reference() {
    let n: usize = 10_000;
    let mut seed: u32 = 0xC0DE_FACE;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & 0xFFF
        })
        .collect();
    let body = make_body_bitpacked(&values);
    let dict_mask = padded_dict_mask(&(1000..1040).collect::<Vec<_>>());

    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, n, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&body, n, &dict_mask);
    assert_eq!(bitmap, expected);
}

#[test]
fn single_rle_run_matches_reference() {
    // Page with one RLE run: index 1023 repeated 4096 times.
    let body = make_body_rle(1023, 4096);
    let mut dict_mask = vec![0u8; 4096];
    dict_mask[1023] = 1;

    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, 4096, &dict_mask, &mut bitmap).unwrap();
    assert_eq!(bitmap, vec![0xFFu8; 512]);

    // And a non-matching RLE.
    let mut dict_mask = vec![0u8; 4096];
    dict_mask[42] = 1; // index 42 is the only match; our run is 1023.
    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, 4096, &dict_mask, &mut bitmap).unwrap();
    assert_eq!(bitmap, vec![0u8; 512]);
}

#[test]
fn mixed_rle_then_bitpacked_matches_reference() {
    // First an RLE run of 64 of index=10, then a bit-packed run.
    let mut body = vec![12u8];
    // RLE: count=64, idx=10.
    write_uvarint(&mut body, 64u64 << 1);
    body.push(10u8);
    body.push(0u8);
    // Bit-packed: 128 values (16 groups of 8).
    let bp_values: Vec<u32> = (0..128u32).map(|i| (i * 31) % 4096).collect();
    write_uvarint(&mut body, (16u64 << 1) | 1);
    body.extend(pack(&bp_values, 12));

    let n = 64 + 128;
    let dict_mask = padded_dict_mask(&[10, 31, 62, 93]); // some that match.

    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, n, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&body, n, &dict_mask);
    assert_eq!(bitmap, expected);
}

#[test]
fn partial_run_truncates_to_num_values() {
    // RLE run claims 64 values, but caller asks for only 50.
    let body = make_body_rle(7, 64);
    let mut dict_mask = vec![0u8; 4096];
    dict_mask[7] = 1;

    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, 50, &dict_mask, &mut bitmap).unwrap();
    // 50 bits set, then 6 unused bits in the last byte.
    assert_eq!(bitmap.len(), 7); // 50.div_ceil(8)
    assert_eq!(&bitmap[..6], &vec![0xFFu8; 6][..]);
    assert_eq!(bitmap[6], 0b00000011); // bits 0,1 set (rows 48,49); bits 2-7 zero.
}

#[test]
fn realistic_q14_shape_matches_reference() {
    // 1M values. Predicate matches ~1% of dict (Q14 shape).
    let n: usize = 1_000_000;
    let mut seed: u32 = 0xFEEDFACE;
    let values: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed % 2525 // simulate l_shipdate-ish dict size
        })
        .collect();
    let body = make_body_bitpacked(&values);
    let dict_mask = padded_dict_mask(&(1200..1230).collect::<Vec<_>>());

    let mut bitmap = Vec::new();
    decode_rle_dictionary_predicate_bitmap_bw12(&body, n, &dict_mask, &mut bitmap).unwrap();
    let expected = reference_bitmap(&body, n, &dict_mask);
    assert_eq!(bitmap, expected);

    let match_count: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
    let expected_count = values.iter().filter(|v| **v >= 1200 && **v < 1230).count();
    assert_eq!(match_count, expected_count);
}

#[test]
fn wrong_bit_width_errors() {
    // First byte != 12.
    let body = vec![14u8, 0, 0, 0, 0, 0];
    let dict_mask = vec![0u8; 4096];
    let mut bitmap = Vec::new();
    let r = decode_rle_dictionary_predicate_bitmap_bw12(&body, 8, &dict_mask, &mut bitmap);
    assert!(r.is_err());
}

#[test]
fn unused_imports_compile() {
    // Keep `unpack_indices_into` in scope to confirm the helper still
    // builds with the new API; useful for downstream callers that
    // touch both paths.
    let _ = unpack_indices_into;
}
