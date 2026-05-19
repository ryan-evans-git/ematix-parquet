//! AVX2 kernel oracle for bw=1, 4, 5, 8, 20, 21 — the widths added
//! alongside the NEON kernels for x86 parity. Mirrors
//! `bitpack_neon_small_widths.rs` shape.
//!
//! Gated on x86_64 with the AVX2 feature detected at run time (the
//! Linux CI runner has AVX2; macOS runners are aarch64 and skip).

#![cfg(target_arch = "x86_64")]

use ematix_parquet_codec::bitpack_avx2::{
    unpack_indices_into_avx2_bw1, unpack_indices_into_avx2_bw2, unpack_indices_into_avx2_bw20,
    unpack_indices_into_avx2_bw21, unpack_indices_into_avx2_bw3, unpack_indices_into_avx2_bw4,
    unpack_indices_into_avx2_bw5, unpack_indices_into_avx2_bw8,
};

fn have_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mask: u64 = (1u64 << bit_width) - 1;
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut byte_ix = 0usize;
    for &v in values {
        acc |= ((v as u64) & mask) << bits;
        bits += bit_width as u32;
        while bits >= 8 {
            out[byte_ix] = (acc & 0xFF) as u8;
            byte_ix += 1;
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out[byte_ix] = (acc & 0xFF) as u8;
    }
    out
}

fn pseudo_random(seed: u32, n: usize, mask: u32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s & mask
        })
        .collect()
}

// ---- per-width round-trip helpers -----------------------------------

fn check(width: u8, values: &[u32]) {
    if !have_avx2() {
        return;
    }
    let mut packed = pack(values, width);
    // bw=3 reads 16 bytes per SIMD block; pad the source so even tiny
    // inputs hit the SIMD path (real callers always have generous
    // trailing slack from page framing).
    if width == 3 {
        packed.resize(packed.len().max(64), 0);
    }
    let mut got = Vec::new();
    match width {
        1 => unpack_indices_into_avx2_bw1(&packed, values.len(), &mut got).unwrap(),
        2 => unpack_indices_into_avx2_bw2(&packed, values.len(), &mut got).unwrap(),
        3 => unpack_indices_into_avx2_bw3(&packed, values.len(), &mut got).unwrap(),
        4 => unpack_indices_into_avx2_bw4(&packed, values.len(), &mut got).unwrap(),
        5 => unpack_indices_into_avx2_bw5(&packed, values.len(), &mut got).unwrap(),
        8 => unpack_indices_into_avx2_bw8(&packed, values.len(), &mut got).unwrap(),
        20 => unpack_indices_into_avx2_bw20(&packed, values.len(), &mut got).unwrap(),
        21 => unpack_indices_into_avx2_bw21(&packed, values.len(), &mut got).unwrap(),
        _ => unreachable!(),
    }
    assert_eq!(got, values, "bw{width}: AVX2 output mismatch");
}

// ---- bw=1 -----------------------------------------------------------

#[test]
fn avx2_bw1_alternating() {
    check(1, &(0..32u32).map(|i| i & 1).collect::<Vec<_>>());
}

#[test]
fn avx2_bw1_all_zeros_and_ones() {
    check(1, &vec![0u32; 256]);
    check(1, &vec![1u32; 256]);
}

#[test]
fn avx2_bw1_partial_tail() {
    for n in [33usize, 65, 100, 1023] {
        check(1, &(0..n as u32).map(|i| (i * 7) & 1).collect::<Vec<_>>());
    }
}

#[test]
fn avx2_bw1_random() {
    check(1, &pseudo_random(0xC0FFEE, 2048, 1));
}

// ---- bw=2 -----------------------------------------------------------

#[test]
fn avx2_bw2_full_range() {
    check(2, &(0..32u32).map(|i| i & 0x03).collect::<Vec<_>>());
}

#[test]
fn avx2_bw2_4_streams_interleave() {
    // First byte should pack [0,1,2,3] LSB-first as 0xE4.
    let v: Vec<u32> = (0..32u32).map(|i| i & 0x03).collect();
    let packed = pack(&v, 2);
    assert_eq!(packed[0], 0xE4);
    check(2, &v);
}

#[test]
fn avx2_bw2_multi_block() {
    check(2, &(0..512u32).map(|i| i & 0x03).collect::<Vec<_>>());
}

#[test]
fn avx2_bw2_partial_tail() {
    for n in [33usize, 34, 35, 36, 64, 65, 100, 511, 1023] {
        check(
            2,
            &(0..n as u32).map(|i| (i * 7) & 0x03).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw2_random() {
    check(2, &pseudo_random(0xC0DEFACE, 2048, 0x03));
}

// ---- bw=3 -----------------------------------------------------------

#[test]
fn avx2_bw3_one_block() {
    check(3, &(0..8u32).collect::<Vec<_>>());
}

#[test]
fn avx2_bw3_full_range() {
    check(3, &(0..256u32).map(|i| i & 0x07).collect::<Vec<_>>());
}

#[test]
fn avx2_bw3_partial_tail() {
    for n in [9usize, 17, 33, 65, 100, 511, 1023] {
        check(
            3,
            &(0..n as u32).map(|i| (i * 13) & 0x07).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw3_random() {
    check(3, &pseudo_random(0xBADBEEF1, 2048, 0x07));
}

// ---- bw=4 -----------------------------------------------------------

#[test]
fn avx2_bw4_one_block() {
    check(4, &(0..32u32).map(|i| i & 0x0F).collect::<Vec<_>>());
}

#[test]
fn avx2_bw4_nibble_interleave() {
    let v: Vec<u32> = (0..32u32).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
    let packed = pack(&v, 4);
    assert_eq!(packed[0], 0x21);
    check(4, &v);
}

#[test]
fn avx2_bw4_partial_tail() {
    for n in [33usize, 34, 64, 65, 100, 511] {
        check(
            4,
            &(0..n as u32).map(|i| (i * 7) & 0x0F).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw4_random() {
    check(4, &pseudo_random(0xCAFEBABE, 1024, 0x0F));
}

// ---- bw=5 -----------------------------------------------------------

#[test]
fn avx2_bw5_full_range() {
    check(5, &(0..32u32).map(|i| i & 0x1F).collect::<Vec<_>>());
}

#[test]
fn avx2_bw5_partial_tail() {
    for n in [9usize, 17, 65, 100, 1023] {
        check(
            5,
            &(0..n as u32).map(|i| (i * 13) & 0x1F).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw5_random() {
    check(5, &pseudo_random(0xFADEBABE, 2048, 0x1F));
}

// ---- bw=8 -----------------------------------------------------------

#[test]
fn avx2_bw8_known() {
    check(8, &(0..32u32).collect::<Vec<_>>());
}

#[test]
fn avx2_bw8_descending() {
    check(8, &(0..256u32).rev().collect::<Vec<_>>());
}

#[test]
fn avx2_bw8_partial_tail() {
    for n in [33usize, 64, 65, 100, 1023] {
        check(
            8,
            &(0..n as u32).map(|i| (i * 31) & 0xFF).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw8_random() {
    check(8, &pseudo_random(0xDEADBEEF, 2048, 0xFF));
}

// ---- bw=20 ----------------------------------------------------------

#[test]
fn avx2_bw20_known() {
    check(
        20,
        &(0..8u32)
            .map(|i| i.wrapping_mul(0xA5A5))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn avx2_bw20_full_range() {
    check(
        20,
        &(0..32u32)
            .map(|i| (i * 31_337) & 0x0F_FFFF)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn avx2_bw20_partial_tail() {
    for n in [9usize, 17, 100, 1023] {
        check(
            20,
            &(0..n as u32)
                .map(|i| (i * 17) & 0x0F_FFFF)
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw20_random() {
    check(20, &pseudo_random(0xBADC0FFE, 2048, 0x0F_FFFF));
}

// ---- bw=21 ----------------------------------------------------------

#[test]
fn avx2_bw21_known() {
    check(
        21,
        &(0..8u32)
            .map(|i| i.wrapping_mul(0x12345))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn avx2_bw21_full_range() {
    check(
        21,
        &(0..32u32)
            .map(|i| (i * 31_337) & 0x1F_FFFF)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn avx2_bw21_partial_tail() {
    for n in [9usize, 17, 100, 1023] {
        check(
            21,
            &(0..n as u32)
                .map(|i| (i * 17) & 0x1F_FFFF)
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
fn avx2_bw21_random() {
    check(21, &pseudo_random(0xFEEDC0DE, 2048, 0x1F_FFFF));
}

// ---- dispatch routing -----------------------------------------------

#[test]
fn dispatch_routes_new_widths_through_avx2() {
    if !have_avx2() {
        return;
    }
    use ematix_parquet_codec::bitpack::unpack_indices_into;
    for &bw in &[1u8, 4, 5, 8, 20, 21] {
        let mask: u32 = (1u32 << bw) - 1;
        let v: Vec<u32> = (0..256u32)
            .map(|i| (i.wrapping_mul(1_597_463)) & mask)
            .collect();
        let packed = pack(&v, bw);
        let mut got = Vec::new();
        unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
        assert_eq!(got, v, "bw{bw} dispatch mismatch");
    }
}

// ---- lookup variants (bw=4 / 6 / 8) ---------------------------------

fn check_lookup_avx2<T: Copy + std::fmt::Debug + PartialEq>(bw: u8, indices: &[u32], dict: &[T]) {
    if !have_avx2() {
        return;
    }
    use ematix_parquet_codec::bitpack_avx2::{
        unpack_lookup_into_avx2_bw4, unpack_lookup_into_avx2_bw6, unpack_lookup_into_avx2_bw8,
    };
    let mut packed = pack(indices, bw);
    if bw == 6 {
        packed.resize(packed.len().max(64), 0);
    }
    let mut got: Vec<T> = Vec::new();
    match bw {
        4 => unpack_lookup_into_avx2_bw4(&packed, indices.len(), dict, &mut got).unwrap(),
        6 => unpack_lookup_into_avx2_bw6(&packed, indices.len(), dict, &mut got).unwrap(),
        8 => unpack_lookup_into_avx2_bw8(&packed, indices.len(), dict, &mut got).unwrap(),
        _ => unreachable!(),
    }
    let expected: Vec<T> = indices.iter().map(|&i| dict[i as usize]).collect();
    assert_eq!(got, expected, "bw{bw}: AVX2 lookup mismatch");
}

#[test]
fn avx2_lookup_bw4_round_trip() {
    let dict: Vec<i64> = (1000..1016i64).collect();
    let indices: Vec<u32> = (0..256u32).map(|i| i & 0x0F).collect();
    check_lookup_avx2(4, &indices, &dict);
}

#[test]
fn avx2_lookup_bw4_tail() {
    let dict: Vec<f64> = (0..16).map(|i| i as f64 * 0.25).collect();
    for n in [33usize, 65, 100, 511, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 7) & 0x0F).collect();
        check_lookup_avx2(4, &indices, &dict);
    }
}

#[test]
fn avx2_lookup_bw4_small_dict_bounds_path() {
    let dict: Vec<u8> = (0..15).collect();
    let indices: Vec<u32> = (0..64u32).map(|i| i & 0x0E).collect();
    check_lookup_avx2(4, &indices, &dict);
}

#[test]
fn avx2_lookup_bw4_out_of_range_errors() {
    if !have_avx2() {
        return;
    }
    use ematix_parquet_codec::bitpack_avx2::unpack_lookup_into_avx2_bw4;
    let dict: Vec<u8> = (0..8).collect();
    let indices: Vec<u32> = (0..32u32)
        .map(|i| if i == 17 { 12 } else { i & 7 })
        .collect();
    let packed = pack(&indices, 4);
    let mut got: Vec<u8> = Vec::new();
    let r = unpack_lookup_into_avx2_bw4(&packed, indices.len(), &dict, &mut got);
    assert!(r.is_err());
}

#[test]
fn avx2_lookup_bw6_round_trip() {
    let dict: Vec<i32> = (2000..2064i32).collect();
    let indices: Vec<u32> = (0..256u32).map(|i| i & 0x3F).collect();
    check_lookup_avx2(6, &indices, &dict);
}

#[test]
fn avx2_lookup_bw6_tail() {
    let dict: Vec<f64> = (0..64).map(|i| i as f64).collect();
    for n in [9usize, 17, 33, 65, 100, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 13) & 0x3F).collect();
        check_lookup_avx2(6, &indices, &dict);
    }
}

#[test]
fn avx2_lookup_bw6_bounds_checked_path() {
    let dict: Vec<u16> = (0..50).collect();
    let indices: Vec<u32> = (0..128u32).map(|i| i % 50).collect();
    check_lookup_avx2(6, &indices, &dict);
}

#[test]
fn avx2_lookup_bw8_round_trip() {
    let dict: Vec<u64> = (10_000..10_256u64).collect();
    let indices: Vec<u32> = (0..1024u32).map(|i| i & 0xFF).collect();
    check_lookup_avx2(8, &indices, &dict);
}

#[test]
fn avx2_lookup_bw8_tail() {
    let dict: Vec<i64> = (0..256i64).collect();
    for n in [33usize, 64, 100, 1023] {
        let indices: Vec<u32> = (0..n as u32).map(|i| (i * 31) & 0xFF).collect();
        check_lookup_avx2(8, &indices, &dict);
    }
}

#[test]
fn avx2_lookup_dispatch_routes_bw4_6_8() {
    if !have_avx2() {
        return;
    }
    use ematix_parquet_codec::bitpack::unpack_lookup_into;
    // bw=4
    {
        let dict: Vec<u64> = (0..16u64).collect();
        let indices: Vec<u32> = (0..256u32).map(|i| i & 0x0F).collect();
        let packed = pack(&indices, 4);
        let mut got: Vec<u64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 4, &dict, &mut got).unwrap();
        let expected: Vec<u64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
    // bw=6
    {
        let dict: Vec<u64> = (0..64u64).collect();
        let indices: Vec<u32> = (0..256u32).map(|i| (i * 5) & 0x3F).collect();
        let mut packed = pack(&indices, 6);
        packed.resize(packed.len().max(64), 0);
        let mut got: Vec<u64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 6, &dict, &mut got).unwrap();
        let expected: Vec<u64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
    // bw=8
    {
        let dict: Vec<i64> = (0..256i64).collect();
        let indices: Vec<u32> = (0..1024u32).map(|i| (i * 13) & 0xFF).collect();
        let packed = pack(&indices, 8);
        let mut got: Vec<i64> = Vec::new();
        unpack_lookup_into(&packed, indices.len(), 8, &dict, &mut got).unwrap();
        let expected: Vec<i64> = indices.iter().map(|&i| dict[i as usize]).collect();
        assert_eq!(got, expected);
    }
}
