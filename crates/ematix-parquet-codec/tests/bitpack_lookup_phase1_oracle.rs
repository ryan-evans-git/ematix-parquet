//! Phase 1 fused-lookup oracle for bw=1, 2, 3, 5, 20, 21.
//!
//! Mirrors the raw-indices oracle pattern: pack known values
//! LSB-first, decode through the SIMD lookup kernel, and compare
//! against the gathered ground truth (`dict[indices[i]]`).
//!
//! Exercises both the bounds-safe fast path (dict_size > (1 << bw)-1)
//! and the bounds-checked path (small dict).

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

fn expected_gather<T: Copy>(indices: &[u32], dict: &[T]) -> Vec<T> {
    indices.iter().map(|&i| dict[i as usize]).collect()
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

// ---- NEON oracle (aarch64) -----------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use ematix_parquet_codec::bitpack_neon::{
        unpack_lookup_into_neon_bw1, unpack_lookup_into_neon_bw2, unpack_lookup_into_neon_bw20,
        unpack_lookup_into_neon_bw21, unpack_lookup_into_neon_bw3, unpack_lookup_into_neon_bw5,
    };

    fn check(width: u8, indices: &[u32], dict: &[u64]) {
        let packed = pack(indices, width);
        let mut got: Vec<u64> = Vec::new();
        match width {
            1 => unpack_lookup_into_neon_bw1(&packed, indices.len(), dict, &mut got).unwrap(),
            2 => unpack_lookup_into_neon_bw2(&packed, indices.len(), dict, &mut got).unwrap(),
            3 => unpack_lookup_into_neon_bw3(&packed, indices.len(), dict, &mut got).unwrap(),
            5 => unpack_lookup_into_neon_bw5(&packed, indices.len(), dict, &mut got).unwrap(),
            20 => unpack_lookup_into_neon_bw20(&packed, indices.len(), dict, &mut got).unwrap(),
            21 => unpack_lookup_into_neon_bw21(&packed, indices.len(), dict, &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, expected_gather(indices, dict), "bw{width} mismatch");
    }

    // Build a dict of `n` u64 entries with a recognisable pattern.
    fn dict_u64(n: usize) -> Vec<u64> {
        (0..n as u64)
            .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xFEEDFACEDEADBEEF)
            .collect()
    }

    // ---- bw=1 ----

    #[test]
    fn neon_bw1_bounds_safe_path() {
        let dict = dict_u64(2); // == (1 << 1), > 1 → bounds-safe
        let idx: Vec<u32> = pseudo_random(0xABCD, 1024, 1);
        check(1, &idx, &dict);
    }

    #[test]
    fn neon_bw1_bounds_checked_path() {
        let dict = dict_u64(2); // == 2, > 1 still safe path (need dict_size > 1)
                                // To force the bounds-checked path, use a tiny dict of size 1 and
                                // ensure all indices are 0.
        let dict_tiny = dict_u64(1);
        let idx_zeros: Vec<u32> = vec![0u32; 256];
        check(1, &idx_zeros, &dict_tiny);
        // Verify out-of-range errors out on the checked path.
        let bad_idx: Vec<u32> = vec![1u32; 32];
        let packed = pack(&bad_idx, 1);
        let mut out: Vec<u64> = Vec::new();
        let r = unpack_lookup_into_neon_bw1(&packed, bad_idx.len(), &dict_tiny, &mut out);
        assert!(r.is_err(), "out-of-range must error on bounds-checked path");
        let _ = dict; // silence warning
    }

    // ---- bw=2 ----

    #[test]
    fn neon_bw2_bounds_safe_path() {
        let dict = dict_u64(4);
        let idx: Vec<u32> = pseudo_random(0x1111, 2048, 0x03);
        check(2, &idx, &dict);
    }

    #[test]
    fn neon_bw2_partial_tail() {
        let dict = dict_u64(4);
        for n in [33usize, 65, 100, 511] {
            let idx: Vec<u32> = pseudo_random(0x2222, n, 0x03);
            check(2, &idx, &dict);
        }
    }

    // ---- bw=3 ----

    #[test]
    fn neon_bw3_bounds_safe_path() {
        let dict = dict_u64(8);
        let idx: Vec<u32> = pseudo_random(0x3333, 1024, 0x07);
        check(3, &idx, &dict);
    }

    #[test]
    fn neon_bw3_bounds_checked_path() {
        let dict = dict_u64(5);
        let idx: Vec<u32> = (0..512u32).map(|i| i % 5).collect();
        check(3, &idx, &dict);
    }

    // ---- bw=5 ----

    #[test]
    fn neon_bw5_bounds_safe_path() {
        let dict = dict_u64(32);
        let idx: Vec<u32> = pseudo_random(0x5555, 1024, 0x1F);
        check(5, &idx, &dict);
    }

    #[test]
    fn neon_bw5_bounds_checked_path() {
        let dict = dict_u64(20);
        let idx: Vec<u32> = (0..256u32).map(|i| i % 20).collect();
        check(5, &idx, &dict);
    }

    // ---- bw=20 ----

    #[test]
    fn neon_bw20_bounds_checked_path() {
        // Realistic: dict size 100, indices well within range.
        let dict = dict_u64(100);
        let idx: Vec<u32> = (0..128u32).map(|i| (i * 7) % 100).collect();
        check(20, &idx, &dict);
    }

    #[test]
    fn neon_bw20_full_range_random() {
        // Random indices up to dict size 4096 (well below (1<<20)-1).
        let dict = dict_u64(4096);
        let idx: Vec<u32> = pseudo_random(0xBADC0FFE, 1024, 4095);
        check(20, &idx, &dict);
    }

    // ---- bw=21 ----

    #[test]
    fn neon_bw21_bounds_checked_path() {
        let dict = dict_u64(200);
        let idx: Vec<u32> = (0..128u32).map(|i| (i * 13) % 200).collect();
        check(21, &idx, &dict);
    }

    #[test]
    fn neon_bw21_full_range_random() {
        let dict = dict_u64(8192);
        let idx: Vec<u32> = pseudo_random(0xFEEDC0DE, 1024, 8191);
        check(21, &idx, &dict);
    }

    // ---- dispatch end-to-end ----

    #[test]
    fn dispatch_routes_phase1_widths_through_neon() {
        use ematix_parquet_codec::bitpack::unpack_lookup_into;
        for &bw in &[1u8, 2, 3, 5, 20, 21] {
            let mask: u32 = (1u32 << bw) - 1;
            let dict_size = (mask as usize + 1).min(4096);
            let dict = dict_u64(dict_size);
            let idx: Vec<u32> = (0..256u32)
                .map(|i| (i.wrapping_mul(1_597_463)) % dict_size as u32)
                .collect();
            let packed = pack(&idx, bw);
            let mut got: Vec<u64> = Vec::new();
            unpack_lookup_into(&packed, idx.len(), bw, &dict, &mut got).unwrap();
            assert_eq!(
                got,
                expected_gather(&idx, &dict),
                "bw{bw} dispatch mismatch"
            );
        }
    }
}

// ---- AVX2 oracle (x86_64) ------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use ematix_parquet_codec::bitpack_avx2::{
        unpack_lookup_into_avx2_bw1, unpack_lookup_into_avx2_bw2, unpack_lookup_into_avx2_bw20,
        unpack_lookup_into_avx2_bw21, unpack_lookup_into_avx2_bw3, unpack_lookup_into_avx2_bw5,
    };

    fn have_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    fn check(width: u8, indices: &[u32], dict: &[u64]) {
        if !have_avx2() {
            return;
        }
        let packed = pack(indices, width);
        let mut got: Vec<u64> = Vec::new();
        match width {
            1 => unpack_lookup_into_avx2_bw1(&packed, indices.len(), dict, &mut got).unwrap(),
            2 => unpack_lookup_into_avx2_bw2(&packed, indices.len(), dict, &mut got).unwrap(),
            3 => unpack_lookup_into_avx2_bw3(&packed, indices.len(), dict, &mut got).unwrap(),
            5 => unpack_lookup_into_avx2_bw5(&packed, indices.len(), dict, &mut got).unwrap(),
            20 => unpack_lookup_into_avx2_bw20(&packed, indices.len(), dict, &mut got).unwrap(),
            21 => unpack_lookup_into_avx2_bw21(&packed, indices.len(), dict, &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, expected_gather(indices, dict), "bw{width} mismatch");
    }

    fn dict_u64(n: usize) -> Vec<u64> {
        (0..n as u64)
            .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xFEEDFACEDEADBEEF)
            .collect()
    }

    #[test]
    fn avx2_bw1_random() {
        let dict = dict_u64(2);
        check(1, &pseudo_random(0xABCD, 1024, 1), &dict);
    }

    #[test]
    fn avx2_bw2_random() {
        let dict = dict_u64(4);
        check(2, &pseudo_random(0x1111, 2048, 0x03), &dict);
    }

    #[test]
    fn avx2_bw3_random() {
        let dict = dict_u64(8);
        check(3, &pseudo_random(0x3333, 1024, 0x07), &dict);
    }

    #[test]
    fn avx2_bw3_bounds_checked() {
        let dict = dict_u64(5);
        check(3, &(0..512u32).map(|i| i % 5).collect::<Vec<_>>(), &dict);
    }

    #[test]
    fn avx2_bw5_random() {
        let dict = dict_u64(32);
        check(5, &pseudo_random(0x5555, 1024, 0x1F), &dict);
    }

    #[test]
    fn avx2_bw5_bounds_checked() {
        let dict = dict_u64(20);
        check(5, &(0..256u32).map(|i| i % 20).collect::<Vec<_>>(), &dict);
    }

    #[test]
    fn avx2_bw20_random() {
        let dict = dict_u64(4096);
        check(20, &pseudo_random(0xBADC0FFE, 1024, 4095), &dict);
    }

    #[test]
    fn avx2_bw21_random() {
        let dict = dict_u64(8192);
        check(21, &pseudo_random(0xFEEDC0DE, 1024, 8191), &dict);
    }

    #[test]
    fn avx2_dispatch_routes_phase1_widths() {
        if !have_avx2() {
            return;
        }
        use ematix_parquet_codec::bitpack::unpack_lookup_into;
        for &bw in &[1u8, 2, 3, 5, 20, 21] {
            let mask: u32 = (1u32 << bw) - 1;
            let dict_size = (mask as usize + 1).min(4096);
            let dict = dict_u64(dict_size);
            let idx: Vec<u32> = (0..256u32)
                .map(|i| (i.wrapping_mul(1_597_463)) % dict_size as u32)
                .collect();
            let packed = pack(&idx, bw);
            let mut got: Vec<u64> = Vec::new();
            unpack_lookup_into(&packed, idx.len(), bw, &dict, &mut got).unwrap();
            assert_eq!(
                got,
                expected_gather(&idx, &dict),
                "bw{bw} dispatch mismatch"
            );
        }
    }
}
