//! Phase 4 raw-indices oracle for bw=22..32.
//!
//! Wide-bit-width SIMD specialisations. bw=22..25 use u32 staging;
//! bw=26..31 use u64 staging; bw=32 is byte-aligned trivial copy.

fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bit_width) - 1
    };
    let mut acc: u128 = 0;
    let mut bits: u32 = 0;
    let mut byte_ix = 0usize;
    for &v in values {
        acc |= ((v as u128) & mask as u128) << bits;
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

fn pseudo_random(seed: u64, n: usize, mask: u32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as u32) & mask
        })
        .collect()
}

const WIDTHS: &[u8] = &[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

fn mask_for(bw: u8) -> u32 {
    if bw == 32 {
        u32::MAX
    } else {
        (1u32 << bw) - 1
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use ematix_parquet_codec::bitpack_neon::{
        unpack_indices_into_neon_bw22, unpack_indices_into_neon_bw23,
        unpack_indices_into_neon_bw24, unpack_indices_into_neon_bw25,
        unpack_indices_into_neon_bw26, unpack_indices_into_neon_bw27,
        unpack_indices_into_neon_bw28, unpack_indices_into_neon_bw29,
        unpack_indices_into_neon_bw30, unpack_indices_into_neon_bw31,
        unpack_indices_into_neon_bw32,
    };

    fn dispatch(width: u8, packed: &[u8], n: usize, out: &mut Vec<u32>) {
        match width {
            22 => unpack_indices_into_neon_bw22(packed, n, out).unwrap(),
            23 => unpack_indices_into_neon_bw23(packed, n, out).unwrap(),
            24 => unpack_indices_into_neon_bw24(packed, n, out).unwrap(),
            25 => unpack_indices_into_neon_bw25(packed, n, out).unwrap(),
            26 => unpack_indices_into_neon_bw26(packed, n, out).unwrap(),
            27 => unpack_indices_into_neon_bw27(packed, n, out).unwrap(),
            28 => unpack_indices_into_neon_bw28(packed, n, out).unwrap(),
            29 => unpack_indices_into_neon_bw29(packed, n, out).unwrap(),
            30 => unpack_indices_into_neon_bw30(packed, n, out).unwrap(),
            31 => unpack_indices_into_neon_bw31(packed, n, out).unwrap(),
            32 => unpack_indices_into_neon_bw32(packed, n, out).unwrap(),
            _ => unreachable!(),
        }
    }

    fn check(width: u8, values: &[u32]) {
        let packed = pack(values, width);
        let mut got = Vec::new();
        dispatch(width, &packed, values.len(), &mut got);
        assert_eq!(got, values, "bw{width} NEON mismatch");
    }

    #[test]
    fn neon_all_widths_random() {
        for &bw in WIDTHS {
            let v = pseudo_random(0xC0DEC0DE ^ (bw as u64), 2048, mask_for(bw));
            check(bw, &v);
        }
    }

    #[test]
    fn neon_all_widths_partial_tail() {
        for &bw in WIDTHS {
            for n in [9usize, 17, 65, 100, 1023] {
                let v: Vec<u32> = (0..n as u32)
                    .map(|i| i.wrapping_mul(31_337) & mask_for(bw))
                    .collect();
                check(bw, &v);
            }
        }
    }

    #[test]
    fn neon_all_widths_max_value() {
        // Ensures the mask is wide enough — last value at the
        // boundary mask_for(bw).
        for &bw in WIDTHS {
            let v: Vec<u32> = vec![mask_for(bw); 64];
            check(bw, &v);
        }
    }

    #[test]
    fn neon_bw32_known_pattern() {
        // bw=32 is byte-aligned trivial copy; sanity-check identity.
        let v: Vec<u32> = (0..256u32).map(|i| i.wrapping_mul(0xDEADBEEF)).collect();
        check(32, &v);
    }

    #[test]
    fn neon_dispatch_routes() {
        use ematix_parquet_codec::bitpack::unpack_indices_into;
        for &bw in WIDTHS {
            let v: Vec<u32> = (0..512u32)
                .map(|i| i.wrapping_mul(31_337) & mask_for(bw))
                .collect();
            let packed = pack(&v, bw);
            let mut got = Vec::new();
            unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
            assert_eq!(got, v, "bw{bw} dispatch mismatch");
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use ematix_parquet_codec::bitpack_avx2::{
        unpack_indices_into_avx2_bw22, unpack_indices_into_avx2_bw23,
        unpack_indices_into_avx2_bw24, unpack_indices_into_avx2_bw25,
        unpack_indices_into_avx2_bw26, unpack_indices_into_avx2_bw27,
        unpack_indices_into_avx2_bw28, unpack_indices_into_avx2_bw29,
        unpack_indices_into_avx2_bw30, unpack_indices_into_avx2_bw31,
        unpack_indices_into_avx2_bw32,
    };

    fn have_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    fn dispatch(width: u8, packed: &[u8], n: usize, out: &mut Vec<u32>) {
        match width {
            22 => unpack_indices_into_avx2_bw22(packed, n, out).unwrap(),
            23 => unpack_indices_into_avx2_bw23(packed, n, out).unwrap(),
            24 => unpack_indices_into_avx2_bw24(packed, n, out).unwrap(),
            25 => unpack_indices_into_avx2_bw25(packed, n, out).unwrap(),
            26 => unpack_indices_into_avx2_bw26(packed, n, out).unwrap(),
            27 => unpack_indices_into_avx2_bw27(packed, n, out).unwrap(),
            28 => unpack_indices_into_avx2_bw28(packed, n, out).unwrap(),
            29 => unpack_indices_into_avx2_bw29(packed, n, out).unwrap(),
            30 => unpack_indices_into_avx2_bw30(packed, n, out).unwrap(),
            31 => unpack_indices_into_avx2_bw31(packed, n, out).unwrap(),
            32 => unpack_indices_into_avx2_bw32(packed, n, out).unwrap(),
            _ => unreachable!(),
        }
    }

    fn check(width: u8, values: &[u32]) {
        if !have_avx2() {
            return;
        }
        let packed = pack(values, width);
        let mut got = Vec::new();
        dispatch(width, &packed, values.len(), &mut got);
        assert_eq!(got, values, "bw{width} AVX2 mismatch");
    }

    #[test]
    fn avx2_all_widths_random() {
        for &bw in WIDTHS {
            let v = pseudo_random(0xC0DEC0DE ^ (bw as u64), 2048, mask_for(bw));
            check(bw, &v);
        }
    }

    #[test]
    fn avx2_all_widths_partial_tail() {
        for &bw in WIDTHS {
            for n in [9usize, 17, 65, 100, 1023] {
                let v: Vec<u32> = (0..n as u32)
                    .map(|i| i.wrapping_mul(31_337) & mask_for(bw))
                    .collect();
                check(bw, &v);
            }
        }
    }

    #[test]
    fn avx2_all_widths_max_value() {
        for &bw in WIDTHS {
            let v: Vec<u32> = vec![mask_for(bw); 64];
            check(bw, &v);
        }
    }

    #[test]
    fn avx2_dispatch_routes() {
        if !have_avx2() {
            return;
        }
        use ematix_parquet_codec::bitpack::unpack_indices_into;
        for &bw in WIDTHS {
            let v: Vec<u32> = (0..512u32)
                .map(|i| i.wrapping_mul(31_337) & mask_for(bw))
                .collect();
            let packed = pack(&v, bw);
            let mut got = Vec::new();
            unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
            assert_eq!(got, v, "bw{bw} dispatch mismatch");
        }
    }
}
