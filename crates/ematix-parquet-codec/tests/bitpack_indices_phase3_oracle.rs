//! Phase 3 raw-indices oracle for bw=9, 10, 11, 13, 19.

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

const WIDTHS: &[u8] = &[9, 10, 11, 13, 19];

fn mask_for(bw: u8) -> u32 {
    (1u32 << bw) - 1
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use ematix_parquet_codec::bitpack_neon::{
        unpack_indices_into_neon_bw10, unpack_indices_into_neon_bw11,
        unpack_indices_into_neon_bw13, unpack_indices_into_neon_bw19, unpack_indices_into_neon_bw9,
    };

    fn check(width: u8, values: &[u32]) {
        let packed = pack(values, width);
        let mut got = Vec::new();
        match width {
            9 => unpack_indices_into_neon_bw9(&packed, values.len(), &mut got).unwrap(),
            10 => unpack_indices_into_neon_bw10(&packed, values.len(), &mut got).unwrap(),
            11 => unpack_indices_into_neon_bw11(&packed, values.len(), &mut got).unwrap(),
            13 => unpack_indices_into_neon_bw13(&packed, values.len(), &mut got).unwrap(),
            19 => unpack_indices_into_neon_bw19(&packed, values.len(), &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, values, "bw{width} NEON mismatch");
    }

    #[test]
    fn neon_all_widths_random() {
        for &bw in WIDTHS {
            let v = pseudo_random(0xC0DEC0DE ^ (bw as u32), 2048, mask_for(bw));
            check(bw, &v);
        }
    }

    #[test]
    fn neon_all_widths_partial_tail() {
        for &bw in WIDTHS {
            for n in [9usize, 17, 65, 100, 1023] {
                let v: Vec<u32> = (0..n as u32).map(|i| (i * 13) & mask_for(bw)).collect();
                check(bw, &v);
            }
        }
    }

    #[test]
    fn neon_all_widths_known() {
        for &bw in WIDTHS {
            let n = (mask_for(bw) as usize + 1).min(512);
            let v: Vec<u32> = (0..n as u32).collect();
            check(bw, &v);
        }
    }

    #[test]
    fn neon_dispatch_routes() {
        use ematix_parquet_codec::bitpack::unpack_indices_into;
        for &bw in WIDTHS {
            let v: Vec<u32> = (0..512u32).map(|i| (i * 23) & mask_for(bw)).collect();
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
        unpack_indices_into_avx2_bw10, unpack_indices_into_avx2_bw11,
        unpack_indices_into_avx2_bw13, unpack_indices_into_avx2_bw19, unpack_indices_into_avx2_bw9,
    };

    fn have_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    fn check(width: u8, values: &[u32]) {
        if !have_avx2() {
            return;
        }
        let packed = pack(values, width);
        let mut got = Vec::new();
        match width {
            9 => unpack_indices_into_avx2_bw9(&packed, values.len(), &mut got).unwrap(),
            10 => unpack_indices_into_avx2_bw10(&packed, values.len(), &mut got).unwrap(),
            11 => unpack_indices_into_avx2_bw11(&packed, values.len(), &mut got).unwrap(),
            13 => unpack_indices_into_avx2_bw13(&packed, values.len(), &mut got).unwrap(),
            19 => unpack_indices_into_avx2_bw19(&packed, values.len(), &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, values, "bw{width} AVX2 mismatch");
    }

    #[test]
    fn avx2_all_widths_random() {
        for &bw in WIDTHS {
            let v = pseudo_random(0xC0DEC0DE ^ (bw as u32), 2048, mask_for(bw));
            check(bw, &v);
        }
    }

    #[test]
    fn avx2_all_widths_partial_tail() {
        for &bw in WIDTHS {
            for n in [9usize, 17, 65, 100, 1023] {
                let v: Vec<u32> = (0..n as u32).map(|i| (i * 13) & mask_for(bw)).collect();
                check(bw, &v);
            }
        }
    }

    #[test]
    fn avx2_all_widths_known() {
        for &bw in WIDTHS {
            let n = (mask_for(bw) as usize + 1).min(512);
            let v: Vec<u32> = (0..n as u32).collect();
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
            let v: Vec<u32> = (0..512u32).map(|i| (i * 23) & mask_for(bw)).collect();
            let packed = pack(&v, bw);
            let mut got = Vec::new();
            unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
            assert_eq!(got, v, "bw{bw} dispatch mismatch");
        }
    }
}
