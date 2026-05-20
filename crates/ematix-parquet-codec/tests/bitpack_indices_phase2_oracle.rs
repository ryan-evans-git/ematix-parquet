//! Phase 2 raw-indices oracle for bw=6, bw=7.
//!
//! Same bit-exact pack helper + known/random/partial-tail patterns
//! as the prior raw-indices oracles.

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

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use ematix_parquet_codec::bitpack_neon::{
        unpack_indices_into_neon_bw6, unpack_indices_into_neon_bw7,
    };

    fn check(width: u8, values: &[u32]) {
        let packed = pack(values, width);
        let mut got = Vec::new();
        match width {
            6 => unpack_indices_into_neon_bw6(&packed, values.len(), &mut got).unwrap(),
            7 => unpack_indices_into_neon_bw7(&packed, values.len(), &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, values, "bw{width}: NEON output mismatch");
    }

    #[test]
    fn neon_bw6_full_range() {
        check(6, &(0..64u32).collect::<Vec<_>>());
    }

    #[test]
    fn neon_bw6_partial_tail() {
        for n in [9usize, 17, 65, 100, 1023] {
            check(
                6,
                &(0..n as u32).map(|i| (i * 11) & 0x3F).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn neon_bw6_random() {
        check(6, &pseudo_random(0x6666, 2048, 0x3F));
    }

    #[test]
    fn neon_bw7_full_range() {
        check(7, &(0..128u32).collect::<Vec<_>>());
    }

    #[test]
    fn neon_bw7_partial_tail() {
        for n in [9usize, 17, 65, 100, 1023] {
            check(
                7,
                &(0..n as u32).map(|i| (i * 11) & 0x7F).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn neon_bw7_random() {
        check(7, &pseudo_random(0x7777, 2048, 0x7F));
    }

    #[test]
    fn neon_dispatch_routes_bw6_bw7() {
        use ematix_parquet_codec::bitpack::unpack_indices_into;
        for bw in [6u8, 7] {
            let mask: u32 = (1u32 << bw) - 1;
            let v: Vec<u32> = (0..512u32).map(|i| (i * 13) & mask).collect();
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
        unpack_indices_into_avx2_bw6, unpack_indices_into_avx2_bw7,
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
            6 => unpack_indices_into_avx2_bw6(&packed, values.len(), &mut got).unwrap(),
            7 => unpack_indices_into_avx2_bw7(&packed, values.len(), &mut got).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(got, values, "bw{width}: AVX2 output mismatch");
    }

    #[test]
    fn avx2_bw6_full_range() {
        check(6, &(0..64u32).collect::<Vec<_>>());
    }

    #[test]
    fn avx2_bw6_partial_tail() {
        for n in [9usize, 17, 65, 100, 1023] {
            check(
                6,
                &(0..n as u32).map(|i| (i * 11) & 0x3F).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn avx2_bw6_random() {
        check(6, &pseudo_random(0x6666, 2048, 0x3F));
    }

    #[test]
    fn avx2_bw7_full_range() {
        check(7, &(0..128u32).collect::<Vec<_>>());
    }

    #[test]
    fn avx2_bw7_partial_tail() {
        for n in [9usize, 17, 65, 100, 1023] {
            check(
                7,
                &(0..n as u32).map(|i| (i * 11) & 0x7F).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn avx2_bw7_random() {
        check(7, &pseudo_random(0x7777, 2048, 0x7F));
    }

    #[test]
    fn avx2_dispatch_routes_bw6_bw7() {
        if !have_avx2() {
            return;
        }
        use ematix_parquet_codec::bitpack::unpack_indices_into;
        for bw in [6u8, 7] {
            let mask: u32 = (1u32 << bw) - 1;
            let v: Vec<u32> = (0..512u32).map(|i| (i * 13) & mask).collect();
            let packed = pack(&v, bw);
            let mut got = Vec::new();
            unpack_indices_into(&packed, v.len(), bw, &mut got).unwrap();
            assert_eq!(got, v, "bw{bw} dispatch mismatch");
        }
    }
}
