//! Micro-bench for the scalar const-generic bit-unpacker.
//!
//! Establishes a baseline for any future SIMD work: if scalar is
//! already at L1 bandwidth, SIMD won't help; if there's a 2× gap,
//! it's worth the engineering. Times 1M values per bit_width across
//! the widths that show up in real workloads (notably l_shipdate
//! at ~14 bits and similar dict-encoded numerics).
//!
//! Output:
//!   - elapsed, ns/value, GB/s of unpacked u32 output
//!
//! Usage: cargo run --release --example bench_unpack

use std::hint::black_box;
use std::time::Instant;

use ematix_parquet_codec::bitpack::unpack_indices_into;
#[cfg(target_arch = "aarch64")]
use ematix_parquet_codec::bitpack_neon::{
    unpack_indices_into_neon_bw12, unpack_indices_into_neon_bw14,
    unpack_indices_into_neon_bw15, unpack_indices_into_neon_bw16,
    unpack_indices_into_neon_bw17, unpack_indices_into_neon_bw18,
};

const N_VALUES: usize = 1_000_000;
const ITERS: usize = 50;

/// Pack `n` u32 values of `bit_width` bits each (LSB-first, contiguous)
/// into a byte buffer. Mirrors the parquet bit-packed format.
fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = values.len() * bit_width as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else if bit_width == 0 {
        0
    } else {
        (1u64 << bit_width) - 1
    };
    for (i, &v) in values.iter().enumerate() {
        let v = (v as u64) & mask;
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

fn run_one(bit_width: u8) {
    let mask: u32 = if bit_width == 32 {
        u32::MAX
    } else {
        (1u32 << bit_width) - 1
    };
    // Pseudo-random values masked to bit_width — represents real
    // dict-index data.
    let mut seed: u32 = 0xC0FFEE ^ bit_width as u32;
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, bit_width);

    // Warmup
    let mut out: Vec<u32> = Vec::with_capacity(N_VALUES);
    for _ in 0..3 {
        out.clear();
        unpack_indices_into(black_box(&packed), N_VALUES, bit_width, &mut out).unwrap();
    }
    // Correctness sanity check
    assert_eq!(&out[..16], &values[..16]);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        unpack_indices_into(black_box(&packed), N_VALUES, bit_width, &mut out).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let bytes_out = N_VALUES * 4; // u32 output
    let gbps_out = bytes_out as f64 / best / 1e9;
    let bytes_in = packed.len();
    let gbps_in = bytes_in as f64 / best / 1e9;
    println!(
        "  bit_width={:>2}: {:>7.3} ms  {:>5.2} ns/val  in={:>4.2} GB/s  out={:>4.2} GB/s",
        bit_width,
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out
    );
}

#[cfg(target_arch = "aarch64")]
fn run_predicate_fused_bw12() {
    use ematix_parquet_codec::bitpack_neon::decode_predicate_bitmap_neon_bw12;

    let mask: u32 = 0xFFF;
    let mut seed: u32 = 0xC0FFEE ^ 0xBEEFu32;
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, 12);
    // Q14-shape predicate: dict_mask matches ~1% of slots.
    let mut dict_mask = vec![0u8; 4096];
    for i in 1000..1040 {
        dict_mask[i] = 1;
    }

    let mut bitmap: Vec<u8> = Vec::with_capacity(N_VALUES.div_ceil(8));

    // Warmup
    for _ in 0..3 {
        bitmap.clear();
        decode_predicate_bitmap_neon_bw12(black_box(&packed), N_VALUES, &dict_mask, &mut bitmap)
            .unwrap();
    }
    // Sanity check
    let match_count: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
    let expected = values.iter().filter(|v| **v >= 1000 && **v < 1040).count();
    assert_eq!(match_count, expected);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        bitmap.clear();
        let t0 = Instant::now();
        decode_predicate_bitmap_neon_bw12(black_box(&packed), N_VALUES, &dict_mask, &mut bitmap)
            .unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let bytes_out = N_VALUES.div_ceil(8);
    let gbps_out = bytes_out as f64 / best / 1e9;
    let gbps_in = packed.len() as f64 / best / 1e9;
    println!(
        "  bw=12 NEON-fused (idx + predicate → bitmap): {:>7.3} ms  {:>5.2} ns/val  in={:>4.2} GB/s  out={:>5.3} GB/s  ({} match)",
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out,
        match_count
    );
}

#[cfg(target_arch = "aarch64")]
fn run_neon_bw17() {
    let mask: u32 = 0x1FFFF;
    let mut seed: u32 = 0xC0FFEE ^ 17u32;
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, 17);

    let mut out: Vec<u32> = Vec::with_capacity(N_VALUES);
    for _ in 0..3 {
        out.clear();
        unpack_indices_into_neon_bw17(black_box(&packed), N_VALUES, &mut out).unwrap();
    }
    assert_eq!(&out[..16], &values[..16]);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        unpack_indices_into_neon_bw17(black_box(&packed), N_VALUES, &mut out).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let gbps_out = (N_VALUES * 4) as f64 / best / 1e9;
    let gbps_in = packed.len() as f64 / best / 1e9;
    println!(
        "  bw=17 NEON: {:>7.3} ms  {:>5.2} ns/val  in={:>5.2} GB/s  out={:>5.2} GB/s",
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out
    );
}

#[cfg(target_arch = "aarch64")]
fn run_neon_bw12() {
    let mask: u32 = 0xFFF;
    let mut seed: u32 = 0xC0FFEE ^ 12u32;
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, 12);

    let mut out: Vec<u32> = Vec::with_capacity(N_VALUES);
    for _ in 0..3 {
        out.clear();
        unpack_indices_into_neon_bw12(black_box(&packed), N_VALUES, &mut out).unwrap();
    }
    assert_eq!(&out[..16], &values[..16]);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        unpack_indices_into_neon_bw12(black_box(&packed), N_VALUES, &mut out).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let gbps_out = (N_VALUES * 4) as f64 / best / 1e9;
    let gbps_in = packed.len() as f64 / best / 1e9;
    println!(
        "  bw=12 NEON: {:>7.3} ms  {:>5.2} ns/val  in={:>4.2} GB/s  out={:>4.2} GB/s",
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out
    );
}

#[cfg(target_arch = "aarch64")]
fn run_neon_bw14() {
    let mask: u32 = 0x3FFF;
    let mut seed: u32 = 0xC0FFEE ^ 14u32;
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, 14);

    let mut out: Vec<u32> = Vec::with_capacity(N_VALUES);
    for _ in 0..3 {
        out.clear();
        unpack_indices_into_neon_bw14(black_box(&packed), N_VALUES, &mut out).unwrap();
    }
    assert_eq!(&out[..16], &values[..16]);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        unpack_indices_into_neon_bw14(black_box(&packed), N_VALUES, &mut out).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let gbps_out = (N_VALUES * 4) as f64 / best / 1e9;
    let gbps_in = packed.len() as f64 / best / 1e9;
    println!(
        "  bw=14 NEON: {:>7.3} ms  {:>5.2} ns/val  in={:>4.2} GB/s  out={:>4.2} GB/s",
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out
    );
}

#[cfg(target_arch = "aarch64")]
fn run_neon_simple(label: &str, bit_width: u8, kernel: fn(&[u8], usize, &mut Vec<u32>) -> ematix_parquet_codec::error::Result<()>) {
    let mask: u32 = if bit_width == 32 { u32::MAX } else { (1u32 << bit_width) - 1 };
    let mut seed: u32 = 0xC0FFEE ^ (bit_width as u32);
    let values: Vec<u32> = (0..N_VALUES)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed & mask
        })
        .collect();
    let packed = pack(&values, bit_width);

    let mut out: Vec<u32> = Vec::with_capacity(N_VALUES);
    for _ in 0..3 {
        out.clear();
        kernel(black_box(&packed), N_VALUES, &mut out).unwrap();
    }
    assert_eq!(&out[..16], &values[..16]);

    let mut best: f64 = f64::INFINITY;
    for _ in 0..ITERS {
        out.clear();
        let t0 = Instant::now();
        kernel(black_box(&packed), N_VALUES, &mut out).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_value = best * 1e9 / N_VALUES as f64;
    let gbps_out = (N_VALUES * 4) as f64 / best / 1e9;
    let gbps_in = packed.len() as f64 / best / 1e9;
    println!(
        "  {label}: {:>7.3} ms  {:>5.2} ns/val  in={:>5.2} GB/s  out={:>5.2} GB/s",
        best * 1e3,
        ns_per_value,
        gbps_in,
        gbps_out
    );
}

fn main() {
    println!(
        "bit-unpack microbench ({} values × {} iters, best-of)",
        N_VALUES, ITERS
    );
    // Workload-relevant widths: l_shipdate ~14, dict-encoded
    // categorical columns 1..6, and full-width 32 for comparison.
    let widths: &[u8] = &[1, 4, 8, 10, 12, 14, 15, 16, 17, 18, 20, 24, 32];
    for &w in widths {
        run_one(w);
    }
    #[cfg(target_arch = "aarch64")]
    {
        run_neon_bw12();
        run_neon_bw14();
        run_neon_simple("bw=15 NEON", 15, unpack_indices_into_neon_bw15);
        run_neon_simple("bw=16 NEON", 16, unpack_indices_into_neon_bw16);
        run_neon_bw17();
        run_neon_simple("bw=18 NEON", 18, unpack_indices_into_neon_bw18);
        run_predicate_fused_bw12();
    }
}
