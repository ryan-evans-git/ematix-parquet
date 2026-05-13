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

fn main() {
    println!(
        "bit-unpack microbench ({} values × {} iters, best-of)",
        N_VALUES, ITERS
    );
    // Workload-relevant widths: l_shipdate ~14, dict-encoded
    // categorical columns 1..6, and full-width 32 for comparison.
    let widths: &[u8] = &[1, 4, 8, 10, 12, 14, 16, 17, 20, 24, 32];
    for &w in widths {
        run_one(w);
    }
}
