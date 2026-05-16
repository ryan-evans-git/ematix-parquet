//! Microbenchmark: AES-GCM seal+open throughput at typical Parquet
//! page sizes (4 / 64 / 256 / 1024 KB). Single-line per size; used to
//! confirm AES is not the bottleneck before each later Π.13 phase.
//!
//! Run: `cargo run --release --example bench_aes_gcm -p ematix-parquet-crypto`

use std::time::Instant;

use ematix_parquet_crypto::aead::{open, seal};
use ematix_parquet_crypto::key::Key;

const SIZES_KB: &[usize] = &[4, 64, 256, 1024];
const ITERS: usize = 200;
const WARMUP: usize = 20;

fn main() {
    let key = Key::Aes256([7u8; 32]);
    let nonce = [42u8; 12];
    let aad = b"bench_aad";

    println!("AES-256-GCM throughput (seal + open, single-thread)");
    println!("{:<10} {:>12} {:>12}", "size", "seal MiB/s", "open MiB/s");
    for &kb in SIZES_KB {
        let plaintext = vec![0xABu8; kb * 1024];
        for _ in 0..WARMUP {
            let ct = seal(&key, &nonce, aad, &plaintext).unwrap();
            let _ = open(&key, &nonce, aad, &ct).unwrap();
        }

        // Seal timing.
        let seal_start = Instant::now();
        let mut last_ct = Vec::new();
        for _ in 0..ITERS {
            last_ct = seal(&key, &nonce, aad, &plaintext).unwrap();
        }
        let seal_secs = seal_start.elapsed().as_secs_f64();
        let seal_mibps = (kb as f64 * ITERS as f64) / 1024.0 / seal_secs;

        // Open timing.
        let open_start = Instant::now();
        for _ in 0..ITERS {
            let _ = open(&key, &nonce, aad, &last_ct).unwrap();
        }
        let open_secs = open_start.elapsed().as_secs_f64();
        let open_mibps = (kb as f64 * ITERS as f64) / 1024.0 / open_secs;

        println!(
            "{:<10} {:>12.0} {:>12.0}",
            format!("{kb}KB"),
            seal_mibps,
            open_mibps
        );
    }
}
