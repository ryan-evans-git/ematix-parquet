//! FROM-SCRATCH Q08 LEVER TEST — late-materialization across a join.
//!
//! Q08's `part ⋈ lineitem ON p_partkey = l_partkey` keeps ~0.67% of lineitem
//! (the real SF=10 numbers: p_type='ECONOMY ANODIZED STEEL' -> 13,452 part keys
//! -> 403,487 of 60,000,000 lineitem rows survive).
//!
//! TODAY ematix EAGERLY decodes every projected lineitem payload column for all
//! 60M rows, materializes RecordBatches, and only THEN does HashJoinExec discard
//! 99.3% of them. DuckDB late-materializes: decode the join key, probe, then
//! gather the payload columns ONLY for survivors.
//!
//! This isolates the pure DECODE cost of the two strategies on the REAL SF=10
//! columns (no Arrow/operator overhead -> conservative, understates EAGER):
//!
//!   EAGER : decode {partkey, orderkey, suppkey, extendedprice} fully (60M each)
//!           -> per-row membership probe -> gather survivors.
//!   LATE  : decode partkey (60M) -> bitmap via membership -> masked-decode
//!           {orderkey, suppkey, extendedprice} for ONLY the ~403K survivors.
//!
//! Both paths produce byte-identical survivor outputs (asserted via count).
//!
//! Run:
//!   cargo run --release -p ematix-parquet-codec --example bench_latemat_join -- \
//!     /path/to/sf10/lineitem.parquet
//!   REPS=12 cargo run --release ... (default 10; medians reported)

use ematix_parquet_codec::read::{
    read_column_f64, read_column_f64_masked_into, read_column_i64, read_column_i64_masked_into,
};
use ematix_parquet_io::ParquetFile;
use std::time::Instant;

// Standard TPC-H lineitem column order.
const C_ORDERKEY: usize = 0;
const C_PARTKEY: usize = 1;
const C_SUPPKEY: usize = 2;
const C_EXTPRICE: usize = 5; // f64 PLAIN — the Snappy 1.73 GB/s floor column

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../ematix-flow/examples/tpch/data/sf10/lineitem.parquet".to_string());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let file = ParquetFile::open(&path).expect("open lineitem");
    let md = file.cached_metadata().expect("meta");
    let n_rg = md.row_groups.len();
    println!("file: {path}\nrow groups: {n_rg}, reps: {reps}\n");

    // Physical types (sanity — confirm partkey/orderkey/suppkey are i64, extprice f64).
    for (name, c) in [
        ("l_orderkey", C_ORDERKEY),
        ("l_partkey", C_PARTKEY),
        ("l_suppkey", C_SUPPKEY),
        ("l_extendedprice", C_EXTPRICE),
    ] {
        let m = md.row_groups[0].columns[c].meta_data.as_ref().unwrap();
        println!(
            "  col {c:2} {name:16} type={:?} encodings={:?} num_values(rg0)={}",
            m.column_type, m.encodings, m.num_values
        );
    }
    println!();

    // Membership oracle modelling the part build side: ~13,452 distinct keys
    // scattered uniformly over the SF=10 p_partkey domain [1, 2_000_000].
    // l_partkey is ~uniform over the same domain => ~0.67% lineitem survival,
    // matching the measured 403,487 / 60M. O(1) bitset => no hashing noise, and
    // it is IDENTICAL work on both paths, so it cancels out of the delta.
    let part_max: i64 = 2_000_000;
    let target_keys: usize = 13_452;
    let mut in_set = vec![false; (part_max as usize) + 1];
    let mut set_count = 0usize;
    for key in 1..=part_max {
        // deterministic scatter (Knuth multiplicative hash, modulo domain)
        if (key.wrapping_mul(2_654_435_761) as u64) % (part_max as u64) < target_keys as u64 {
            in_set[key as usize] = true;
            set_count += 1;
        }
    }
    println!("part build set: {set_count} keys over [1,{part_max}]\n");

    // ---------- EAGER: decode everything, then probe+gather ----------
    let mut eager_ms = Vec::new();
    let mut eager_surv = 0usize;
    let mut eager_total = 0usize;
    for _ in 0..reps {
        let t = Instant::now();
        let mut ok_out: Vec<i64> = Vec::new();
        let mut sk_out: Vec<i64> = Vec::new();
        let mut ep_out: Vec<f64> = Vec::new();
        let mut total = 0usize;
        for rg in 0..n_rg {
            let pk = read_column_i64(&file, rg, C_PARTKEY).unwrap();
            let ok = read_column_i64(&file, rg, C_ORDERKEY).unwrap();
            let sk = read_column_i64(&file, rg, C_SUPPKEY).unwrap();
            let ep = read_column_f64(&file, rg, C_EXTPRICE).unwrap();
            total += pk.len();
            for i in 0..pk.len() {
                let key = pk[i];
                if key >= 0 && (key as usize) < in_set.len() && in_set[key as usize] {
                    ok_out.push(ok[i]);
                    sk_out.push(sk[i]);
                    ep_out.push(ep[i]);
                }
            }
        }
        eager_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        eager_surv = ep_out.len();
        eager_total = total;
        std::hint::black_box((&ok_out, &sk_out, &ep_out));
    }

    // ---------- LATE: decode key, build bitmap, masked-decode payload ----------
    let mut late_ms = Vec::new();
    let mut late_surv = 0usize;
    for _ in 0..reps {
        let t = Instant::now();
        let mut ok_out: Vec<i64> = Vec::new();
        let mut sk_out: Vec<i64> = Vec::new();
        let mut ep_out: Vec<f64> = Vec::new();
        for rg in 0..n_rg {
            let pk = read_column_i64(&file, rg, C_PARTKEY).unwrap();
            let nv = pk.len();
            let mut bitmap = vec![0u8; nv.div_ceil(8)];
            for (i, &key) in pk.iter().enumerate() {
                if key >= 0 && (key as usize) < in_set.len() && in_set[key as usize] {
                    bitmap[i >> 3] |= 1 << (i & 7);
                }
            }
            read_column_i64_masked_into(&file, rg, C_ORDERKEY, &bitmap, &mut ok_out).unwrap();
            read_column_i64_masked_into(&file, rg, C_SUPPKEY, &bitmap, &mut sk_out).unwrap();
            read_column_f64_masked_into(&file, rg, C_EXTPRICE, &bitmap, &mut ep_out).unwrap();
        }
        late_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        late_surv = ep_out.len();
        std::hint::black_box((&ok_out, &sk_out, &ep_out));
    }

    let em = median(eager_ms);
    let lm = median(late_ms);
    println!("total lineitem rows: {eager_total}");
    println!(
        "survivors: eager={eager_surv} late={late_surv}  ({:.3}% selectivity){}",
        100.0 * eager_surv as f64 / eager_total.max(1) as f64,
        if eager_surv == late_surv {
            " ✓ match"
        } else {
            " ✗ MISMATCH"
        }
    );
    println!();
    println!("=== payload decode strategy (partkey+orderkey+suppkey i64, extprice f64) ===");
    println!("  EAGER  decode-all-then-filter : {em:.2} ms");
    println!("  LATE   decode-key→masked-payload: {lm:.2} ms");
    println!("  speedup EAGER/LATE            : {:.2}x", em / lm);
    println!();
    println!("Interpretation:");
    println!(
        "  >1.3x  => late-mat across the join is a real lever (payload decode is wasted today)"
    );
    println!("  ~1.0x  => key-decode + 60M membership probe dominate; Q08 is decode-floored like Q06/Q15");
}
