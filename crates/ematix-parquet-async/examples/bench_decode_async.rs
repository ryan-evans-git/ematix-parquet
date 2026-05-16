//! Π.11f — async decode vs sync decode against the same local file.
//!
//! Acceptance bar from the Π.11 design doc:
//!   "On LocalFileSystem ObjectStore, async i64 decode of a TPC-H
//!    lineitem column is within 5% of the sync path. Confirms tokio
//!    scheduling isn't eating into our parser wins."
//!
//! This bench measures the codec-layer parity. Cloud-throughput
//! bench (against S3) is deferred to Π.11e once AWS credentials are
//! wired into a nightly CI job.
//!
//! Usage:
//!   cargo run --release --example bench_decode_async
//!   TPCH_DATA_DIR=/path/to/sf1 cargo run --release --example bench_decode_async

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ematix_parquet_async::{read_column_i64_async_into, AsyncParquetFile};
use ematix_parquet_codec::read::read_column_i64_into;
use ematix_parquet_io::ParquetFile;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;

const WARMUPS: usize = 3;
const ITERS: usize = 12;

// l_suppkey @ col index 2 — INT64, dict-encoded bw=14 (Π.8 hot path).
const COL_SUPPKEY: usize = 2;

fn data_dir() -> Option<PathBuf> {
    if let Ok(s) = std::env::var("TPCH_DATA_DIR") {
        let p = PathBuf::from(s);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .parent()?
        .parent()?
        .join("ematix-flow/examples/tpch/data/sf1");
    p.exists().then_some(p)
}

fn fmt(d: Duration) -> String {
    format!("{:>7.3} ms", d.as_secs_f64() * 1e3)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let dir = match data_dir() {
        Some(d) => d,
        None => {
            println!(
                "TPC-H SF=1 lineitem not found.\n\
                 Set TPCH_DATA_DIR=/path/to/sf1 or check out ematix-flow\n\
                 alongside this repo with its sf1 fixture."
            );
            return;
        }
    };
    let abs_path = dir.join("lineitem.parquet");
    if !abs_path.exists() {
        println!("lineitem.parquet not at {}", abs_path.display());
        return;
    }

    // Open sync + async views of the same on-disk file.
    let sync_file = ParquetFile::open(&abs_path).expect("sync open");
    let store = Arc::new(LocalFileSystem::new_with_prefix(&dir).unwrap());
    let aps = AsyncParquetFile::open(store, OsPath::from("lineitem.parquet"))
        .await
        .expect("async open");

    let total = aps.metadata().unwrap().row_groups[0].columns[COL_SUPPKEY]
        .meta_data
        .as_ref()
        .unwrap()
        .num_values as usize;

    println!("== Π.11f — async vs sync decode on lineitem l_suppkey (col {COL_SUPPKEY}) ==");
    println!("data: {}", abs_path.display());
    println!("row_group 0: {total} rows (i64, dict bw=14)");
    println!("warmups: {WARMUPS}, iters: {ITERS}");
    println!();

    // Reusable buffer to amortise allocation cost.
    let mut buf: Vec<i64> = Vec::with_capacity(total);

    // Sync timings.
    for _ in 0..WARMUPS {
        buf.clear();
        read_column_i64_into(&sync_file, 0, COL_SUPPKEY, &mut buf).unwrap();
        black_box(&buf);
    }
    let mut sync_times: Vec<Duration> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        buf.clear();
        let t0 = Instant::now();
        read_column_i64_into(&sync_file, 0, COL_SUPPKEY, &mut buf).unwrap();
        sync_times.push(t0.elapsed());
        black_box(&buf);
    }
    sync_times.sort();
    let sync_med = sync_times[ITERS / 2];

    // Async timings.
    for _ in 0..WARMUPS {
        buf.clear();
        read_column_i64_async_into(&aps, 0, COL_SUPPKEY, &mut buf)
            .await
            .unwrap();
        black_box(&buf);
    }
    let mut async_times: Vec<Duration> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        buf.clear();
        let t0 = Instant::now();
        read_column_i64_async_into(&aps, 0, COL_SUPPKEY, &mut buf)
            .await
            .unwrap();
        async_times.push(t0.elapsed());
        black_box(&buf);
    }
    async_times.sort();
    let async_med = async_times[ITERS / 2];

    let ratio = async_med.as_secs_f64() / sync_med.as_secs_f64();
    let pct = (ratio - 1.0) * 100.0;
    let arrow = if ratio <= 1.05 { "✓" } else { "✗" };

    println!(
        "  sync  read_column_i64_into       : median {}",
        fmt(sync_med)
    );
    println!(
        "  async read_column_i64_async_into : median {}",
        fmt(async_med)
    );
    println!(
        "  async/sync ratio: {:.2}× ({:+.1}%)  {arrow} within 5% acceptance bar",
        ratio, pct
    );
    println!();
    println!(
        "Cloud throughput (S3 etc.) is bandwidth-bound; this bench just\n\
         confirms the async surface doesn't add measurable parser overhead.\n\
         S3 integration bench lives in Π.11e."
    );
}
