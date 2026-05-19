use ematix_parquet_io::ParquetFile;
fn main() {
    let base = "/Users/ryanevans/RustroverProjects/ematix-flow/examples/tpch/data/sf1";
    let mut entries: Vec<_> = std::fs::read_dir(base)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        if p.extension().and_then(|s| s.to_str()) == Some("parquet") {
            let f = ParquetFile::open(&p).unwrap();
            let md = f.metadata().unwrap();
            let total_rows: i64 = md.row_groups.iter().map(|r| r.num_rows).sum();
            println!(
                "{:20}  rows={:>10}  rgs={}",
                p.file_name().unwrap().to_string_lossy(),
                total_rows,
                md.row_groups.len()
            );
        }
    }
}
