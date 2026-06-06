//! PV.M.6 Phase-0 — dump per-column encoding + codec for a parquet file, so we
//! know where a const-generic BIT-unpacker even applies (RLE/dict-index packing)
//! vs PLAIN f64 (Snappy-decompress + memcpy, no bit-unpacking). Run on SF=100
//! lineitem to scope the decode-kernel rewrite before committing to it.
//!
//! Usage: cargo run --release -p ematix-parquet-codec --example dump_encodings -- <path>

use ematix_parquet_io::ParquetFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/ryanevans/RustroverProjects/ematix-flow/examples/tpch/data/sf100/lineitem.parquet"
            .to_string()
    });
    let file = ParquetFile::open(&path)?;
    let md = file.metadata()?;
    let rg0 = &md.row_groups[0];
    println!("file: {path}");
    println!("row_groups={}  cols={}", md.row_groups.len(), rg0.columns.len());
    println!("{:<18} {:<10} {:>12} {:>12} {:<8} {}", "column", "codec", "comp_MB(rg0)", "uncomp_MB", "dict?", "encodings");
    let mut tot_comp = 0u64;
    let mut tot_uncomp = 0u64;
    for c in &rg0.columns {
        let cm = match c.meta_data.as_ref() {
            Some(m) => m,
            None => continue,
        };
        let name = cm
            .path_in_schema
            .iter()
            .map(|p| std::str::from_utf8(p).unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(".");
        let codec = format!("{:?}", cm.codec);
        let comp = cm.total_compressed_size as f64 / 1e6;
        let uncomp = cm.total_uncompressed_size as f64 / 1e6;
        let has_dict = cm.dictionary_page_offset.is_some();
        let encs: Vec<String> = cm.encodings.iter().map(|e| format!("{e:?}")).collect();
        tot_comp += cm.total_compressed_size as u64;
        tot_uncomp += cm.total_uncompressed_size as u64;
        println!(
            "{:<18} {:<10} {:>12.2} {:>12.2} {:<8} {}",
            name, codec, comp, uncomp, has_dict, encs.join(",")
        );
    }
    println!(
        "\nrg0 totals: comp={:.1} MB  uncomp={:.1} MB  ratio={:.2}",
        tot_comp as f64 / 1e6,
        tot_uncomp as f64 / 1e6,
        tot_comp as f64 / tot_uncomp as f64
    );
    println!("\nDICT/RLE columns → bit-packed indices (const-generic unpacker applies).");
    println!("PLAIN f64/i64 columns → Snappy-decompress + memcpy (unpacker does NOT apply).");
    Ok(())
}
