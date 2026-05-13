//! Probe the actual bit_widths used by RLE_DICTIONARY data pages in
//! a real TPC-H lineitem file. Tells us which widths the SIMD work
//! should target.
//!
//! Usage:
//!   cargo run --release --example probe_bitwidths -- <path/to/lineitem.parquet>

use std::collections::BTreeMap;

use ematix_parquet_codec::compression::decompress_snappy_into;
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/tpch/data/sf1/lineitem.parquet".into());
    let path_buf = std::path::PathBuf::from(&path);
    let path_buf = if path_buf.is_absolute() {
        path_buf
    } else {
        std::env::current_dir().unwrap().join(&path_buf)
    };
    println!("probing {}", path_buf.display());
    let file = ParquetFile::open(&path_buf).expect("open");
    let md = file.metadata().expect("metadata");

    let schema_columns: Vec<String> = md
        .schema
        .iter()
        .skip(1)
        .map(|e| String::from_utf8_lossy(e.name).into_owned())
        .collect();

    // Per (col_name, bit_width) → total values
    let mut counts: BTreeMap<(String, u8), usize> = BTreeMap::new();

    for (rg_idx, rg) in md.row_groups.iter().enumerate() {
        for (col_idx, col) in rg.columns.iter().enumerate() {
            let cm = col.meta_data.as_ref().unwrap();
            let col_name = schema_columns
                .get(col_idx)
                .cloned()
                .unwrap_or_else(|| format!("col_{col_idx}"));
            let start = cm
                .dictionary_page_offset
                .unwrap_or(cm.data_page_offset) as u64;
            let length = cm.total_compressed_size as u64;
            let chunk = file.read_range(start, length).unwrap();
            let mut walker = PageWalker::new(&chunk);
            let mut decomp: Vec<u8> = Vec::new();
            let mut seen_values = 0usize;
            while let Some((hdr, body)) = walker.next_page().unwrap() {
                if hdr.dictionary_page_header.is_some() {
                    continue;
                }
                let dph = match hdr.data_page_header.as_ref() {
                    Some(h) => h,
                    None => continue, // v2 not handled here
                };
                let n = dph.num_values as usize;
                seen_values += n;
                if matches!(
                    dph.encoding,
                    Encoding::RleDictionary | Encoding::PlainDictionary
                ) {
                    // RLE_DICTIONARY data page body: first byte is the
                    // bit_width of indices, then the RLE/bit-packed
                    // stream. We need to decompress first.
                    decompress_snappy_into(body, &mut decomp).unwrap();
                    if decomp.is_empty() {
                        continue;
                    }
                    let bw = decomp[0];
                    *counts.entry((col_name.clone(), bw)).or_insert(0) += n;
                }
                if seen_values >= cm.num_values as usize {
                    break;
                }
            }
            let _ = rg_idx;
        }
    }

    println!();
    println!("==> dictionary-page bit_widths observed (RG-aggregated)");
    let mut by_col: BTreeMap<String, Vec<(u8, usize)>> = BTreeMap::new();
    for ((col, bw), n) in counts {
        by_col.entry(col).or_default().push((bw, n));
    }
    for (col, mut widths) in by_col {
        widths.sort_by(|a, b| b.1.cmp(&a.1));
        let total: usize = widths.iter().map(|(_, n)| *n).sum();
        let parts: Vec<String> = widths
            .iter()
            .map(|(bw, n)| {
                format!(
                    "bw={}: {} vals ({:.0}%)",
                    bw,
                    n,
                    100.0 * *n as f64 / total as f64
                )
            })
            .collect();
        println!("  {col:<16} {}", parts.join("  |  "));
    }
}
