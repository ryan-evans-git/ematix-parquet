//! End-to-end oracle for the codec crate's first real decoder.
//!
//! Reaches across all three crates:
//!   - `ematix-parquet-io::ParquetFile`   to open lineitem
//!   - `ematix-parquet-io::PageWalker`    to yield the first page
//!   - `decompress_snappy`                to inflate the page body
//!   - `decode_plain_i64`                 to turn bytes into i64s
//!
//! Then asks parquet-rs the same question (give me lineitem rg 0
//! col 0's dictionary page values) and asserts our Vec<i64> matches
//! theirs value-by-value.

use std::fs::File;
use std::path::PathBuf;

use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::plain::decode_plain_i64;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::column::page::Page;
use parquet::file::reader::{FileReader, SerializedFileReader};

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

#[test]
fn lineitem_rg0_col0_dictionary_matches_parquet_rs() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    // ---- Our path: io + codec ------------------------------------------
    let file = ParquetFile::open(&path).expect("open lineitem");
    let md = file.metadata().expect("decode metadata");
    let rg = &md.row_groups[0];
    let col = &rg.columns[0];
    let cm = col.meta_data.as_ref().expect("inline ColumnMetaData");

    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let chunk_bytes = file.read_range(start, length).expect("read chunk");

    let mut walker = PageWalker::new(&chunk_bytes);
    let (first_hdr, first_body) = walker
        .next_page()
        .expect("walk first page")
        .expect("at least one page");
    let dict_hdr = first_hdr
        .dictionary_page_header
        .as_ref()
        .expect("first page should be the dictionary page for l_orderkey");
    let dict_num_values = dict_hdr.num_values as usize;
    eprintln!(
        "dict page header says num_values={dict_num_values}, encoding={:?}",
        dict_hdr.encoding
    );
    eprintln!(
        "  page body sizes: compressed={}, uncompressed={}",
        first_hdr.compressed_page_size, first_hdr.uncompressed_page_size,
    );

    let decompressed = decompress_snappy(first_body).expect("snappy decompress");
    assert_eq!(
        decompressed.len(),
        first_hdr.uncompressed_page_size as usize,
        "decompressed size should equal page header's uncompressed_page_size"
    );

    let ours = decode_plain_i64(&decompressed).expect("PLAIN i64 decode");
    assert_eq!(ours.len(), dict_num_values, "value count");

    // ---- parquet-rs path -----------------------------------------------
    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(0).unwrap();
    let mut pages = rgr.get_column_page_reader(0).unwrap();
    let dict_page = loop {
        let p = pages.get_next_page().unwrap().expect("expected dict page");
        if let Page::DictionaryPage { .. } = &p {
            break p;
        }
    };
    let Page::DictionaryPage {
        buf, num_values, ..
    } = dict_page
    else {
        unreachable!();
    };
    let theirs = decode_plain_i64(&buf).expect("decode parquet-rs dict bytes");
    assert_eq!(num_values as usize, ours.len(), "value count vs parquet-rs");

    // ---- Value-by-value comparison -------------------------------------
    assert_eq!(ours.len(), theirs.len(), "length");
    for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        assert_eq!(a, b, "mismatch at index {i}: ours={a}, theirs={b}");
    }
    eprintln!(
        "PASS: {} dictionary i64 values match parquet-rs byte-for-byte",
        ours.len()
    );
    eprintln!(
        "      first 4: {:?}, last 4: {:?}",
        &ours[..4.min(ours.len())],
        &ours[ours.len().saturating_sub(4)..]
    );
}
