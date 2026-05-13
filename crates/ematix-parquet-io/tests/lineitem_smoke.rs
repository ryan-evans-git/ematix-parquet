//! End-to-end smoke test: open real `lineitem.parquet`, get its
//! metadata via our io crate, walk the pages of one column chunk,
//! and compare the result against parquet-rs.
//!
//! This is the io-layer analog of the format crate's oracle test.
//! It validates two things the format-crate oracle doesn't:
//!   1. ParquetFile::open correctly slices out the footer (any byte
//!      offset error would corrupt the FileMetaData decode).
//!   2. PageWalker correctly threads the header → body → next-header
//!      cursor across a real column chunk (~ tens of MB).

use std::fs::File;
use std::path::PathBuf;

use ematix_parquet_io::{PageWalker, ParquetFile};
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
fn open_lineitem_and_walk_first_column_pages() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open lineitem");
    eprintln!(
        "opened {} ({} bytes, footer at {})",
        path.display(),
        file.file_size(),
        file.footer_offset()
    );

    let md = file.metadata().expect("decode metadata");
    assert_eq!(md.num_rows, 6_001_215);
    assert_eq!(md.row_groups.len(), 6);

    // Pick row group 0, column 0 (l_orderkey, INT64).
    let rg = &md.row_groups[0];
    let col = &rg.columns[0];
    let cm = col
        .meta_data
        .as_ref()
        .expect("inline ColumnMetaData expected");

    // Compute the byte range for the chunk. The first byte of the
    // chunk is the dict page offset if a dict page is present and
    // precedes the data, else the data page offset.
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    eprintln!(
        "chunk [{}, {}+{}) — {} values",
        start, start, length, cm.num_values
    );

    let chunk_bytes = file.read_range(start, length).expect("read chunk");

    // Walk pages, counting and summing num_values across data pages.
    let mut walker = PageWalker::new(&chunk_bytes);
    let mut total_data_values: i64 = 0;
    let mut data_pages = 0;
    let mut dict_pages = 0;
    while let Some((hdr, _body)) = walker.next_page().expect("walk page") {
        if let Some(dph) = &hdr.data_page_header {
            data_pages += 1;
            total_data_values += dph.num_values as i64;
        } else if let Some(v2) = &hdr.data_page_header_v2 {
            data_pages += 1;
            total_data_values += v2.num_values as i64;
        } else if hdr.dictionary_page_header.is_some() {
            dict_pages += 1;
        }
        if total_data_values >= cm.num_values {
            break;
        }
    }

    eprintln!(
        "  → {data_pages} data pages, {dict_pages} dictionary pages, {total_data_values} values"
    );
    assert_eq!(
        total_data_values, cm.num_values,
        "sum of page num_values must equal ColumnMetaData.num_values"
    );

    // Cross-check against parquet-rs: same data_page count.
    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(0).unwrap();
    let mut pages = rgr.get_column_page_reader(0).unwrap();
    let mut pr_data_pages = 0;
    let mut pr_dict_pages = 0;
    while let Some(page) = pages.get_next_page().unwrap() {
        use parquet::column::page::Page;
        match page {
            Page::DataPage { .. } | Page::DataPageV2 { .. } => pr_data_pages += 1,
            Page::DictionaryPage { .. } => pr_dict_pages += 1,
        }
    }
    eprintln!("  parquet-rs sees {pr_data_pages} data pages, {pr_dict_pages} dict pages");
    assert_eq!(data_pages, pr_data_pages, "data page count vs parquet-rs");
    assert_eq!(dict_pages, pr_dict_pages, "dict page count vs parquet-rs");
}
