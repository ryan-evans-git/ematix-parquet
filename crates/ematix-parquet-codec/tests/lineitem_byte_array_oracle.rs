//! Decode lineitem's BYTE_ARRAY (string) columns and compare value-
//! by-value against parquet-rs.
//!
//! Two columns covered:
//!   - l_returnflag (col 8) — exactly 3 distinct values ('A','N','R'),
//!     so every data page should be dict-encoded.
//!   - l_comment (col 15) — high-cardinality free text; the writer
//!     likely starts dict-encoded and falls back to PLAIN.

use std::fs::File;
use std::path::PathBuf;

use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::decode_rle_dictionary_indices;
use ematix_parquet_codec::plain::{decode_plain_byte_array, decode_plain_byte_array_n};
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::column::reader::ColumnReader;
use parquet::data_type::ByteArray;
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

fn read_chunk_bytes(file: &ParquetFile, rg_idx: usize, col_idx: usize) -> (Vec<u8>, i64) {
    let md = file.metadata().expect("metadata");
    let rg = &md.row_groups[rg_idx];
    let col = &rg.columns[col_idx];
    let cm = col.meta_data.as_ref().expect("inline col meta");
    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let bytes = file.read_range(start, length).expect("read chunk");
    (bytes, cm.num_values)
}

fn decode_byte_array_column(
    file: &ParquetFile,
    rg_idx: usize,
    col_idx: usize,
) -> (Vec<Vec<u8>>, usize, usize) {
    let (chunk_bytes, total_values) = read_chunk_bytes(file, rg_idx, col_idx);
    let total_values = total_values as usize;
    let mut walker = PageWalker::new(&chunk_bytes);

    let (first_hdr, first_body) = walker.next_page().unwrap().unwrap();
    let dict_decompressed = decompress_snappy(first_body).expect("dict snappy");
    let dict: Vec<Vec<u8>> = if first_hdr.dictionary_page_header.is_some() {
        decode_plain_byte_array(&dict_decompressed)
            .expect("dict PLAIN byte_array")
            .into_iter()
            .map(|s| s.to_vec())
            .collect()
    } else {
        // First page is itself a data page; the column has no dict.
        // For lineitem we expect a dict on every column, so this is a
        // defensive panic.
        panic!("expected dictionary page for byte_array column")
    };

    let mut out: Vec<Vec<u8>> = Vec::with_capacity(total_values);
    let mut dict_pages = 0;
    let mut plain_pages = 0;
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header.as_ref().expect("v1 data page");
        let n = dph.num_values as usize;
        let decompressed = decompress_snappy(body).unwrap();
        match dph.encoding {
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices = decode_rle_dictionary_indices(&decompressed, n).unwrap();
                // lookup_dict is generic over T: Copy; Vec<u8> is not
                // Copy, so we look up by index and clone.
                for &idx in &indices {
                    out.push(dict[idx as usize].clone());
                }
                dict_pages += 1;
            }
            Encoding::Plain => {
                let values = decode_plain_byte_array_n(&decompressed, n).unwrap();
                for v in values {
                    out.push(v.to_vec());
                }
                plain_pages += 1;
            }
            other => panic!("unhandled byte_array data page encoding {other:?}"),
        }
        if out.len() >= total_values {
            break;
        }
    }
    (out, dict_pages, plain_pages)
}

fn parquet_rs_read_byte_array(
    path: &PathBuf,
    rg_idx: usize,
    col_idx: usize,
    n: usize,
) -> Vec<Vec<u8>> {
    let pr_reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(rg_idx).unwrap();
    let mut typed = match rgr.get_column_reader(col_idx).unwrap() {
        ColumnReader::ByteArrayColumnReader(t) => t,
        _ => panic!("expected ByteArrayColumnReader"),
    };
    let mut out: Vec<ByteArray> = Vec::with_capacity(n);
    typed.read_records(n, None, None, &mut out).unwrap();
    out.into_iter().map(|ba| ba.data().to_vec()).collect()
}

#[test]
fn lineitem_rg0_l_returnflag_matches_parquet_rs() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open");
    let (ours, dict_pgs, plain_pgs) = decode_byte_array_column(&file, 0, 8);
    let theirs = parquet_rs_read_byte_array(&path, 0, 8, ours.len());
    assert_eq!(ours.len(), theirs.len());
    assert_eq!(ours, theirs);

    // Sanity: l_returnflag only takes 'A', 'N', 'R'.
    let mut distinct: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    for v in &ours {
        distinct.insert(v.clone());
    }
    eprintln!(
        "PASS l_returnflag: {} byte_array values match parquet-rs \
         ({} distinct, {dict_pgs} dict pages, {plain_pgs} plain pages, \
         first 5: {:?})",
        ours.len(),
        distinct.len(),
        ours.iter().take(5).map(|v| String::from_utf8_lossy(v).into_owned()).collect::<Vec<_>>()
    );
    assert_eq!(distinct.len(), 3, "l_returnflag should have exactly 3 distinct values");
}

#[test]
fn lineitem_rg0_l_comment_matches_parquet_rs() {
    // Higher-cardinality column — likely dict + plain mix like
    // l_orderkey. Validates both paths for BYTE_ARRAY.
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    let file = ParquetFile::open(&path).expect("open");
    let (ours, dict_pgs, plain_pgs) = decode_byte_array_column(&file, 0, 15);
    let theirs = parquet_rs_read_byte_array(&path, 0, 15, ours.len());
    assert_eq!(ours.len(), theirs.len());

    // Avoid printing 1M strings if they don't match — find first
    // divergence instead.
    if ours == theirs {
        let avg_len: usize = ours.iter().map(|v| v.len()).sum::<usize>() / ours.len();
        eprintln!(
            "PASS l_comment: {} byte_array values match parquet-rs \
             ({dict_pgs} dict + {plain_pgs} plain pages, avg len {avg_len}, \
             first: {:?})",
            ours.len(),
            String::from_utf8_lossy(&ours[0])
        );
        return;
    }
    for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        if a != b {
            panic!(
                "value mismatch at index {i}: ours={:?}, theirs={:?}",
                String::from_utf8_lossy(a),
                String::from_utf8_lossy(b)
            );
        }
    }
}
