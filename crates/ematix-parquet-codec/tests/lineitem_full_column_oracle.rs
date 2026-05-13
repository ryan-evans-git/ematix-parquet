//! The big oracle: decode every value of `lineitem.l_orderkey` (1.05M
//! INT64 values, RG 0) through our full stack and compare value-by-
//! value to parquet-rs's typed column reader.
//!
//! Stack covered:
//!   - ParquetFile::open
//!   - PageWalker (dict page + N data pages)
//!   - decompress_snappy
//!   - decode_plain_i64 (dictionary values)
//!   - decode_rle_dictionary_indices (per data page)
//!   - lookup_dict_i64 (index → value)

use std::fs::File;
use std::path::PathBuf;

use ematix_parquet_codec::compression::decompress_snappy;
use ematix_parquet_codec::dict::{decode_rle_dictionary_indices, lookup_dict};
use ematix_parquet_codec::plain::decode_plain_i64;
use ematix_parquet_format::types::Encoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::column::reader::ColumnReader;
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
fn full_rg0_orderkey_column_matches_parquet_rs() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: TPC-H data not found");
        return;
    };
    let path = dir.join("lineitem.parquet");
    if !path.exists() {
        eprintln!("SKIP: {} missing", path.display());
        return;
    }

    // ---- Our path ------------------------------------------------------
    let file = ParquetFile::open(&path).expect("open lineitem");
    let md = file.metadata().expect("metadata");
    let rg = &md.row_groups[0];
    let col = &rg.columns[0];
    let cm = col.meta_data.as_ref().expect("inline col meta");
    let total_values = cm.num_values as usize;

    let start = cm
        .dictionary_page_offset
        .filter(|&d| d < cm.data_page_offset)
        .unwrap_or(cm.data_page_offset) as u64;
    let length = cm.total_compressed_size as u64;
    let chunk_bytes = file.read_range(start, length).expect("read chunk");

    let mut walker = PageWalker::new(&chunk_bytes);

    // First page: dictionary.
    let (dict_hdr, dict_body) = walker.next_page().unwrap().unwrap();
    assert!(
        dict_hdr.dictionary_page_header.is_some(),
        "first page should be a dictionary page"
    );
    let dict_decompressed = decompress_snappy(dict_body).expect("dict snappy");
    let dict = decode_plain_i64(&dict_decompressed).expect("dict PLAIN i64");
    eprintln!("dict: {} entries", dict.len());

let mut ours: Vec<i64> = Vec::with_capacity(total_values);
    let mut dict_data_pages = 0;
    let mut plain_data_pages = 0;
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr
            .data_page_header
            .as_ref()
            .expect("expected v1 data page");
        let n_values = dph.num_values as usize;
        let decompressed = decompress_snappy(body).expect("data snappy");

        match dph.encoding {
            // Parquet writers usually use the modern code (8) but
            // also accept the legacy alias (2). Both mean dict-indices.
            Encoding::RleDictionary | Encoding::PlainDictionary => {
                let indices =
                    decode_rle_dictionary_indices(&decompressed, n_values).expect("rle indices");
                let values = lookup_dict(&dict, &indices).expect("dict lookup");
                ours.extend(values);
                dict_data_pages += 1;
            }
            // Writers fall back to PLAIN once the dictionary grows
            // beyond their threshold; subsequent pages encode raw
            // values directly.
            Encoding::Plain => {
                let values = decode_plain_i64(&decompressed).expect("plain i64");
                ours.extend(values);
                plain_data_pages += 1;
            }
            other => panic!("unhandled data page encoding: {other:?}"),
        }
        if ours.len() >= total_values {
            break;
        }
    }
    eprintln!(
        "decoded {} values across {} dict-encoded + {} plain-encoded data pages",
        ours.len(),
        dict_data_pages,
        plain_data_pages
    );
    assert_eq!(ours.len(), total_values, "total value count");

    // ---- parquet-rs path -----------------------------------------------
    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let rgr = pr_reader.get_row_group(0).unwrap();
    let mut typed = match rgr.get_column_reader(0).unwrap() {
        ColumnReader::Int64ColumnReader(t) => t,
        _ => panic!("expected Int64ColumnReader"),
    };
    let mut theirs: Vec<i64> = Vec::with_capacity(total_values);
    let (n_read, _, _) = typed
        .read_records(total_values, None, None, &mut theirs)
        .expect("parquet-rs read");
    assert_eq!(n_read, total_values, "parquet-rs read count");
    assert_eq!(theirs.len(), total_values, "parquet-rs vec length");

    // ---- Value-by-value -----------------------------------------------
    assert_eq!(ours.len(), theirs.len());
    // Cheap path first: equality.
    if ours == theirs {
        eprintln!(
            "PASS: {} l_orderkey i64 values match parquet-rs byte-for-byte",
            ours.len()
        );
        eprintln!(
            "      first 5: {:?}",
            &ours[..5.min(ours.len())]
        );
        eprintln!(
            "      last 5:  {:?}",
            &ours[ours.len().saturating_sub(5)..]
        );
        return;
    }
    // Slow path: find first divergence and report.
    for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        if a != b {
            panic!(
                "value mismatch at index {i}: ours={a}, theirs={b} \
                 (surrounding: ours={:?}, theirs={:?})",
                &ours[i.saturating_sub(2)..(i + 3).min(ours.len())],
                &theirs[i.saturating_sub(2)..(i + 3).min(theirs.len())]
            );
        }
    }
}
