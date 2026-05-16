//! Correctness oracle: decode the same Parquet footer bytes via our
//! `read_file_metadata` and via parquet-rs's `SerializedFileReader`,
//! then assert every field we can compare matches.
//!
//! Sweeps all 8 TPC-H tables at SF=1. Each table is its own test so
//! failures pinpoint which schema shape broke.
//!
//! Data source lookup:
//!   $TPCH_DATA_DIR   — explicit override
//!   ../../ematix-flow/examples/tpch/data/sf1/  — sibling fallback
//!   neither resolved → skip (the suite is a sanity gate, not a CI
//!   dependency).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::read_file_metadata;
use ematix_parquet_format::types::ParquetType;
use parquet::basic::Type as PrType;
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

fn read_footer_bytes(path: &PathBuf) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let n = f.metadata()?.len();
    f.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    f.read_exact(&mut tail)?;
    assert_eq!(&tail[4..], b"PAR1", "missing PAR1 magic at end of {path:?}");
    let footer_len = u32::from_le_bytes(tail[0..4].try_into().unwrap()) as u64;
    let footer_start = n - 8 - footer_len;
    f.seek(SeekFrom::Start(footer_start))?;
    let mut buf = vec![0u8; footer_len as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn map_parquet_type(t: ParquetType) -> PrType {
    match t {
        ParquetType::Boolean => PrType::BOOLEAN,
        ParquetType::Int32 => PrType::INT32,
        ParquetType::Int64 => PrType::INT64,
        ParquetType::Int96 => PrType::INT96,
        ParquetType::Float => PrType::FLOAT,
        ParquetType::Double => PrType::DOUBLE,
        ParquetType::ByteArray => PrType::BYTE_ARRAY,
        ParquetType::FixedLenByteArray => PrType::FIXED_LEN_BYTE_ARRAY,
    }
}

fn check_footer(table: &str) {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP {table}: TPC-H data not found");
        return;
    };
    let path = dir.join(format!("{table}.parquet"));
    if !path.exists() {
        eprintln!("SKIP {table}: {} missing", path.display());
        return;
    }

    let footer = read_footer_bytes(&path).expect("read footer");
    let mut cur = Cursor::new(&footer);
    let our_md = read_file_metadata(&mut cur)
        .unwrap_or_else(|e| panic!("{table}: our decoder failed: {e:?}"));

    let pr_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let pr_md = pr_reader.metadata();
    let pr_fm = pr_md.file_metadata();

    assert_eq!(our_md.version, pr_fm.version(), "{table}: version");
    assert_eq!(our_md.num_rows, pr_fm.num_rows(), "{table}: num_rows");
    let ours_created_by = our_md
        .created_by
        .map(|b| std::str::from_utf8(b).expect("created_by is valid UTF-8"));
    assert_eq!(ours_created_by, pr_fm.created_by(), "{table}: created_by");

    let pr_leaves = pr_fm.schema_descr().num_columns();
    assert_eq!(
        our_md.schema.len(),
        1 + pr_leaves,
        "{table}: schema flat size should be 1 root + {pr_leaves} leaves"
    );
    assert_eq!(
        our_md.schema[0].num_children,
        Some(pr_leaves as i32),
        "{table}: root num_children"
    );

    assert_eq!(
        our_md.row_groups.len(),
        pr_md.num_row_groups(),
        "{table}: num row groups"
    );

    for (i, (ours, theirs)) in our_md.row_groups.iter().zip(pr_md.row_groups()).enumerate() {
        assert_eq!(ours.num_rows, theirs.num_rows(), "{table} rg {i} num_rows");
        assert_eq!(
            ours.total_byte_size,
            theirs.total_byte_size(),
            "{table} rg {i} total_byte_size"
        );
        assert_eq!(
            ours.columns.len(),
            theirs.num_columns(),
            "{table} rg {i} column count"
        );

        for (j, (our_col, their_col)) in ours.columns.iter().zip(theirs.columns()).enumerate() {
            let our_meta = our_col
                .meta_data
                .as_ref()
                .unwrap_or_else(|| panic!("{table} rg {i} col {j} missing inline ColumnMetaData"));

            assert_eq!(
                map_parquet_type(our_meta.column_type),
                their_col.column_type(),
                "{table} rg {i} col {j} physical type"
            );

            // Codec: parquet-rs uses level-parametric variants; we
            // compare the kind name only.
            let theirs_codec = format!("{:?}", their_col.compression());
            let ours_codec = format!("{:?}", our_meta.codec);
            let theirs_kind = theirs_codec
                .split(|c: char| !c.is_ascii_alphanumeric())
                .next()
                .unwrap();
            let ours_kind = ours_codec
                .split(|c: char| !c.is_ascii_alphanumeric())
                .next()
                .unwrap();
            assert!(
                theirs_kind.eq_ignore_ascii_case(ours_kind),
                "{table} rg {i} col {j} codec: ours={ours_codec} theirs={theirs_codec}"
            );

            assert_eq!(
                our_meta.num_values,
                their_col.num_values(),
                "{table} rg {i} col {j} num_values"
            );
            assert_eq!(
                our_meta.total_compressed_size,
                their_col.compressed_size(),
                "{table} rg {i} col {j} total_compressed_size"
            );
            assert_eq!(
                our_meta.total_uncompressed_size,
                their_col.uncompressed_size(),
                "{table} rg {i} col {j} total_uncompressed_size"
            );
        }
    }

    eprintln!(
        "PASS {table}: {} row groups × {} columns × {} rows",
        our_md.row_groups.len(),
        our_md
            .row_groups
            .first()
            .map(|rg| rg.columns.len())
            .unwrap_or(0),
        our_md.num_rows,
    );
}

#[test]
fn tpch_lineitem() {
    check_footer("lineitem");
}
#[test]
fn tpch_orders() {
    check_footer("orders");
}
#[test]
fn tpch_customer() {
    check_footer("customer");
}
#[test]
fn tpch_part() {
    check_footer("part");
}
#[test]
fn tpch_partsupp() {
    check_footer("partsupp");
}
#[test]
fn tpch_supplier() {
    check_footer("supplier");
}
#[test]
fn tpch_nation() {
    check_footer("nation");
}
#[test]
fn tpch_region() {
    check_footer("region");
}
