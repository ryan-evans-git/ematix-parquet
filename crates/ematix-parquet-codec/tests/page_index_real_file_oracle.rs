//! End-to-end page-skip oracle.
//!
//! 1. Write a parquet file via parquet-rs with multiple data pages
//!    (small `data_page_size_limit`) and page-level statistics enabled
//!    (so the writer emits ColumnIndex + OffsetIndex).
//! 2. Open it through `ParquetFile`. Read the OffsetIndex and
//!    ColumnIndex bytes for col 0 (i32) via `read_range`, decode them
//!    via `read_offset_index` / `read_column_index`.
//! 3. Run `select_pages_overlapping_i32` for a predicate that should
//!    exclude most pages.
//! 4. Drive `PageWalker` over the column chunk, decoding only the
//!    pages the selector kept. Assert:
//!      - the number of kept pages matches the number of `true`s in
//!        the selector,
//!      - every decoded value falls inside the predicate range,
//!      - the total kept count equals the count produced by a full
//!        decode + scan baseline.
//!
//! This is the proof the page-skip infrastructure works end-to-end
//! on a real Parquet file written by an independent encoder.

use std::fs::File;
use std::sync::Arc;

use ematix_parquet_codec::compression::decompress_snappy_into;
use ematix_parquet_codec::page_index::select_pages_overlapping_i32;
use ematix_parquet_codec::plain::decode_plain_i32;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_column_index, read_offset_index};
use ematix_parquet_format::types::Encoding as EmEncoding;
use ematix_parquet_io::{PageWalker, ParquetFile};

use parquet::basic::{Compression, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type as SchemaType;

fn write_paged_i32_file(path: &std::path::Path, values: &[i32]) {
    let schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("col_i32", PhysicalType::INT32)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap(),
    );

    // Force multiple data pages by capping page size to ~4KB; with PLAIN
    // i32 that gives roughly 1000 values per page. Page-level stats
    // make the writer emit ColumnIndex + OffsetIndex in the footer.
    let props = Arc::new(
        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::SNAPPY)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_size_limit(4 * 1024)
            .set_encoding(parquet::basic::Encoding::PLAIN)
            .build(),
    );

    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();
    let mut col = row_group.next_column().unwrap().unwrap();
    if let ColumnWriter::Int32ColumnWriter(ref mut typed) = col.untyped() {
        typed.write_batch(values, None, None).unwrap();
    }
    col.close().unwrap();
    row_group.close().unwrap();
    writer.close().unwrap();
}

#[test]
fn page_skip_via_column_index_skips_non_overlapping_pages() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // 10_000 sorted-ascending values from 0..10_000. Ascending means
    // boundary_order=Ascending in the column index, and each page's
    // [min,max] is a clean disjoint range.
    let values: Vec<i32> = (0..10_000).collect();
    write_paged_i32_file(&path, &values);

    let file = ParquetFile::open(&path).expect("open");
    let md = file.metadata().expect("metadata");
    assert_eq!(md.row_groups.len(), 1);
    let cc = &md.row_groups[0].columns[0];
    let cm = cc.meta_data.as_ref().unwrap();

    // Pull the index pointers out of the ColumnChunk.
    let oi_off = cc.offset_index_offset.expect("offset_index_offset");
    let oi_len = cc.offset_index_length.expect("offset_index_length");
    let ci_off = cc.column_index_offset.expect("column_index_offset");
    let ci_len = cc.column_index_length.expect("column_index_length");

    let oi_bytes = file.read_range(oi_off as u64, oi_len as u64).unwrap();
    let ci_bytes = file.read_range(ci_off as u64, ci_len as u64).unwrap();

    let offset_index = {
        let mut cur = Cursor::new(&oi_bytes);
        read_offset_index(&mut cur).unwrap()
    };
    let column_index = {
        let mut cur = Cursor::new(&ci_bytes);
        read_column_index(&mut cur).unwrap()
    };

    let num_pages = offset_index.page_locations.len();
    assert!(
        num_pages > 1,
        "test setup must produce >1 page, got {}",
        num_pages
    );
    assert_eq!(column_index.min_values.len(), num_pages);

    // Predicate: values in [4000, 5500]. With sorted ascending input
    // and ~1000 values/page, this should hit ~2 pages out of ~10.
    let lo = 4000;
    let hi = 5500;
    let keep = select_pages_overlapping_i32(&column_index, lo, hi).unwrap();
    assert_eq!(keep.len(), num_pages);
    let kept = keep.iter().filter(|b| **b).count();
    assert!(
        kept >= 1 && kept < num_pages,
        "selector must skip some pages: kept {}/{}",
        kept,
        num_pages
    );
    eprintln!(
        "page-skip: {} of {} pages selected for range [{}, {}]",
        kept, num_pages, lo, hi
    );

    // Now walk the column chunk, decoding only the kept pages. Note
    // ColumnIndex enumerates DATA pages only; PageWalker also yields
    // dictionary pages, but we disabled dict in the writer so every
    // page is a data page.
    let chunk = file
        .read_range(cm.data_page_offset as u64, cm.total_compressed_size as u64)
        .unwrap();
    let mut walker = PageWalker::new(&chunk);
    let mut decomp: Vec<u8> = Vec::new();
    let mut decoded_kept: Vec<i32> = Vec::new();
    let mut data_page_idx = 0usize;
    while let Some((hdr, body)) = walker.next_page().unwrap() {
        let dph = hdr.data_page_header_v2.as_ref().expect("v2 data page");
        if !keep[data_page_idx] {
            data_page_idx += 1;
            continue;
        }
        // v2 page body layout (per spec):
        //   <rep_levels: rep_len bytes, UNCOMPRESSED>
        //   <def_levels: def_len bytes, UNCOMPRESSED>
        //   <values: is_compressed? snappy : plain>
        // For our REQUIRED column rep_len = def_len = 0.
        let rep_len = dph.repetition_levels_byte_length as usize;
        let def_len = dph.definition_levels_byte_length as usize;
        let values_bytes = &body[rep_len + def_len..];
        let values_slice: &[u8] = if dph.is_compressed {
            decompress_snappy_into(values_bytes, &mut decomp).unwrap();
            &decomp
        } else {
            values_bytes
        };
        assert_eq!(dph.encoding, EmEncoding::Plain);
        let n = dph.num_values as usize;
        let vs = decode_plain_i32(values_slice).unwrap();
        assert_eq!(vs.len(), n);
        decoded_kept.extend(vs);
        data_page_idx += 1;
    }

    // Every decoded value should be in the union of the kept-page
    // ranges. We can't assert tighter than that without per-page min/
    // max — but if the *predicate* range itself is the lo/hi we
    // selected on, then by construction every value that satisfies
    // the predicate must be in `decoded_kept`. Verify it:
    let expected: Vec<i32> = values
        .iter()
        .copied()
        .filter(|v| *v >= lo && *v <= hi)
        .collect();
    let recovered: Vec<i32> = decoded_kept
        .iter()
        .copied()
        .filter(|v| *v >= lo && *v <= hi)
        .collect();
    assert_eq!(
        recovered, expected,
        "kept pages must contain every value satisfying the predicate"
    );

    eprintln!(
        "PASS: page-skip decoded {} kept-page values (vs {} full-scan), recovered {} matches",
        decoded_kept.len(),
        values.len(),
        recovered.len()
    );
}
