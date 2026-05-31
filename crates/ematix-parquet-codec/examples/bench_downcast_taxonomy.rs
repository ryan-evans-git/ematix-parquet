//! DOWNCAST.CONT — full-taxonomy footprint ceiling across every column of
//! a parquet file (default: the ematix-flow SF=100 `lineitem`).
//!
//! For each fixed-width INTEGER column it reads the row-group statistics,
//! computes the narrowest lossless target two ways — ABSOLUTE
//! ([`narrowest_int_target`]) and FRAME-OF-REFERENCE
//! ([`frame_of_reference_target`], offset-from-min) — and reports the
//! decoded in-memory footprint we pay today vs the narrowed footprint.
//!
//! It answers the gate question for reviving flow-side consumption (which
//! REV.13 shelved on the keys-only i64 slice): *if every narrowable column
//! decoded into its narrowest width, how much of `lineitem`'s fixed-width
//! footprint do we reclaim?* Wider than REV.13's key-only test — this also
//! credits DATE(INT32)->i16 and tiny-int->i8.
//!
//! Metadata only — never decodes a data page, so it is instant even at
//! SF=100. `f64`/`Double` columns are excluded from narrowing on purpose
//! (the `f64 -> f32` precision-loss policy); `ByteArray` strings are not
//! fixed-width and are listed for context only.
//!
//! Usage:
//!   PARQUET=/abs/path/lineitem.parquet \
//!     cargo run --release -p ematix-parquet-codec --example bench_downcast_taxonomy

use ematix_parquet_codec::downcast::{frame_of_reference_target, narrowest_int_target, IntTarget};
use ematix_parquet_format::types::ParquetType;
use ematix_parquet_io::ParquetFile;

fn col_name(path: &[&[u8]]) -> String {
    path.iter()
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect::<Vec<_>>()
        .join(".")
}

fn type_label(t: ParquetType) -> &'static str {
    match t {
        ParquetType::Boolean => "BOOL",
        ParquetType::Int32 => "INT32",
        ParquetType::Int64 => "INT64",
        ParquetType::Int96 => "INT96",
        ParquetType::Float => "FLOAT",
        ParquetType::Double => "DOUBLE",
        ParquetType::ByteArray => "BYTES",
        ParquetType::FixedLenByteArray => "FLBA",
    }
}

fn target_label(t: IntTarget) -> &'static str {
    match t {
        IntTarget::I8 => "i8",
        IntTarget::U8 => "u8",
        IntTarget::I16 => "i16",
        IntTarget::U16 => "u16",
        IntTarget::I32 => "i32",
        IntTarget::U32 => "u32",
        IntTarget::I64 => "i64",
    }
}

/// Dense decoded bytes-per-value for a fixed-width physical type (what we
/// pay today). Variable-width types (`ByteArray`/`FLBA`) return `None`.
fn source_width(t: ParquetType) -> Option<usize> {
    match t {
        ParquetType::Boolean => Some(1),
        ParquetType::Int32 | ParquetType::Float => Some(4),
        ParquetType::Int64 | ParquetType::Double => Some(8),
        ParquetType::Int96 => Some(12),
        ParquetType::ByteArray | ParquetType::FixedLenByteArray => None,
    }
}

/// Parquet stats store min/max as little-endian bytes of the physical
/// type: 4 for INT32, 8 for INT64. Widen to i64 for the range decision.
fn le_to_i64(bytes: &[u8]) -> Option<i64> {
    match bytes.len() {
        4 => Some(i32::from_le_bytes(bytes.try_into().ok()?) as i64),
        8 => Some(i64::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

fn main() {
    let path = std::env::var("PARQUET").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/RustroverProjects/ematix-flow/examples/tpch/data/sf100/lineitem.parquet")
    });
    let file = ParquetFile::open(&path).expect("open parquet file");
    let md = file.cached_metadata().expect("metadata");
    let n_rg = md.row_groups.len();
    let n_cols = md
        .row_groups
        .first()
        .map(|rg| rg.columns.len())
        .unwrap_or(0);

    println!("=== downcast taxonomy: footprint ceiling ===");
    println!("file       : {path}");
    println!("row_groups : {n_rg}   columns: {n_cols}\n");
    println!(
        "{:<3} {:<16} {:>6} {:>13} {:>6} {:>6} {:>9} {:>9} {:>6}",
        "#", "column", "type", "rows", "abs", "frame", "src MB", "narr MB", "save"
    );

    let mut tot_src_int = 0u128; // bytes for narrowable INT columns today
    let mut tot_narr_int = 0u128; // bytes after best (frame) narrowing
    let mut tot_decoded = 0u128; // total decoded footprint incl. doubles+strings

    for col in 0..n_cols {
        // Aggregate over row groups: type/name from the first chunk that
        // has them; rows summed; global min = min of per-RG mins, global
        // max = max of per-RG maxes (the single width that fits the whole
        // column — the conservative ceiling vs per-RG narrowing).
        let mut rows: i64 = 0;
        let mut gmin: Option<i64> = None;
        let mut gmax: Option<i64> = None;
        let mut ptype: Option<ParquetType> = None;
        let mut name = String::new();
        let mut ba_uncompressed: i64 = 0;

        for rg in &md.row_groups {
            let Some(cm) = rg.columns.get(col).and_then(|c| c.meta_data.as_ref()) else {
                continue;
            };
            ptype = Some(cm.column_type);
            if name.is_empty() {
                name = col_name(&cm.path_in_schema);
            }
            rows += cm.num_values;
            ba_uncompressed += cm.total_uncompressed_size;
            if let Some(stats) = cm.statistics.as_ref() {
                if let Some(mn) = stats.min_value.or(stats.min).and_then(le_to_i64) {
                    gmin = Some(gmin.map_or(mn, |g: i64| g.min(mn)));
                }
                if let Some(mx) = stats.max_value.or(stats.max).and_then(le_to_i64) {
                    gmax = Some(gmax.map_or(mx, |g: i64| g.max(mx)));
                }
            }
        }
        let Some(ptype) = ptype else { continue };
        let rows_u = rows.max(0) as u128;

        match source_width(ptype) {
            // Narrowable integer column.
            Some(sw) if matches!(ptype, ParquetType::Int32 | ParquetType::Int64) => {
                let src = rows_u * sw as u128;
                tot_decoded += src;
                match (gmin, gmax) {
                    (Some(mn), Some(mx)) if mn <= mx => {
                        let abs = narrowest_int_target(mn, mx);
                        let (_off, frm) = frame_of_reference_target(mn, mx);
                        // Can't "narrow" past the source width (an i32 source
                        // never widens to i64).
                        let nw = frm.width_bytes().min(sw);
                        let narr = rows_u * nw as u128;
                        tot_src_int += src;
                        tot_narr_int += narr;
                        let save = 100.0 * (1.0 - narr as f64 / src.max(1) as f64);
                        println!(
                            "{:<3} {:<16} {:>6} {:>13} {:>6} {:>6} {:>9.1} {:>9.1} {:>5.0}%",
                            col,
                            name,
                            type_label(ptype),
                            rows,
                            target_label(abs),
                            target_label(frm),
                            src as f64 / 1e6,
                            narr as f64 / 1e6,
                            save,
                        );
                    }
                    _ => {
                        tot_src_int += src;
                        tot_narr_int += src; // no stats -> can't prove narrowing
                        println!(
                            "{:<3} {:<16} {:>6} {:>13} {:>6} {:>6} {:>9.1} {:>9.1} {:>6}",
                            col,
                            name,
                            type_label(ptype),
                            rows,
                            "-",
                            "-",
                            src as f64 / 1e6,
                            src as f64 / 1e6,
                            "no-stats",
                        );
                    }
                }
            }
            // Fixed-width but excluded from narrowing (Double/Float/Int96/Bool).
            Some(sw) => {
                let src = rows_u * sw as u128;
                tot_decoded += src;
                println!(
                    "{:<3} {:<16} {:>6} {:>13} {:>6} {:>6} {:>9.1} {:>9.1} {:>6}",
                    col,
                    name,
                    type_label(ptype),
                    rows,
                    "-",
                    "-",
                    src as f64 / 1e6,
                    src as f64 / 1e6,
                    "excl",
                );
            }
            // Variable-width strings — use uncompressed chunk size as a rough
            // decoded-footprint proxy; never a narrowing candidate.
            None => {
                tot_decoded += ba_uncompressed.max(0) as u128;
                println!(
                    "{:<3} {:<16} {:>6} {:>13} {:>6} {:>6} {:>9.1} {:>9} {:>6}",
                    col,
                    name,
                    type_label(ptype),
                    rows,
                    "-",
                    "-",
                    ba_uncompressed as f64 / 1e6,
                    "-",
                    "string",
                );
            }
        }
    }

    let int_save = 100.0 * (tot_src_int - tot_narr_int) as f64 / tot_src_int.max(1) as f64;
    let whole_save = 100.0 * (tot_src_int - tot_narr_int) as f64 / tot_decoded.max(1) as f64;
    println!("\n--- totals (decoded in-memory footprint) ---");
    println!(
        "narrowable INT columns : {:.0} MB -> {:.0} MB   ({:.0}% smaller)",
        tot_src_int as f64 / 1e6,
        tot_narr_int as f64 / 1e6,
        int_save,
    );
    println!(
        "as a share of ALL decoded columns ({:.0} MB incl. doubles+strings) : {:.1}% reclaimed",
        tot_decoded as f64 / 1e6,
        whole_save,
    );
    println!(
        "\nnote: ceiling uses one global width per column; per-row-group frame\n      narrowing can do better. f64 excluded by policy; strings est. from\n      uncompressed chunk size. This is footprint only (decode-cost delta\n      is bench_downcast for i64; REV.14 SIMD addresses the +4% decode tax)."
    );
}
