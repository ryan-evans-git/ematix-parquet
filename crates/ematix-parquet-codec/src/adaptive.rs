//! Π.14a — adaptive predicate dispatch primitives.
//!
//! Two static paths exist today for dict-encoded numeric / byte
//! columns under a predicate:
//!
//! - **Fused** (Π.9b): `decode_rle_dictionary_predicate_bitmap` —
//!   builds a packed bitmap directly from indices + a precomputed
//!   `dict_mask`. No materialised values. Wins at low selectivity:
//!   downstream consumers receive a bitmap and don't pay the
//!   gather.
//! - **Materialised**: decode indices → gather dict values → filter.
//!   Wins at high selectivity: most rows pass anyway, the
//!   bitmap-pack adds overhead, and most consumers want values.
//!
//! Π.14 picks between them per chunk based on selectivity observed
//! from the first N pages.
//!
//! This module ships the **primitive**:
//!
//! - `Dispatch` — which path was chosen.
//! - `PageProbe` — per-page rows_in / rows_passed.
//! - `AdaptiveDictPredicate` — config (dict mask, threshold, probe
//!   page budget).
//! - `popcount_bitmap_prefix` — counts the set bits in the first
//!   `num_bits` of a packed bitmap, ignoring any padding tail.
//! - `probe_page_fused` — runs the fused kernel on one data-page
//!   body, returns `(emitted_bytes, PageProbe)`.
//!
//! The multi-page chunk runner that consumes these to make a
//! per-chunk dispatch decision lands in Π.14b.

use crate::dict::{decode_rle_dictionary_into, decode_rle_dictionary_predicate_bitmap};
use crate::error::Result;

/// Which decode path the adaptive runner chose for a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Predicate-fused: bitmap output, no values materialised.
    Fused,
    /// Decode-then-filter: values materialised, filter applied.
    Materialized,
}

/// Selectivity reading from one probed page. `rows_passed /
/// rows_in` gives the page's selectivity; the chunk runner
/// aggregates across probe pages before deciding.
#[derive(Debug, Clone, Copy)]
pub struct PageProbe {
    /// `num_values` from the data-page header.
    pub rows_in: usize,
    /// Set bits in the emitted bitmap (within the first `rows_in`
    /// bits — padding excluded).
    pub rows_passed: usize,
}

impl PageProbe {
    /// Selectivity as a fraction in `[0.0, 1.0]`. Returns 0.0 for an
    /// empty page (no division-by-zero panic).
    pub fn selectivity(&self) -> f32 {
        if self.rows_in == 0 {
            0.0
        } else {
            self.rows_passed as f32 / self.rows_in as f32
        }
    }
}

/// Config for the adaptive dispatcher.
///
/// `threshold` is the selectivity fraction above which the chunk
/// runner switches the remaining pages from fused to materialised.
/// The Π.14c bench sweep tunes this against lineitem-shape data;
/// the current default is a placeholder until that sweep lands.
///
/// `probe_pages` is how many pages to run through the fused path
/// before making the dispatch call. 2 is a reasonable default —
/// one page can be unrepresentative (e.g. a sorted column where
/// the first page is all-out-of-range); two gives some signal
/// without burning too much of the chunk on probing.
#[derive(Debug, Clone)]
pub struct AdaptiveDictPredicate {
    /// Per-dict-entry predicate result. `dict_mask.len()` must be
    /// `≥ 1 << bit_width` for the chunk's data pages. Same shape
    /// `decode_rle_dictionary_predicate_bitmap` expects.
    pub dict_mask: Vec<u8>,
    /// Switch to materialised if observed selectivity exceeds this
    /// after the probe phase. Range `[0.0, 1.0]`.
    pub threshold: f32,
    /// Number of pages to probe with the fused path before
    /// dispatching the rest of the chunk.
    pub probe_pages: usize,
}

impl AdaptiveDictPredicate {
    /// Benchmark-derived crossover between `Fused` (with the
    /// downstream bitmap-then-gather step a values-consumer pays)
    /// and `Materialized` (which avoids the bitmap step entirely).
    ///
    /// Measured on TPC-H SF1 lineitem `l_shipdate` (1M rows, 52
    /// pages, dict_len=2526, bw=12), median of 51 release-mode
    /// iterations on aarch64:
    ///
    /// ```text
    /// selectivity   fused+gather   materialised   winner
    ///       0.1%        0.37 ms       0.99 ms     fused+gather  (2.7×)
    ///       1.0%        0.53 ms       1.07 ms     fused+gather  (2.0×)
    ///       5.0%        1.21 ms       1.29 ms     fused+gather  (1.07×)
    ///      10.0%        1.72 ms       1.59 ms     materialised  (1.08×)
    ///      20.0%        2.17 ms       2.21 ms     ~tied
    ///      50.0%        3.69 ms       4.30 ms     fused+gather  (1.16×)
    ///      90.0%        2.39 ms       2.29 ms     materialised  (1.04×)
    /// ```
    ///
    /// At 10% selectivity the two paths cross; below that the
    /// fused path wins decisively (up to 2.7× at Q14-shape ~1%
    /// selectivity), above it materialised is competitive and
    /// usually wins. The mid-range (20-50%) is noisy because the
    /// curves are close and depend on cache + allocator behaviour
    /// of the specific workload.
    ///
    /// `bench_adaptive_dispatch` (in `examples/`) reproduces the
    /// numbers above and is the right place to retune if a future
    /// workload shifts the crossover.
    ///
    /// For bitmap-consuming callers (filter chain, COUNT
    /// aggregator) the fused path always wins; those callers
    /// should keep using `decode_rle_dictionary_predicate_bitmap`
    /// directly rather than going through the adaptive runner.
    pub const DEFAULT_THRESHOLD: f32 = 0.10;

    /// Default probe budget — three pages gives a more robust
    /// selectivity reading than two (averages out a single
    /// unrepresentative page, e.g. the first page of a sorted
    /// column where every row falls outside the range), without
    /// burning too much of the chunk on probing.
    pub const DEFAULT_PROBE_PAGES: usize = 3;

    /// Build with default threshold + probe budget.
    pub fn new(dict_mask: Vec<u8>) -> Self {
        Self {
            dict_mask,
            threshold: Self::DEFAULT_THRESHOLD,
            probe_pages: Self::DEFAULT_PROBE_PAGES,
        }
    }

    /// Choose the path for the remainder of the chunk given an
    /// aggregated probe reading.
    pub fn dispatch_for(&self, aggregated: PageProbe) -> Dispatch {
        if aggregated.selectivity() > self.threshold {
            Dispatch::Materialized
        } else {
            Dispatch::Fused
        }
    }
}

/// Caller-facing config for the per-type adaptive façade entry
/// points (`read_column_*_predicate_adaptive`). Mirrors the
/// runtime knobs on `AdaptiveDictPredicate` minus the `dict_mask`
/// (which the façade builds from the caller's predicate).
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDispatchOptions {
    pub threshold: f32,
    pub probe_pages: usize,
}

impl Default for AdaptiveDispatchOptions {
    fn default() -> Self {
        Self {
            threshold: AdaptiveDictPredicate::DEFAULT_THRESHOLD,
            probe_pages: AdaptiveDictPredicate::DEFAULT_PROBE_PAGES,
        }
    }
}

/// Count the set bits in the first `num_bits` of a packed bitmap.
/// Bits at indices `≥ num_bits` within the final byte are ignored
/// even if non-zero (the fused kernel doesn't guarantee they're
/// zeroed — bitmap bytes only need to be correct within
/// `num_values`).
///
/// `bitmap.len()` must be `≥ num_bits.div_ceil(8)`.
pub fn popcount_bitmap_prefix(bitmap: &[u8], num_bits: usize) -> usize {
    if num_bits == 0 {
        return 0;
    }
    let full_bytes = num_bits / 8;
    let tail_bits = num_bits % 8;
    let mut acc: usize = 0;
    for &b in &bitmap[..full_bytes] {
        acc += b.count_ones() as usize;
    }
    if tail_bits != 0 {
        let mask: u8 = (1u8 << tail_bits) - 1;
        acc += (bitmap[full_bytes] & mask).count_ones() as usize;
    }
    acc
}

/// Run the fused predicate kernel on one data-page body and
/// return a probe reading. Appends the produced bitmap (`num_values
/// .div_ceil(8)` bytes) to `out`. `body` is the data-page body
/// after decompression, starting with the bit_width prefix.
pub fn probe_page_fused(
    body: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    out: &mut Vec<u8>,
) -> Result<PageProbe> {
    let bitmap_start = out.len();
    decode_rle_dictionary_predicate_bitmap(body, num_values, dict_mask, out)?;
    let rows_passed = popcount_bitmap_prefix(&out[bitmap_start..], num_values);
    Ok(PageProbe {
        rows_in: num_values,
        rows_passed,
    })
}

/// One decompressed page handed to the adaptive chunk runner.
///
/// The runner does not own decompression — pages arrive already
/// decompressed so the runner stays decoupled from the I/O / page-
/// walker layer. The per-type façade in Π.14e does the decompress
/// step and feeds these in.
#[derive(Debug, Clone)]
pub struct AdaptivePageInput<'a> {
    /// Decompressed data-page body, starting with the bit_width prefix.
    pub body: &'a [u8],
    /// Number of values in this page (from `PageHeader.num_values`).
    pub num_values: usize,
}

/// Result of running the adaptive runner across a chunk's pages.
///
/// `dispatch` records the path the runner committed to; `kind`
/// carries the actual output (bitmap or values). `total_rows` is
/// the sum of `num_values` across all input pages — exposed so
/// callers can sanity-check coverage without re-summing.
#[derive(Debug)]
pub struct AdaptiveChunkOutput<T> {
    pub dispatch: Dispatch,
    pub total_rows: usize,
    pub kind: AdaptiveOutputKind<T>,
}

/// Two output shapes the runner can produce. Which one comes back
/// depends on the dispatch decision made after the probe phase.
#[derive(Debug)]
pub enum AdaptiveOutputKind<T> {
    /// Fused path: packed predicate bitmap. Bit `i` of byte `k`
    /// represents row `8k+i` across the whole chunk; bits past
    /// `total_rows` are unspecified.
    Bitmap { bitmap: Vec<u8>, set_bits: usize },
    /// Materialised path: filtered values, in row order. Length
    /// equals `set_bits` from the equivalent bitmap output.
    Values(Vec<T>),
}

/// Per-chunk telemetry record passed to the optional callback.
/// Lets consumers log how often the runner picks each path and
/// what selectivity drove the decision.
#[derive(Debug, Clone, Copy)]
pub struct SelectivityProbe {
    pub pages_probed: usize,
    pub rows_in: usize,
    pub rows_passed: usize,
    pub selectivity: f32,
    pub dispatch: Dispatch,
}

/// Π.14b core: run the adaptive predicate runner across the pages
/// of one column chunk.
///
/// **Output-shape choice happens once per chunk** — the runner
/// commits to either a bitmap or a values vector before emitting
/// anything, so callers always see one consistent shape per chunk.
///
/// Algorithm:
/// 1. **Probe phase** — run the fused kernel on up to
///    `cfg.probe_pages` pages. Accumulate the bitmap + a popcount
///    so we know the observed selectivity.
/// 2. **Dispatch** — if observed selectivity > `cfg.threshold`,
///    commit to `Materialized`; otherwise `Fused`.
/// 3. **Emit phase** —
///    - `Fused`: keep running fused on remaining pages, return
///      the bitmap.
///    - `Materialized`: redecode the probed pages' indices and
///      gather values, then decode-then-filter remaining pages.
///      The duplicate decode of probed pages is bounded
///      (≤ `probe_pages` × page size) and tiny vs the rest of
///      the chunk.
///
/// `telemetry`, if provided, is called once with the final
/// `SelectivityProbe`.
pub fn run_adaptive_dict_chunk<T: Copy>(
    pages: &[AdaptivePageInput<'_>],
    dict: &[T],
    cfg: &AdaptiveDictPredicate,
    mut telemetry: Option<&mut dyn FnMut(SelectivityProbe)>,
) -> Result<AdaptiveChunkOutput<T>> {
    let total_rows: usize = pages.iter().map(|p| p.num_values).sum();

    // Empty chunk: emit empty bitmap, skip dispatch.
    if total_rows == 0 {
        if let Some(cb) = telemetry.as_mut() {
            cb(SelectivityProbe {
                pages_probed: 0,
                rows_in: 0,
                rows_passed: 0,
                selectivity: 0.0,
                dispatch: Dispatch::Fused,
            });
        }
        return Ok(AdaptiveChunkOutput {
            dispatch: Dispatch::Fused,
            total_rows: 0,
            kind: AdaptiveOutputKind::Bitmap {
                bitmap: Vec::new(),
                set_bits: 0,
            },
        });
    }

    // Probe phase — fused on the first `probe_pages` pages.
    let probe_end = cfg.probe_pages.min(pages.len());
    let mut probe_bitmap: Vec<u8> = Vec::new();
    let mut probe_rows_in: usize = 0;
    let mut probe_rows_passed: usize = 0;
    for page in &pages[..probe_end] {
        let p = probe_page_fused(
            page.body,
            page.num_values,
            &cfg.dict_mask,
            &mut probe_bitmap,
        )?;
        probe_rows_in += p.rows_in;
        probe_rows_passed += p.rows_passed;
    }
    let aggregated = PageProbe {
        rows_in: probe_rows_in,
        rows_passed: probe_rows_passed,
    };
    let dispatch = cfg.dispatch_for(aggregated);

    if let Some(cb) = telemetry.as_mut() {
        cb(SelectivityProbe {
            pages_probed: probe_end,
            rows_in: probe_rows_in,
            rows_passed: probe_rows_passed,
            selectivity: aggregated.selectivity(),
            dispatch,
        });
    }

    match dispatch {
        Dispatch::Fused => {
            // Continue with fused on remaining pages.
            let mut bitmap = probe_bitmap;
            for page in &pages[probe_end..] {
                let p = probe_page_fused(page.body, page.num_values, &cfg.dict_mask, &mut bitmap)?;
                probe_rows_passed += p.rows_passed;
            }
            Ok(AdaptiveChunkOutput {
                dispatch,
                total_rows,
                kind: AdaptiveOutputKind::Bitmap {
                    bitmap,
                    set_bits: probe_rows_passed,
                },
            })
        }
        Dispatch::Materialized => {
            // Re-decode probed pages to recover indices → values,
            // then filter via dict_mask. Subsequent pages go
            // straight to the materialise-then-filter path.
            let mut values: Vec<T> = Vec::with_capacity(probe_rows_passed);
            let mut tmp_values: Vec<T> = Vec::new();
            for page in pages.iter() {
                tmp_values.clear();
                decode_rle_dictionary_into(page.body, dict, page.num_values, &mut tmp_values)?;
                materialise_filter::<T>(
                    page.body,
                    page.num_values,
                    &cfg.dict_mask,
                    &tmp_values,
                    &mut values,
                )?;
            }
            Ok(AdaptiveChunkOutput {
                dispatch,
                total_rows,
                kind: AdaptiveOutputKind::Values(values),
            })
        }
    }
}

/// Decode one page's indices to a temporary bitmap, then emit
/// `values[row]` for each row where the bitmap bit is set. Uses
/// the same fused kernel as `probe_page_fused` for the bitmap step
/// so the filter is byte-identical with the fused path; the only
/// difference is that here we materialise the surviving values
/// instead of returning the bitmap itself.
fn materialise_filter<T: Copy>(
    body: &[u8],
    num_values: usize,
    dict_mask: &[u8],
    values: &[T],
    out: &mut Vec<T>,
) -> Result<()> {
    let mut bm: Vec<u8> = Vec::with_capacity(num_values.div_ceil(8));
    decode_rle_dictionary_predicate_bitmap(body, num_values, dict_mask, &mut bm)?;
    for row in 0..num_values {
        let bit = (bm[row / 8] >> (row % 8)) & 1;
        if bit != 0 {
            out.push(values[row]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcount_handles_partial_tail() {
        // Bits laid out so that the all-ones byte has 5 set bits
        // counted within the first 5 positions.
        let bm = [0b11111111u8, 0b00000111];
        assert_eq!(popcount_bitmap_prefix(&bm, 8), 8);
        assert_eq!(popcount_bitmap_prefix(&bm, 11), 8 + 3);
        // Bits past num_bits in the same byte must be ignored.
        let bm2 = [0b11111111u8];
        assert_eq!(popcount_bitmap_prefix(&bm2, 3), 3);
    }

    #[test]
    fn popcount_zero_bits_is_zero() {
        assert_eq!(popcount_bitmap_prefix(&[0xFF, 0xFF], 0), 0);
    }

    #[test]
    fn selectivity_handles_empty() {
        let p = PageProbe {
            rows_in: 0,
            rows_passed: 0,
        };
        assert_eq!(p.selectivity(), 0.0);
    }

    #[test]
    fn dispatch_threshold_boundary() {
        let cfg = AdaptiveDictPredicate {
            dict_mask: vec![],
            threshold: 0.10,
            probe_pages: 3,
        };
        // Exactly at threshold → still fused (strict greater-than).
        assert_eq!(
            cfg.dispatch_for(PageProbe {
                rows_in: 100,
                rows_passed: 10
            }),
            Dispatch::Fused
        );
        // Above threshold → materialised.
        assert_eq!(
            cfg.dispatch_for(PageProbe {
                rows_in: 100,
                rows_passed: 11
            }),
            Dispatch::Materialized
        );
        // Q14-shape (~1%) → fused.
        assert_eq!(
            cfg.dispatch_for(PageProbe {
                rows_in: 10_000,
                rows_passed: 100
            }),
            Dispatch::Fused
        );
    }
}
