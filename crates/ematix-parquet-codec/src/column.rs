//! `DictColumnChunk<T>` — late-materialization view over a parquet
//! column chunk.
//!
//! The dict-encoded vs PLAIN-encoded distinction is preserved at the
//! API level rather than collapsed into a `Vec<T>` per parquet-rs's
//! eager model. Downstream consumers (filter, aggregate) can:
//!
//!   - Evaluate predicates against the **dict** (small) once,
//!     then look up per-row via index. For Q14 shipdate, that's
//!     2,555 predicate calls + 1M index reads vs the eager path's
//!     1M predicate calls.
//!   - Skip materialization entirely when only counts/sums are
//!     needed.
//!   - Late-bind to Arrow buffers when the consumer actually needs
//!     a `Vec<T>`.
//!
//! v0 only ships the data type + a small set of consumers (filter,
//! collect, count_matching). The reader-side constructor lives in
//! `reader.rs`.

use std::sync::Arc;

/// Multi-page column-chunk view. `segments[i]` corresponds to data
/// page `i` (post-dict-page). A page is either an array of indices
/// into `dict`, or a contiguous run of `Plain` values (when the
/// writer abandoned the dictionary).
///
/// `dict` is `None` only for columns that never had a dictionary
/// page; lineitem columns always have one.
#[derive(Debug, Clone)]
pub struct DictColumnChunk<T: Copy> {
    pub dict: Option<Arc<Vec<T>>>,
    pub segments: Vec<Segment<T>>,
    pub num_values: usize,
}

#[derive(Debug, Clone)]
pub enum Segment<T: Copy> {
    DictIndices(Vec<u32>),
    Plain(Vec<T>),
}

impl<T: Copy> DictColumnChunk<T> {
    pub fn new(dict: Option<Arc<Vec<T>>>, segments: Vec<Segment<T>>, num_values: usize) -> Self {
        Self {
            dict,
            segments,
            num_values,
        }
    }

    /// Build a per-row mask via `predicate`. Dict-encoded segments
    /// evaluate `predicate` once per dict entry, then index per row.
    /// PLAIN segments evaluate per value.
    pub fn filter<P: Fn(T) -> bool>(&self, predicate: P) -> Vec<bool> {
        let dict_mask: Option<Vec<bool>> = self
            .dict
            .as_ref()
            .map(|d| d.iter().copied().map(&predicate).collect());
        let mut out = Vec::with_capacity(self.num_values);
        for seg in &self.segments {
            match seg {
                Segment::DictIndices(idx) => {
                    let m = dict_mask
                        .as_ref()
                        .expect("dict-encoded segment but no dict was present");
                    for &i in idx {
                        out.push(m[i as usize]);
                    }
                }
                Segment::Plain(values) => {
                    for &v in values {
                        out.push(predicate(v));
                    }
                }
            }
        }
        out
    }

    /// Count rows where `predicate` is true. Doesn't allocate a mask;
    /// purely a tight reduce over each segment.
    pub fn count_matching<P: Fn(T) -> bool>(&self, predicate: P) -> usize {
        let dict_mask: Option<Vec<bool>> = self
            .dict
            .as_ref()
            .map(|d| d.iter().copied().map(&predicate).collect());
        let mut count = 0usize;
        for seg in &self.segments {
            match seg {
                Segment::DictIndices(idx) => {
                    let m = dict_mask.as_ref().expect("dict-encoded without dict");
                    for &i in idx {
                        count += m[i as usize] as usize;
                    }
                }
                Segment::Plain(values) => {
                    for &v in values {
                        count += predicate(v) as usize;
                    }
                }
            }
        }
        count
    }

    /// Materialize every row to a `Vec<T>`. Equivalent to the eager
    /// decoder's output.
    pub fn collect(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.num_values);
        for seg in &self.segments {
            match seg {
                Segment::DictIndices(idx) => {
                    let d = self.dict.as_ref().expect("dict-encoded without dict");
                    for &i in idx {
                        out.push(d[i as usize]);
                    }
                }
                Segment::Plain(values) => out.extend_from_slice(values),
            }
        }
        out
    }

    /// Materialize only the rows where `mask[i]` is true.
    pub fn gather(&self, mask: &[bool]) -> Vec<T> {
        assert_eq!(mask.len(), self.num_values, "mask length mismatch");
        let mut out = Vec::with_capacity(mask.iter().filter(|&&b| b).count());
        let mut row = 0;
        for seg in &self.segments {
            match seg {
                Segment::DictIndices(idx) => {
                    let d = self.dict.as_ref().expect("dict-encoded without dict");
                    for &i in idx {
                        if mask[row] {
                            out.push(d[i as usize]);
                        }
                        row += 1;
                    }
                }
                Segment::Plain(values) => {
                    for &v in values {
                        if mask[row] {
                            out.push(v);
                        }
                        row += 1;
                    }
                }
            }
        }
        out
    }
}
