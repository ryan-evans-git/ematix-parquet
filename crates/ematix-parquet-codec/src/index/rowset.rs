//! v2 tagged rowset encoding.
//!
//! v1 sidecars stored every rowset as a raw page bitmap
//! (`num_page_values/8` bytes per distinct `(value, page)`), which
//! explodes on high-cardinality keys: a real 3 GB TPC-H lineitem part
//! (~19 M distinct `l_orderkey`) produced a 47 GB rowset column —
//! past snappy's 4 GiB single-buffer cap at build time, and infeasible
//! to load even if written.
//!
//! v2 prefixes every rowset with a 1-byte tag and picks the smaller
//! form per rowset:
//!
//! ```text
//! [0x00][bitmap bytes …]                                  dense
//! [0x01][u32le num_page_values][u32le count][u32le row…]  sparse
//! ```
//!
//! Sparse embeds `num_page_values` so normalization back to a bitmap
//! needs no page-layout context. The public [`IndexHit`] contract is
//! unchanged — hits always carry a packed bitmap; stored sparse
//! rowsets are normalized at hit time (hits are few by construction:
//! they exist only for the looked-up key).
//!
//! Version fencing lives in the manifest KV key
//! (`ematix_index_manifest_v2`): a v1 reader finds no v1 manifest on a
//! v2 sidecar and refuses loudly instead of misreading tagged bytes as
//! a bitmap. The v2 reader reads both.
//!
//! [`IndexHit`]: crate::index::types::IndexHit

use crate::error::{CodecError, Result};

pub(crate) const TAG_BITMAP: u8 = 0x00;
pub(crate) const TAG_SPARSE: u8 = 0x01;

/// How stored rowset bytes are interpreted, decided by which manifest
/// KV key the sidecar carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowsetFormat {
    /// `ematix_index_manifest_v1`: raw bitmap, no tag.
    V1Raw,
    /// `ematix_index_manifest_v2`: 1-byte tag + payload.
    V2Tagged,
}

/// Encode page-relative row positions as a tagged rowset, choosing
/// whichever form is smaller. Positions must be `< num_page_values`
/// (debug-asserted, mirroring the v1 builder).
pub(crate) fn encode_tagged(positions: &[u32], num_page_values: usize) -> Vec<u8> {
    let bitmap_len = num_page_values.div_ceil(8);
    let sparse_len = 1 + 4 + 4 + 4 * positions.len();
    if sparse_len < 1 + bitmap_len {
        let mut out = Vec::with_capacity(sparse_len);
        out.push(TAG_SPARSE);
        out.extend_from_slice(&(num_page_values as u32).to_le_bytes());
        out.extend_from_slice(&(positions.len() as u32).to_le_bytes());
        for p in positions {
            debug_assert!(
                (*p as usize) < num_page_values,
                "row_within_page out of range"
            );
            out.extend_from_slice(&p.to_le_bytes());
        }
        out
    } else {
        let mut out = vec![0u8; 1 + bitmap_len];
        out[0] = TAG_BITMAP;
        for &r in positions {
            let r = r as usize;
            debug_assert!(r < num_page_values, "row_within_page out of range");
            out[1 + r / 8] |= 1 << (r % 8);
        }
        out
    }
}

/// Normalize stored rowset bytes to the packed bitmap the
/// [`IndexHit`](crate::index::types::IndexHit) contract promises.
pub(crate) fn to_bitmap(bytes: &[u8], format: RowsetFormat) -> Result<Vec<u8>> {
    match format {
        RowsetFormat::V1Raw => Ok(bytes.to_vec()),
        RowsetFormat::V2Tagged => match bytes.first() {
            Some(&TAG_BITMAP) => Ok(bytes[1..].to_vec()),
            Some(&TAG_SPARSE) => {
                if bytes.len() < 9 {
                    return Err(CodecError::InvalidInput(format!(
                        "v2 sparse rowset truncated: {} bytes",
                        bytes.len()
                    )));
                }
                let npv = u32::from_le_bytes(bytes[1..5].try_into().expect("4 bytes")) as usize;
                let count = u32::from_le_bytes(bytes[5..9].try_into().expect("4 bytes")) as usize;
                if bytes.len() != 9 + 4 * count {
                    return Err(CodecError::InvalidInput(format!(
                        "v2 sparse rowset length mismatch: {} bytes for count={count}",
                        bytes.len()
                    )));
                }
                let mut bm = vec![0u8; npv.div_ceil(8)];
                for i in 0..count {
                    let off = 9 + 4 * i;
                    let r = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 bytes"))
                        as usize;
                    if r >= npv {
                        return Err(CodecError::InvalidInput(format!(
                            "v2 sparse rowset row {r} out of page range {npv}"
                        )));
                    }
                    bm[r / 8] |= 1 << (r % 8);
                }
                Ok(bm)
            }
            Some(tag) => Err(CodecError::InvalidInput(format!(
                "v2 rowset has unknown tag {tag:#04x}"
            ))),
            None => Err(CodecError::InvalidInput("v2 rowset is empty".into())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_round_trips_to_bitmap() {
        // 3 positions on a 200-value page: sparse = 21 B < bitmap 26 B.
        let enc = encode_tagged(&[0, 3, 17], 200);
        assert_eq!(enc[0], TAG_SPARSE);
        let bm = to_bitmap(&enc, RowsetFormat::V2Tagged).unwrap();
        assert_eq!(bm.len(), 25);
        assert_eq!(bm[0], 0b0000_1001); // rows 0, 3
        assert_eq!(bm[2], 0b0000_0010); // row 17
        assert!(bm[3..].iter().all(|b| *b == 0));
    }

    #[test]
    fn dense_picks_bitmap_and_round_trips() {
        let positions: Vec<u32> = (0..64).collect();
        let enc = encode_tagged(&positions, 64);
        assert_eq!(enc[0], TAG_BITMAP);
        assert_eq!(enc.len(), 1 + 8);
        let bm = to_bitmap(&enc, RowsetFormat::V2Tagged).unwrap();
        assert_eq!(bm, vec![0xFF; 8]);
    }

    #[test]
    fn choice_boundary_is_by_size() {
        // 1 position: sparse = 13 B; bitmap for 200 values = 26 B.
        assert_eq!(encode_tagged(&[5], 200)[0], TAG_SPARSE);
        // Many positions on a small page: bitmap wins.
        assert_eq!(encode_tagged(&[0, 1, 2, 3, 4, 5, 6, 7], 8)[0], TAG_BITMAP);
    }

    #[test]
    fn v1_raw_passes_through() {
        let raw = vec![0xAB, 0xCD];
        assert_eq!(to_bitmap(&raw, RowsetFormat::V1Raw).unwrap(), raw);
    }

    #[test]
    fn malformed_v2_fails_loud() {
        assert!(to_bitmap(&[], RowsetFormat::V2Tagged).is_err());
        assert!(to_bitmap(&[0x02, 1, 2], RowsetFormat::V2Tagged).is_err());
        assert!(to_bitmap(&[TAG_SPARSE, 1, 0], RowsetFormat::V2Tagged).is_err());
        // Row id beyond the declared page size.
        let mut bad = vec![TAG_SPARSE];
        bad.extend_from_slice(&8u32.to_le_bytes());
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&9u32.to_le_bytes());
        assert!(to_bitmap(&bad, RowsetFormat::V2Tagged).is_err());
    }
}
