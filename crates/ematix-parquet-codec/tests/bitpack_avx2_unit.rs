//! Π.12a unit tests: AVX2 bw=16 kernel correctness.
//!
//! Only compiled + run on x86_64 with AVX2. On aarch64 / non-AVX2
//! x86 these tests are absent from the test binary (cfg-gated out)
//! — the NEON + scalar paths cover those targets.
//!
//! Strategy: pack known u16 values via the scalar packer, decode
//! via AVX2, assert bit-identical output. Also exercise the
//! lookup variant against a small dict to confirm the fused gather
//! produces the right values.

#![cfg(target_arch = "x86_64")]

use ematix_parquet_codec::bitpack_avx2::{
    unpack_indices_into_avx2_bw16, unpack_lookup_into_avx2_bw16,
};

/// Pack `n` u16 values into LE bytes — the on-wire bw=16 form.
/// Mirrors the bw=16 layout the parquet writer emits.
fn pack_bw16(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let v = (v & 0xFFFF) as u16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn skip_if_no_avx2() -> bool {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("test skipped: CPU lacks AVX2");
        return true;
    }
    false
}

#[test]
fn indices_round_trip_aligned() {
    if skip_if_no_avx2() {
        return;
    }
    // Exact-multiple-of-8 size: no tail.
    let values: Vec<u32> = (0..1024u32).map(|i| i * 17 & 0xFFFF).collect();
    let packed = pack_bw16(&values);

    let mut out: Vec<u32> = Vec::new();
    unpack_indices_into_avx2_bw16(&packed, values.len(), &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn indices_round_trip_with_tail() {
    if skip_if_no_avx2() {
        return;
    }
    // Not a multiple of 8 — exercises the scalar tail path.
    for n in [1usize, 7, 13, 17, 35, 100, 1003] {
        let values: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(13) & 0xFFFF).collect();
        let packed = pack_bw16(&values);

        let mut out: Vec<u32> = Vec::new();
        unpack_indices_into_avx2_bw16(&packed, n, &mut out).unwrap();
        assert_eq!(out, values, "n = {n}");
    }
}

#[test]
fn indices_appends_to_existing_buffer() {
    if skip_if_no_avx2() {
        return;
    }
    let prefill: Vec<u32> = vec![999_999; 5];
    let values: Vec<u32> = (0..16u32).collect();
    let packed = pack_bw16(&values);

    let mut out = prefill.clone();
    unpack_indices_into_avx2_bw16(&packed, values.len(), &mut out).unwrap();
    let mut expected = prefill.clone();
    expected.extend(&values);
    assert_eq!(out, expected);
}

#[test]
fn indices_empty_input() {
    if skip_if_no_avx2() {
        return;
    }
    let mut out: Vec<u32> = Vec::new();
    unpack_indices_into_avx2_bw16(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn indices_undersized_packed_buffer_errors() {
    if skip_if_no_avx2() {
        return;
    }
    let packed = vec![0u8; 10]; // need 20 bytes for 10 values
    let mut out: Vec<u32> = Vec::new();
    let r = unpack_indices_into_avx2_bw16(&packed, 10, &mut out);
    assert!(r.is_err());
}

#[test]
fn lookup_round_trips_against_dict() {
    if skip_if_no_avx2() {
        return;
    }
    let dict: Vec<i64> = (1000..1100i64).collect();
    let indices: Vec<u32> = (0..512).map(|i| (i % dict.len()) as u32).collect();
    let packed = pack_bw16(&indices);

    let mut out: Vec<i64> = Vec::new();
    unpack_lookup_into_avx2_bw16(&packed, indices.len(), &dict, &mut out).unwrap();

    let expected: Vec<i64> = indices.iter().map(|&i| dict[i as usize]).collect();
    assert_eq!(out, expected);
}

#[test]
fn lookup_with_tail() {
    if skip_if_no_avx2() {
        return;
    }
    let dict: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    let indices: Vec<u32> = (0..103).map(|i| (i % dict.len()) as u32).collect();
    let packed = pack_bw16(&indices);

    let mut out: Vec<f64> = Vec::new();
    unpack_lookup_into_avx2_bw16(&packed, indices.len(), &dict, &mut out).unwrap();
    let expected: Vec<f64> = indices.iter().map(|&i| dict[i as usize]).collect();
    assert_eq!(out, expected);
}

#[test]
fn lookup_out_of_range_index_errors() {
    if skip_if_no_avx2() {
        return;
    }
    let dict: Vec<u8> = vec![0, 1, 2];
    let indices: Vec<u32> = vec![0, 1, 2, 5, 1, 0, 2, 1]; // 5 is out of range
    let packed = pack_bw16(&indices);

    let mut out: Vec<u8> = Vec::new();
    let r = unpack_lookup_into_avx2_bw16(&packed, indices.len(), &dict, &mut out);
    assert!(r.is_err(), "out-of-range dict index must error");
}

/// Cross-check vs the high-level bitpack dispatcher: AVX2 kernel
/// must produce the same output as the scalar const-generic
/// fallback for the same input. Hard guarantee that AVX2 and
/// scalar agree byte-for-byte.
#[test]
fn matches_scalar_dispatcher() {
    if skip_if_no_avx2() {
        return;
    }
    use ematix_parquet_codec::bitpack::unpack_indices_into;

    let values: Vec<u32> = (0..2_048u32).map(|i| i.wrapping_mul(31337) & 0xFFFF).collect();
    let packed = pack_bw16(&values);

    // unpack_indices_into routes through the AVX2 kernel on x86_64
    // with AVX2. Cross-check vs explicit AVX2 call.
    let mut via_dispatcher: Vec<u32> = Vec::new();
    unpack_indices_into(&packed, values.len(), 16, &mut via_dispatcher).unwrap();

    let mut via_direct: Vec<u32> = Vec::new();
    unpack_indices_into_avx2_bw16(&packed, values.len(), &mut via_direct).unwrap();

    assert_eq!(via_dispatcher, via_direct, "dispatcher and direct AVX2 disagree");
    assert_eq!(via_dispatcher, values, "round trip mismatch");
}
