//! TDD pin for the thrift compact-protocol primitive readers.
//!
//! Reference vectors come from the protocol spec
//! (https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md)
//! and from manually-encoded boundary values. The zigzag formula is
//! the unambiguous oracle:
//!   encode(v: i64) = (v << 1) ^ (v >> 63)
//!   decode(u: u64) = (u >> 1) as i64 ^ -((u & 1) as i64)
//! so test inputs can be derived without depending on any other library.

use ematix_parquet_format::compact::{read_uvarint, read_zigzag_i32, read_zigzag_i64, Cursor};
use ematix_parquet_format::error::FormatError;

#[test]
fn uvarint_single_byte_zero() {
    let bytes = [0x00];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 0);
    assert_eq!(cur.position(), 1);
}

#[test]
fn uvarint_single_byte_one() {
    let bytes = [0x01];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 1);
    assert_eq!(cur.position(), 1);
}

#[test]
fn uvarint_single_byte_max_127() {
    // High bit clear → terminal byte, value = 127.
    let bytes = [0x7f];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 127);
    assert_eq!(cur.position(), 1);
}

#[test]
fn uvarint_two_byte_128() {
    // 128 = 0b10000000 → 7-bit groups: [0000001][0000000]
    // LEB128 little-endian: low 7 bits first with continuation bit set,
    // then high 7 bits.
    let bytes = [0x80, 0x01];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 128);
    assert_eq!(cur.position(), 2);
}

#[test]
fn uvarint_advances_past_consumed_bytes_only() {
    // Two back-to-back varints: [127], [128]. After the first read the
    // cursor must be at 1, leaving the second varint intact.
    let bytes = [0x7f, 0x80, 0x01];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 127);
    assert_eq!(cur.position(), 1);
    assert_eq!(read_uvarint(&mut cur).unwrap(), 128);
    assert_eq!(cur.position(), 3);
}

#[test]
fn uvarint_eof_returns_error() {
    let bytes = [];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_uvarint(&mut cur),
        Err(FormatError::UnexpectedEof { .. })
    ));
}

#[test]
fn uvarint_truncated_continuation_returns_error() {
    // Continuation bit set but no follow-up byte.
    let bytes = [0x80];
    let mut cur = Cursor::new(&bytes);
    assert!(matches!(
        read_uvarint(&mut cur),
        Err(FormatError::UnexpectedEof { .. })
    ));
}

#[test]
fn uvarint_overflow_after_ten_continuation_bytes() {
    // 11 bytes all with continuation bit set → must not OOM-loop.
    let bytes = [0x80u8; 11];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_uvarint(&mut cur), Err(FormatError::VarintOverflow));
}

#[test]
fn zigzag_i32_zero() {
    let bytes = [0x00];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 0);
}

#[test]
fn zigzag_i32_one() {
    // zz(1) = 2 → uvarint 0x02
    let bytes = [0x02];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), 1);
}

#[test]
fn zigzag_i32_negative_one() {
    // zz(-1) = 1 → uvarint 0x01
    let bytes = [0x01];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), -1);
}

#[test]
fn zigzag_i32_negative_two() {
    // zz(-2) = 3 → uvarint 0x03
    let bytes = [0x03];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), -2);
}

#[test]
fn zigzag_i32_max() {
    // zz(i32::MAX = 2147483647) = 4294967294 = 0xFFFFFFFE
    // uvarint encoding of 0xFFFFFFFE:
    //   byte0 = 0xFE | 0x80 = 0xFE  (low 7 bits = 0b1111110)
    //   Actually let's compute: 0xFFFFFFFE in 7-bit groups, LSB first:
    //     g0 = 0xFE & 0x7F = 0x7E (continuation: more bytes follow)
    //     g1 = (0xFFFFFFFE >> 7) & 0x7F = 0x7F (cont)
    //     g2 = (>> 14) & 0x7F = 0x7F (cont)
    //     g3 = (>> 21) & 0x7F = 0x7F (cont)
    //     g4 = (>> 28) & 0x7F = 0x0F (terminal)
    let bytes = [0xFE, 0xFF, 0xFF, 0xFF, 0x0F];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), i32::MAX);
    assert_eq!(cur.position(), 5);
}

#[test]
fn zigzag_i32_min() {
    // zz(i32::MIN = -2147483648) = 4294967295 = 0xFFFFFFFF
    let bytes = [0xFF, 0xFF, 0xFF, 0xFF, 0x0F];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(read_zigzag_i32(&mut cur).unwrap(), i32::MIN);
    assert_eq!(cur.position(), 5);
}

#[test]
fn zigzag_i64_roundtrip_boundary_values() {
    // Derive expected bytes via the spec formula so the test is
    // self-contained.
    fn encode_zigzag_i64(v: i64) -> Vec<u8> {
        let mut u = ((v << 1) ^ (v >> 63)) as u64;
        let mut out = Vec::new();
        loop {
            if u < 0x80 {
                out.push(u as u8);
                return out;
            }
            out.push(((u & 0x7F) | 0x80) as u8);
            u >>= 7;
        }
    }
    for v in [0i64, 1, -1, 2, -2, 100, -100, i64::MAX, i64::MIN] {
        let bytes = encode_zigzag_i64(v);
        let mut cur = Cursor::new(&bytes);
        assert_eq!(
            read_zigzag_i64(&mut cur).unwrap(),
            v,
            "round-trip failed for {v}"
        );
        assert_eq!(cur.position(), bytes.len());
    }
}

#[test]
fn cursor_remaining_tracks_position() {
    let bytes = [0x01, 0x02, 0x03];
    let mut cur = Cursor::new(&bytes);
    assert_eq!(cur.remaining(), 3);
    let _ = read_uvarint(&mut cur).unwrap();
    assert_eq!(cur.remaining(), 2);
}
