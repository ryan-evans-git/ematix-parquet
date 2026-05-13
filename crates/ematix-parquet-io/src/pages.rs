//! `PageWalker` — given the in-memory bytes of a column chunk, yield
//! `(PageHeader, body_slice)` pairs one at a time.
//!
//! Each page on disk is laid out as:
//!   [thrift-compact PageHeader] [compressed_page_size bytes of body]
//! and column chunks are just a sequence of these. The walker reads
//! one `PageHeader`, slices off `compressed_page_size` body bytes,
//! and advances the internal cursor.

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_page_header, PageHeader};

use crate::error::{IoError, Result};

pub struct PageWalker<'a> {
    cur: Cursor<'a>,
    /// Cap on bytes the walker is allowed to consume. When the
    /// internal cursor reaches this offset (or runs out of bytes
    /// at the buffer level), the walker yields `None`.
    end_offset: usize,
}

impl<'a> PageWalker<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            cur: Cursor::new(bytes),
            end_offset: bytes.len(),
        }
    }

    /// Construct a walker that stops once it has consumed `byte_limit`
    /// bytes from the buffer. Useful when the caller has read a
    /// larger range than just the chunk (e.g. a footer-aware bound).
    pub fn with_byte_limit(bytes: &'a [u8], byte_limit: usize) -> Self {
        Self {
            cur: Cursor::new(bytes),
            end_offset: byte_limit.min(bytes.len()),
        }
    }

    pub fn position(&self) -> usize {
        self.cur.position()
    }

    /// Read the next page header + body slice.
    /// Returns `Ok(None)` at clean end-of-chunk.
    pub fn next_page(&mut self) -> Result<Option<(PageHeader<'a>, &'a [u8])>> {
        if self.cur.position() >= self.end_offset {
            return Ok(None);
        }
        let header = read_page_header(&mut self.cur).map_err(IoError::from)?;
        let body_len = header.compressed_page_size as usize;
        let body = self.cur.take(body_len).map_err(IoError::from)?;
        Ok(Some((header, body)))
    }
}
