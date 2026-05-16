//! `AsyncParquetFile` — opens a Parquet file via `object_store`,
//! parses the footer with the suffix-range trick (≤ 2 round trips),
//! and exposes async byte-range reads.
//!
//! Footer-parse shape mirrors the sync `ematix-parquet-io::ParquetFile`
//! exactly — every metadata-level decode is shared via
//! `ematix-parquet-format`. The async crate just owns the I/O
//! integration with `object_store`.

use std::sync::Arc;

use bytes::Bytes;
use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_file_metadata, FileMetaData};
use object_store::{path::Path, GetOptions, GetRange, ObjectStore};

use crate::error::{AsyncError, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
const FOOTER_TRAILER_LEN: u64 = 8; // 4-byte length + 4-byte magic
/// Suffix-range size for the cold-open: covers most footers in a
/// single GET. Files with a larger footer (many columns × many
/// row groups) fall back to a second range fetch.
const COLD_OPEN_SUFFIX_BYTES: u64 = 8 * 1024;

/// Async parquet file backed by any `object_store::ObjectStore`.
///
/// Construct with `AsyncParquetFile::open`. After construction the
/// footer bytes are cached in-memory; `metadata()` re-parses them
/// per call (parse is microseconds; we don't cache a decoded
/// `FileMetaData` because it borrows from the footer bytes).
///
/// `read_range(offset, length)` issues one GET per call.
pub struct AsyncParquetFile {
    store: Arc<dyn ObjectStore>,
    path: Path,
    file_size: u64,
    footer_bytes: Bytes,
    /// Byte offset of the start of the FileMetaData struct in the
    /// underlying file (= `file_size - 8 - footer_len`). Used by
    /// the read façade as the upper bound on the last row group's
    /// chunk bodies.
    footer_offset: u64,
}

impl AsyncParquetFile {
    /// Cold-open: ≤ 2 round trips to fetch + cache the footer.
    ///
    /// 1. GET the last `COLD_OPEN_SUFFIX_BYTES` of the file via
    ///    `GetRange::Suffix`. Response includes the object size in
    ///    `meta.size`.
    /// 2. Parse the trailing 8 bytes: `[footer_len:u32 LE][magic 4]`.
    /// 3. If the suffix response already covers the full footer
    ///    (common case, footer < 8 KB), slice it out and we're done
    ///    in one GET.
    /// 4. Otherwise fall back to a precise GET for the whole footer
    ///    range (second round trip).
    pub async fn open(store: Arc<dyn ObjectStore>, path: Path) -> Result<Self> {
        // Step 1: suffix-range GET. object_store handles Range:
        // bytes=-N (the "give me the last N bytes" form).
        let opts = GetOptions {
            range: Some(GetRange::Suffix(COLD_OPEN_SUFFIX_BYTES as usize)),
            ..Default::default()
        };
        let result = store.get_opts(&path, opts).await?;
        let file_size = result.meta.size as u64;
        let suffix: Bytes = result.bytes().await?;
        if suffix.len() < FOOTER_TRAILER_LEN as usize {
            return Err(AsyncError::TruncatedFooter {
                file_size,
                declared_footer_length: 0,
            });
        }

        // Step 2: parse the trailer.
        let trailer = &suffix[suffix.len() - FOOTER_TRAILER_LEN as usize..];
        let tail_magic: [u8; 4] = trailer[4..].try_into().unwrap();
        if &tail_magic != PARQUET_MAGIC {
            return Err(AsyncError::NotAParquetFile {
                expected: PARQUET_MAGIC,
                found: tail_magic,
                position: "file tail",
            });
        }
        let footer_len = u32::from_le_bytes(trailer[0..4].try_into().unwrap()) as u64;
        if footer_len > file_size - FOOTER_TRAILER_LEN - 4 {
            return Err(AsyncError::TruncatedFooter {
                file_size,
                declared_footer_length: footer_len,
            });
        }
        let footer_offset = file_size - FOOTER_TRAILER_LEN - footer_len;

        // Step 3 / 4: extract footer bytes — from the suffix if it
        // covers them, otherwise reissue a precise range GET.
        let footer_bytes: Bytes = if suffix.len() as u64 >= footer_len + FOOTER_TRAILER_LEN {
            let start = suffix.len() - FOOTER_TRAILER_LEN as usize - footer_len as usize;
            let end = suffix.len() - FOOTER_TRAILER_LEN as usize;
            suffix.slice(start..end)
        } else {
            let start = footer_offset as usize;
            let end = (footer_offset + footer_len) as usize;
            let opts = GetOptions {
                range: Some(GetRange::Bounded(start..end)),
                ..Default::default()
            };
            let result = store.get_opts(&path, opts).await?;
            let bytes = result.bytes().await?;
            if bytes.len() as u64 != footer_len {
                return Err(AsyncError::EmptyResponse);
            }
            bytes
        };

        Ok(Self {
            store,
            path,
            file_size,
            footer_bytes,
            footer_offset,
        })
    }

    /// Path within the object store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total file size in bytes (from the GET response metadata).
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Byte offset where the FileMetaData struct begins in the
    /// underlying file. Equal to `file_size - 8 - footer_len`.
    pub fn footer_offset(&self) -> u64 {
        self.footer_offset
    }

    /// Cached raw footer bytes (the thrift FileMetaData payload).
    pub fn footer_bytes(&self) -> &[u8] {
        &self.footer_bytes
    }

    /// Decode the file's `FileMetaData`. Re-decodes on every call;
    /// callers that need it repeatedly should bind it.
    pub fn metadata(&self) -> Result<FileMetaData<'_>> {
        let mut cur = Cursor::new(&self.footer_bytes);
        Ok(read_file_metadata(&mut cur)?)
    }

    /// Async byte-range read. Issues one GET; returns a `Bytes`
    /// borrowed from the underlying transport (zero-copy where the
    /// store supports it).
    pub async fn read_range(&self, offset: u64, length: u64) -> Result<Bytes> {
        if offset.saturating_add(length) > self.file_size {
            return Err(AsyncError::OutOfRangeRead {
                offset,
                length,
                file_size: self.file_size,
            });
        }
        if length == 0 {
            return Ok(Bytes::new());
        }
        let start = offset as usize;
        let end = (offset + length) as usize;
        let opts = GetOptions {
            range: Some(GetRange::Bounded(start..end)),
            ..Default::default()
        };
        let result = self.store.get_opts(&self.path, opts).await?;
        Ok(result.bytes().await?)
    }
}
