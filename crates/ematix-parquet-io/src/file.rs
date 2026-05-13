//! `ParquetFile` — opens a Parquet file, caches its footer bytes, and
//! exposes synchronous byte-range reads.
//!
//! The footer is parsed on demand via `metadata()`. We don't cache a
//! decoded `FileMetaData` because it borrows from the footer bytes
//! (self-referential ownership). Footer decoding is microseconds —
//! callers that need the metadata more than once can hold the
//! returned struct for as long as they hold `&ParquetFile`.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ematix_parquet_format::compact::Cursor;
use ematix_parquet_format::metadata::{read_file_metadata, FileMetaData};

use crate::error::{IoError, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
const FOOTER_TRAILER_LEN: u64 = 8; // 4 bytes length + 4 bytes magic

pub struct ParquetFile {
    path: PathBuf,
    file: Mutex<File>,
    file_size: u64,
    footer_bytes: Vec<u8>,
    /// Byte offset of the start of the footer struct (where the thrift
    /// FileMetaData payload begins). Equal to `file_size - 8 - footer_len`.
    /// Used as the upper bound for the last column chunk in row-group N
    /// because parquet writes the row-group bodies immediately followed
    /// by the footer trailer.
    footer_offset: u64,
}

impl ParquetFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_size = file.metadata()?.len();

        // Leading magic: every Parquet file starts with "PAR1".
        if file_size < 4 + FOOTER_TRAILER_LEN {
            return Err(IoError::TruncatedFooter {
                file_size,
                declared_footer_length: 0,
            });
        }
        let mut head = [0u8; 4];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut head)?;
        if &head != PARQUET_MAGIC {
            return Err(IoError::NotAParquetFile {
                expected: PARQUET_MAGIC,
                found: head,
                position: "file head",
            });
        }

        // Trailing magic + footer length.
        let mut tail = [0u8; 8];
        file.seek(SeekFrom::End(-(FOOTER_TRAILER_LEN as i64)))?;
        file.read_exact(&mut tail)?;
        let tail_magic: [u8; 4] = tail[4..].try_into().unwrap();
        if &tail_magic != PARQUET_MAGIC {
            return Err(IoError::NotAParquetFile {
                expected: PARQUET_MAGIC,
                found: tail_magic,
                position: "file tail",
            });
        }
        let footer_len = u32::from_le_bytes(tail[0..4].try_into().unwrap()) as u64;
        if footer_len > file_size - FOOTER_TRAILER_LEN - 4 {
            return Err(IoError::TruncatedFooter {
                file_size,
                declared_footer_length: footer_len,
            });
        }
        let footer_offset = file_size - FOOTER_TRAILER_LEN - footer_len;

        // Footer bytes proper.
        let mut footer_bytes = vec![0u8; footer_len as usize];
        file.seek(SeekFrom::Start(footer_offset))?;
        file.read_exact(&mut footer_bytes)?;

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_size,
            footer_bytes,
            footer_offset,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn footer_offset(&self) -> u64 {
        self.footer_offset
    }

    pub fn footer_bytes(&self) -> &[u8] {
        &self.footer_bytes
    }

    /// Decode the file's `FileMetaData`. Re-decodes on every call;
    /// callers that need it repeatedly should bind it.
    pub fn metadata(&self) -> Result<FileMetaData<'_>> {
        let mut cur = Cursor::new(&self.footer_bytes);
        Ok(read_file_metadata(&mut cur)?)
    }

    /// Read `length` bytes starting at byte `offset` into a fresh Vec.
    pub fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        if offset.saturating_add(length) > self.file_size {
            return Err(IoError::OutOfRangeRead {
                offset,
                length,
                file_size: self.file_size,
            });
        }
        let mut buf = vec![0u8; length as usize];
        let mut file = self.file.lock().expect("file mutex poisoned");
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buf)?;
        Ok(buf)
    }
}
