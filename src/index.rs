use crate::binary::{read_header, skip_record};
use crate::format::StringTable;
use crate::utils::{read_varint, write_varint};
use log::info;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, Write};

/// Index type: raw byte offsets or BGZF virtual positions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    /// Raw byte offsets into uncompressed file
    RawOffset = 0,
    /// BGZF virtual positions for whole-file BGZIP mode
    VirtualPosition = 1,
}

impl IndexType {
    fn to_u8(self) -> u8 {
        self as u8
    }

    fn from_u8(value: u8) -> io::Result<Self> {
        match value {
            0 => Ok(IndexType::RawOffset),
            1 => Ok(IndexType::VirtualPosition),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid index type: {}", value),
            )),
        }
    }
}

/// File offset index for O(1) random access.
/// Supports both raw byte offsets (per-record mode) and BGZF virtual positions (all-records mode).
pub struct TpaIndex {
    /// Type of positions stored (raw offsets or virtual positions)
    index_type: IndexType,
    /// Positions for each record (byte offset or BGZF virtual position)
    positions: Vec<u64>,
}

impl TpaIndex {
    const INDEX_MAGIC: &'static [u8; 4] = b"TPAI";

    /// Create a new index with raw byte offsets (per-record mode)
    pub fn new_raw(offsets: Vec<u64>) -> Self {
        Self {
            index_type: IndexType::RawOffset,
            positions: offsets,
        }
    }

    /// Create a new index with BGZF virtual positions (all-records mode)
    pub fn new_virtual(positions: Vec<u64>) -> Self {
        Self {
            index_type: IndexType::VirtualPosition,
            positions,
        }
    }

    /// Save index to .tpa.idx file
    pub fn save(&self, idx_path: &str) -> io::Result<()> {
        let mut file = File::create(idx_path)?;

        file.write_all(Self::INDEX_MAGIC)?;
        file.write_all(&[1u8])?; // Version 1
        file.write_all(&[self.index_type.to_u8()])?; // Index type

        write_varint(&mut file, self.positions.len() as u64)?;

        // For virtual positions, use fixed u64 for precision; for raw offsets, use varint
        match self.index_type {
            IndexType::RawOffset => {
                for &pos in &self.positions {
                    write_varint(&mut file, pos)?;
                }
            }
            IndexType::VirtualPosition => {
                for &pos in &self.positions {
                    file.write_all(&pos.to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    /// Load index from .tpa.idx file
    pub fn load(idx_path: &str) -> io::Result<Self> {
        let file = File::open(idx_path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != Self::INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid index magic",
            ));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", version[0]),
            ));
        }

        let mut type_buf = [0u8; 1];
        reader.read_exact(&mut type_buf)?;
        let index_type = IndexType::from_u8(type_buf[0])?;

        let num_positions = read_varint(&mut reader)? as usize;

        let mut positions = Vec::with_capacity(num_positions);
        match index_type {
            IndexType::RawOffset => {
                for _ in 0..num_positions {
                    positions.push(read_varint(&mut reader)?);
                }
            }
            IndexType::VirtualPosition => {
                for _ in 0..num_positions {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf)?;
                    positions.push(u64::from_le_bytes(buf));
                }
            }
        }

        Ok(Self {
            index_type,
            positions,
        })
    }

    /// Get number of records in index
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get the index type
    pub fn index_type(&self) -> IndexType {
        self.index_type
    }

    /// Get position for a specific record ID (raw offset or virtual position depending on type)
    pub fn get_position(&self, record_id: u64) -> Option<u64> {
        self.positions.get(record_id as usize).copied()
    }
}

/// Build an index from a TPA file by scanning record offsets (per-record mode).
/// For all-records mode, use the index built during compression.
pub fn build_index(tpa_path: &str) -> io::Result<TpaIndex> {
    info!("Building index for {}", tpa_path);

    let file = File::open(tpa_path)?;
    let mut reader = BufReader::with_capacity(131072, file);

    let header = read_header(&mut reader)?;
    StringTable::read(&mut reader, header.num_strings)?;

    let mut offsets = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        offsets.push(reader.stream_position()?);
        skip_record(&mut reader, header.tracepoint_type)?;
    }

    info!("Index built: {} records", offsets.len());
    Ok(TpaIndex::new_raw(offsets))
}
