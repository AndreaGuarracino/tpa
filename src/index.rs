use crate::binary::{read_header, skip_record, skip_record_sequential};
use crate::format::StringTable;
use crate::utils::{read_varint, write_varint};
use log::info;
use noodles::bgzf;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};

/// Index type: raw byte offsets or BGZF virtual positions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    /// Raw byte offsets into uncompressed file
    RawOffset = 0,
    /// BGZF virtual positions for all-records mode
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
    /// BGZF section start offset (for all-records mode, 0 for per-record mode)
    bgzf_section_start: u64,
}

impl TpaIndex {
    const INDEX_MAGIC: &'static [u8; 4] = b"TPAI";
    const INDEX_VERSION: u8 = 2;

    /// Create a new index with raw byte offsets (classic per-record mode)
    pub fn new_raw(offsets: Vec<u64>) -> Self {
        Self {
            index_type: IndexType::RawOffset,
            positions: offsets,
            bgzf_section_start: 0,
        }
    }

    /// Create a new index with BGZF virtual positions and section start (all-records mode)
    pub fn new_virtual_with_section_start(positions: Vec<u64>, bgzf_section_start: u64) -> Self {
        Self {
            index_type: IndexType::VirtualPosition,
            positions,
            bgzf_section_start,
        }
    }

    /// Save index to .tpa.idx file
    pub fn save(&self, idx_path: &str) -> io::Result<()> {
        let mut file = File::create(idx_path)?;

        file.write_all(Self::INDEX_MAGIC)?;
        file.write_all(&[Self::INDEX_VERSION])?;
        file.write_all(&[self.index_type.to_u8()])?;
        file.write_all(&self.bgzf_section_start.to_le_bytes())?;
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
        if version[0] != Self::INDEX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unsupported index version: {} (expected {})",
                    version[0],
                    Self::INDEX_VERSION
                ),
            ));
        }

        let mut type_buf = [0u8; 1];
        reader.read_exact(&mut type_buf)?;
        let index_type = IndexType::from_u8(type_buf[0])?;

        let mut bgzf_buf = [0u8; 8];
        reader.read_exact(&mut bgzf_buf)?;
        let bgzf_section_start = u64::from_le_bytes(bgzf_buf);

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
            bgzf_section_start,
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

    /// Get the BGZF section start offset (for all-records mode, 0 for per-record mode)
    pub fn bgzf_section_start(&self) -> u64 {
        self.bgzf_section_start
    }
}

/// Build an index from a TPA file by scanning record offsets (per-record mode).
pub fn build_index_per_record(tpa_path: &str) -> io::Result<TpaIndex> {
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

/// Build an index from an all-records mode TPA file by scanning BGZF virtual positions.
/// This allows recovery if the .idx file is deleted.
pub fn build_index_all_records(tpa_path: &str) -> io::Result<TpaIndex> {
    info!("Building index for all-records mode: {}", tpa_path);

    let mut file = File::open(tpa_path)?;

    // Read header (plain bytes)
    let header = read_header(&mut file)?;
    if !header.all_records() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "File is not in all-records mode, use build_index_per_record() instead",
        ));
    }

    // Skip string table (plain bytes)
    StringTable::read(&mut file, header.num_strings)?;

    // Record where BGZF section starts
    let bgzf_section_start = file.stream_position()?;

    // Create BGZF reader starting at records section
    file.seek(SeekFrom::Start(bgzf_section_start))?;
    let buf_reader = BufReader::with_capacity(131072, file);
    let mut bgzf_reader = bgzf::io::Reader::new(buf_reader);

    // Scan records and capture virtual positions
    let mut virtual_positions = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        // Capture virtual position BEFORE reading record
        let relative_vpos = u64::from(bgzf_reader.virtual_position());
        let block_offset = relative_vpos >> 16;
        let uncompressed_offset = relative_vpos & 0xFFFF;
        let absolute_block_offset = block_offset + bgzf_section_start;
        let absolute_vpos = (absolute_block_offset << 16) | uncompressed_offset;
        virtual_positions.push(absolute_vpos);

        // Skip record using read-only function (no Seek required)
        skip_record_sequential(&mut bgzf_reader, header.tracepoint_type)?;
    }

    info!(
        "Index built: {} records (all-records mode)",
        virtual_positions.len()
    );
    Ok(TpaIndex::new_virtual_with_section_start(
        virtual_positions,
        bgzf_section_start,
    ))
}
