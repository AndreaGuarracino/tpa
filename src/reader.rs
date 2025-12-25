use crate::binary::{read_record, read_tracepoints, read_tracepoints_at_offset};
use crate::format::{
    AlignmentRecord, CompressionLayer, CompressionStrategy, StringTable, TpaHeader,
};
use crate::index::{build_index_all_records, build_index_per_record, IndexType, TpaIndex};
use crate::utils::read_varint;
use log::{debug, info};
use noodles::bgzf;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

/// Helper: create "reader not initialized" error
#[inline]
fn err_not_initialized() -> io::Error {
    io::Error::other("Reader not properly initialized")
}

/// Helper: create "record out of bounds" error
#[inline]
fn err_out_of_bounds(record_id: u64) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Record id {} out of bounds", record_id),
    )
}

/// Helper: skip record header fields to reach tracepoint data
/// Skips: query_name_id, query_start, query_end, strand, target_name_id,
///        target_start, target_end, residue_matches, alignment_block_len, mapping_quality
#[inline]
fn skip_record_header<R: Read>(reader: &mut R) -> io::Result<()> {
    read_varint(reader)?; // query_name_id
    read_varint(reader)?; // query_start
    read_varint(reader)?; // query_end
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?; // strand
    read_varint(reader)?; // target_name_id
    read_varint(reader)?; // target_start
    read_varint(reader)?; // target_end
    read_varint(reader)?; // residue_matches
    read_varint(reader)?; // alignment_block_len
    reader.read_exact(&mut buf)?; // mapping_quality
    Ok(())
}

/// TPA reader supporting both per-record and all-records compression modes.
pub struct TpaReader {
    /// Raw file handle (used in per-record compression mode)
    file: Option<File>,
    /// BGZF reader (used in all-records mode)
    bgzf_reader: Option<bgzf::io::Reader<BufReader<File>>>,
    index: TpaIndex,
    header: TpaHeader,
    string_table: StringTable,
    /// Position of string table (raw file offset)
    string_table_pos: u64,
    /// File path for reopening when needed (all-records mode)
    tpa_path: Option<String>,
    /// BGZF section start offset (for all-records mode, 0 for per-record mode)
    bgzf_section_start: u64,
}

impl TpaReader {
    /// Create a TPA reader with index (builds index if .tpa.idx doesn't exist)
    /// Automatically detects compression mode based on header's all_records flag:
    /// - true → all-records mode (header/string table plain, records BGZIP-compressed)
    /// - false → per-record compression mode
    pub fn new(tpa_path: &str) -> io::Result<Self> {
        let mut file = File::open(tpa_path)?;
        let header = TpaHeader::read(&mut file)?;

        if header.all_records() {
            Self::open_all_records_mode(tpa_path, file, header)
        } else {
            Self::open_per_record_mode(tpa_path, file, header)
        }
    }

    /// Open a per-record compressed TPA file
    fn open_per_record_mode(tpa_path: &str, mut file: File, header: TpaHeader) -> io::Result<Self> {
        let string_table_pos = file.stream_position()?;

        let idx_path = format!("{}.idx", tpa_path);

        let index = if Path::new(&idx_path).exists() {
            debug!("Loading existing index: {}", idx_path);
            let loaded_idx = TpaIndex::load(&idx_path)?;
            // Validate index matches header record count - rebuild if stale
            if loaded_idx.len() != header.num_records() as usize {
                info!(
                    "Index record count ({}) doesn't match header ({}), rebuilding...",
                    loaded_idx.len(),
                    header.num_records()
                );
                let idx = build_index_per_record(tpa_path)?;
                idx.save(&idx_path)?;
                debug!("Index saved to {}", idx_path);
                idx
            } else {
                loaded_idx
            }
        } else {
            info!("No index found, building...");
            let idx = build_index_per_record(tpa_path)?;
            idx.save(&idx_path)?;
            debug!("Index saved to {}", idx_path);
            idx
        };

        let string_table = StringTable::new();

        Ok(Self {
            file: Some(file),
            bgzf_reader: None,
            index,
            header,
            string_table,
            string_table_pos,
            tpa_path: None, // Not needed for per-record mode
            bgzf_section_start: 0,
        })
    }

    /// Open an all-records mode TPA file
    /// Format: [Header (plain)] [StringTable (plain)] [BGZF: Records...] [BGZF EOF] [Footer (plain)]
    fn open_all_records_mode(
        tpa_path: &str,
        mut file: File,
        header: TpaHeader,
    ) -> io::Result<Self> {
        // String table position is current position after header
        let string_table_pos = file.stream_position()?;

        // Load or rebuild index
        let idx_path = format!("{}.idx", tpa_path);
        let index = if Path::new(&idx_path).exists() {
            debug!("Loading existing index: {}", idx_path);
            let loaded_idx = TpaIndex::load(&idx_path)?;
            // Validate index matches header record count - rebuild if stale
            if loaded_idx.len() != header.num_records() as usize {
                info!(
                    "Index record count ({}) doesn't match header ({}), rebuilding...",
                    loaded_idx.len(),
                    header.num_records()
                );
                drop(file); // Release file handle before rebuilding
                let idx = build_index_all_records(tpa_path)?;
                idx.save(&idx_path)?;
                debug!("Index saved to {}", idx_path);
                // Reopen file after rebuilding
                file = File::open(tpa_path)?;
                idx
            } else {
                loaded_idx
            }
        } else {
            info!("No index found, building...");
            drop(file); // Release file handle before rebuilding
            let idx = build_index_all_records(tpa_path)?;
            idx.save(&idx_path)?;
            debug!("Index saved to {}", idx_path);
            // Reopen file after rebuilding
            file = File::open(tpa_path)?;
            idx
        };

        // Verify index type
        if index.index_type() != IndexType::VirtualPosition {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "All-records mode requires virtual position index, found {:?}",
                    index.index_type()
                ),
            ));
        }

        let bgzf_start = index.bgzf_section_start();

        // Create BGZF reader starting at records section
        file.seek(SeekFrom::Start(bgzf_start))?;
        let buf_reader = BufReader::new(file);
        let bgzf_reader = bgzf::io::Reader::new(buf_reader);

        let string_table = StringTable::new();

        Ok(Self {
            file: None,
            bgzf_reader: Some(bgzf_reader),
            index,
            header,
            string_table,
            string_table_pos,
            tpa_path: Some(tpa_path.to_string()), // Need path to reopen for string table
            bgzf_section_start: bgzf_start,
        })
    }

    /// Get number of records
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if reader is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get header information
    pub fn header(&self) -> &TpaHeader {
        &self.header
    }

    /// Check if this reader is in all-records mode
    pub fn is_all_records_mode(&self) -> bool {
        self.bgzf_reader.is_some()
    }

    /// Get BGZF section start offset (0 for per-record mode)
    pub fn bgzf_section_start(&self) -> u64 {
        self.bgzf_section_start
    }

    /// Get string table (loads on first access if needed)
    pub fn string_table(&mut self) -> io::Result<&StringTable> {
        if self.string_table.is_empty() {
            self.load_string_table()?;
        }
        Ok(&self.string_table)
    }

    /// Load string table (call this if you need sequence names)
    pub fn load_string_table(&mut self) -> io::Result<()> {
        if !self.string_table.is_empty() {
            return Ok(());
        }

        // String table is at raw file offset (both per-record and all-records modes)
        if let Some(ref mut file) = self.file {
            // Per-record mode: use existing file handle
            file.seek(SeekFrom::Start(self.string_table_pos))?;
            self.string_table = StringTable::read(file, self.header.num_strings())?;
        } else if let Some(ref path) = self.tpa_path {
            // All-records mode: reopen file to read plain bytes
            let mut file = File::open(path)?;
            file.seek(SeekFrom::Start(self.string_table_pos))?;
            self.string_table = StringTable::read(&mut file, self.header.num_strings())?;
        }
        Ok(())
    }

    /// Get immutable reference to string table (must be loaded first with load_string_table)
    pub fn string_table_ref(&self) -> &StringTable {
        &self.string_table
    }

    /// Get full alignment record by ID - O(1) random access
    pub fn get_alignment_record(&mut self, record_id: u64) -> io::Result<AlignmentRecord> {
        let position = self
            .index
            .get_position(record_id)
            .ok_or_else(|| err_out_of_bounds(record_id))?;

        let (first_strategy, second_strategy) = self.header.strategies()?;

        if let Some(ref mut bgzf) = self.bgzf_reader {
            // All-records mode: seek to virtual position
            let vpos = bgzf::VirtualPosition::from(position);
            bgzf.seek(vpos)?;
            read_record(
                bgzf,
                first_strategy,
                second_strategy,
                self.header.tracepoint_type,
                self.header.first_layer,
                self.header.second_layer,
            )
        } else if let Some(ref mut file) = self.file {
            // Per-record mode: seek to raw offset
            file.seek(SeekFrom::Start(position))?;
            read_record(
                file,
                first_strategy,
                second_strategy,
                self.header.tracepoint_type,
                self.header.first_layer,
                self.header.second_layer,
            )
        } else {
            Err(err_not_initialized())
        }
    }

    /// Get alignment record by file offset/virtual position (for impg compatibility)
    pub fn get_alignment_record_at_offset(&mut self, position: u64) -> io::Result<AlignmentRecord> {
        let (first_strategy, second_strategy) = self.header.strategies()?;

        if let Some(ref mut bgzf) = self.bgzf_reader {
            // All-records mode: position is a virtual position
            let vpos = bgzf::VirtualPosition::from(position);
            bgzf.seek(vpos)?;
            read_record(
                bgzf,
                first_strategy,
                second_strategy,
                self.header.tracepoint_type,
                self.header.first_layer,
                self.header.second_layer,
            )
        } else if let Some(ref mut file) = self.file {
            // Per-record mode: position is raw file offset
            file.seek(SeekFrom::Start(position))?;
            read_record(
                file,
                first_strategy,
                second_strategy,
                self.header.tracepoint_type,
                self.header.first_layer,
                self.header.second_layer,
            )
        } else {
            Err(err_not_initialized())
        }
    }

    /// Get tracepoints by record ID
    pub fn get_tracepoints(
        &mut self,
        record_id: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u32)> {
        let tracepoint_offset = self.get_tracepoint_offset(record_id)?;
        self.get_tracepoints_at_offset(tracepoint_offset)
    }

    /// Get tracepoint offset by record ID
    /// Returns the byte offset/virtual position where tracepoint data starts within the record
    pub fn get_tracepoint_offset(&mut self, record_id: u64) -> io::Result<u64> {
        let record_position = self
            .index
            .get_position(record_id)
            .ok_or_else(|| err_out_of_bounds(record_id))?;

        if let Some(ref mut bgzf) = self.bgzf_reader {
            // All-records mode
            let vpos = bgzf::VirtualPosition::from(record_position);
            bgzf.seek(vpos)?;
            skip_record_header(bgzf)?;
            Ok(u64::from(bgzf.virtual_position()))
        } else if let Some(ref mut file) = self.file {
            // Per-record mode
            file.seek(SeekFrom::Start(record_position))?;
            skip_record_header(file)?;
            file.stream_position()
        } else {
            Err(err_not_initialized())
        }
    }

    /// Get tracepoints by tracepoint offset/virtual position
    /// Seeks directly to tracepoint data within a record
    pub fn get_tracepoints_at_offset(
        &mut self,
        tracepoint_position: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u32)> {
        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        let (first_strategy, second_strategy) = self.header.strategies()?;

        let tracepoints = if let Some(ref mut bgzf) = self.bgzf_reader {
            // All-records mode: seek to virtual position, then read directly
            // (BGZF reader doesn't implement std::io::Seek, only bgzf::io::Seek)
            let vpos = bgzf::VirtualPosition::from(tracepoint_position);
            bgzf.seek(vpos)?;
            crate::binary::read_tracepoints(
                bgzf,
                tp_type,
                first_strategy,
                second_strategy,
                self.header.first_layer,
                self.header.second_layer,
            )?
        } else if let Some(ref mut file) = self.file {
            // Per-record mode
            read_tracepoints_at_offset(
                file,
                tracepoint_position,
                tp_type,
                first_strategy,
                second_strategy,
                self.header.first_layer,
                self.header.second_layer,
            )?
        } else {
            return Err(err_not_initialized());
        };

        Ok((tracepoints, complexity_metric, max_complexity))
    }

    /// Iterator over all records (sequential access)
    pub fn iter_records(&mut self) -> RecordIterator<'_> {
        RecordIterator {
            reader: self,
            current_id: 0,
        }
    }
}

// ============================================================================
// STANDALONE FUNCTIONS (NO TPAREADER OVERHEAD)
// ============================================================================

/// Decode standard tracepoints with explicit strategies for first/second streams.
#[inline]
pub fn read_standard_tracepoints_at_offset_with_strategies<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, usize)>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Standard,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => Ok(tps),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at offset: {:?}", other),
        )),
    }
}

/// Fastest access: decode variable tracepoints directly from file at offset.
/// Requires pre-computed offset and compression strategies from header.
#[inline]
pub fn read_variable_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Variable,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Variable(tps) => Ok(tps),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at offset: {:?}", other),
        )),
    }
}

/// Fastest access: decode mixed tracepoints directly from file at offset.
/// Requires pre-computed offset and compression strategies from header.
#[inline]
pub fn read_mixed_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<MixedRepresentation>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Mixed,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Mixed(items) => Ok(items),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at offset: {:?}", other),
        )),
    }
}

// ============================================================================
// BGZF-aware standalone functions for Mode B benchmark with all-records mode
// ============================================================================

/// Decode standard tracepoints from a BGZF reader at a virtual position.
/// This is the BGZF equivalent of read_standard_tracepoints_at_offset_with_strategies.
#[inline]
pub fn read_standard_tracepoints_at_vpos(
    bgzf_reader: &mut bgzf::io::Reader<std::fs::File>,
    vpos: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, usize)>> {
    bgzf_reader.seek(bgzf::VirtualPosition::from(vpos))?;
    let data = read_tracepoints(
        bgzf_reader,
        TracepointType::Standard,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => Ok(tps),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at vpos: {:?}", other),
        )),
    }
}

/// Decode variable tracepoints from a BGZF reader at a virtual position.
#[inline]
pub fn read_variable_tracepoints_at_vpos(
    bgzf_reader: &mut bgzf::io::Reader<std::fs::File>,
    vpos: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    bgzf_reader.seek(bgzf::VirtualPosition::from(vpos))?;
    let data = read_tracepoints(
        bgzf_reader,
        TracepointType::Variable,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Variable(tps) => Ok(tps),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at vpos: {:?}", other),
        )),
    }
}

/// Decode mixed tracepoints from a BGZF reader at a virtual position.
#[inline]
pub fn read_mixed_tracepoints_at_vpos(
    bgzf_reader: &mut bgzf::io::Reader<std::fs::File>,
    vpos: u64,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<MixedRepresentation>> {
    bgzf_reader.seek(bgzf::VirtualPosition::from(vpos))?;
    let data = read_tracepoints(
        bgzf_reader,
        TracepointType::Mixed,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;
    match data {
        TracepointData::Mixed(items) => Ok(items),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at vpos: {:?}", other),
        )),
    }
}

pub struct RecordIterator<'a> {
    reader: &'a mut TpaReader,
    current_id: u64,
}

impl<'a> Iterator for RecordIterator<'a> {
    type Item = io::Result<AlignmentRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id >= self.reader.len() as u64 {
            return None;
        }

        let result = self.reader.get_alignment_record(self.current_id);
        self.current_id += 1;
        Some(result)
    }
}
