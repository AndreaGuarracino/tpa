use crate::binary::{read_record, read_tracepoints_at_offset};
use crate::format::{
    AlignmentRecord, BinaryPafHeader, CompressionLayer, CompressionStrategy, StringTable,
    open_with_footer,
};
use crate::index::{build_index, BpafIndex};
use crate::utils::read_varint;
use lib_tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};
use log::{debug, info};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

pub struct BpafReader {
    file: File,
    index: BpafIndex,
    header: BinaryPafHeader,
    string_table: StringTable,
    string_table_pos: u64,
}

impl BpafReader {
    /// Open a BPAF file with index (builds index if .bpaf.idx doesn't exist)
    pub fn open(bpaf_path: &str) -> io::Result<Self> {
        let (file, header, string_table_pos) = open_with_footer(bpaf_path)?;

        let idx_path = format!("{}.idx", bpaf_path);

        let index = if Path::new(&idx_path).exists() {
            debug!("Loading existing index: {}", idx_path);
            BpafIndex::load(&idx_path)?
        } else {
            info!("No index found, building...");
            let idx = build_index(bpaf_path)?;
            idx.save(&idx_path)?;
            debug!("Index saved to {}", idx_path);
            idx
        };

        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
            string_table_pos,
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
    pub fn header(&self) -> &BinaryPafHeader {
        &self.header
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

        self.file.seek(SeekFrom::Start(self.string_table_pos))?;
        self.string_table =
            StringTable::read(&mut self.file, self.header.num_strings())?;
        Ok(())
    }

    /// Get immutable reference to string table (must be loaded first with load_string_table)
    pub fn string_table_ref(&self) -> &StringTable {
        &self.string_table
    }

    /// Get full alignment record by ID - O(1) random access
    pub fn get_alignment_record(&mut self, record_id: u64) -> io::Result<AlignmentRecord> {
        let offset = self.index.get_offset(record_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Record id {} out of bounds", record_id),
            )
        })?;
        self.get_alignment_record_at_offset(offset)
    }

    /// Get alignment record by file offset (for impg compatibility)
    pub fn get_alignment_record_at_offset(&mut self, offset: u64) -> io::Result<AlignmentRecord> {
        self.file.seek(SeekFrom::Start(offset))?;

        // Read with strategy from header
        let strategy = self.header.strategy()?;
        read_record(
            &mut self.file,
            strategy,
            self.header.tracepoint_type,
            self.header.first_layer,
            self.header.second_layer,
        )
    }

    /// Get tracepoints by record ID
    pub fn get_tracepoints(
        &mut self,
        record_id: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u64)> {
        let tracepoint_offset = self.get_tracepoint_offset(record_id)?;
        self.get_tracepoints_at_offset(tracepoint_offset)
    }

    /// Get tracepoint offset by record ID
    /// Returns the byte offset where tracepoint data starts within the record
    pub fn get_tracepoint_offset(&mut self, record_id: u64) -> io::Result<u64> {
        let record_offset = self.index.get_offset(record_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Record id {} out of bounds", record_id),
            )
        })?;
        self.file.seek(SeekFrom::Start(record_offset))?;

        // Skip 7 varints + 2 single bytes to reach tracepoint data
        read_varint(&mut self.file)?; // query_name_id
        read_varint(&mut self.file)?; // query_start
        read_varint(&mut self.file)?; // query_end
        self.file.seek(SeekFrom::Current(1))?; // strand
        read_varint(&mut self.file)?; // target_name_id
        read_varint(&mut self.file)?; // target_start
        read_varint(&mut self.file)?; // target_end
        read_varint(&mut self.file)?; // residue_matches
        read_varint(&mut self.file)?; // alignment_block_len
        self.file.seek(SeekFrom::Current(1))?; // mapping_quality

        self.file.stream_position()
    }

    /// Get tracepoints by tracepoint offset
    /// Seeks directly to tracepoint data within a record
    pub fn get_tracepoints_at_offset(
        &mut self,
        tracepoint_offset: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u64)> {
        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        // Read tracepoints with strategy from header
        let strategy = self.header.strategy()?;
        let tracepoints = read_tracepoints_at_offset(
            &mut self.file,
            tracepoint_offset,
            tp_type,
            strategy,
            self.header.first_layer,
            self.header.second_layer,
        )?;

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
// STANDALONE FUNCTIONS (NO BPAFREADER OVERHEAD)
// ============================================================================

/// Fastest access: decode standard tracepoints directly from file at offset.
/// Requires pre-computed offset and compression strategy.
#[inline]
pub fn read_standard_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, usize)>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Standard,
        strategy,
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
/// Requires pre-computed offset.
#[inline]
pub fn read_variable_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Variable,
        CompressionStrategy::Raw(0), // strategy ignored for variable
        CompressionLayer::Nocomp,
        CompressionLayer::Nocomp,
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
/// Requires pre-computed offset.
#[inline]
pub fn read_mixed_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
) -> io::Result<Vec<MixedRepresentation>> {
    let data = read_tracepoints_at_offset(
        file,
        offset,
        TracepointType::Mixed,
        CompressionStrategy::Raw(0), // strategy ignored for mixed
        CompressionLayer::Nocomp,
        CompressionLayer::Nocomp,
    )?;
    match data {
        TracepointData::Mixed(items) => Ok(items),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unexpected tracepoint type at offset: {:?}", other),
        )),
    }
}

pub struct RecordIterator<'a> {
    reader: &'a mut BpafReader,
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
