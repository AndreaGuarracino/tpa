//! Utility functions for binary I/O

use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use crate::Distance;

// ============================================================================
// VARINT ENCODING (LEB128)
// ============================================================================

/// Encode an unsigned integer as a varint
pub(crate) fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }
        bytes.push(byte);
        if value == 0 {
            break;
        }
    }
    bytes
}

/// Write a varint to a writer
#[inline]
pub(crate) fn write_varint<W: Write>(writer: &mut W, value: u64) -> io::Result<usize> {
    let bytes = encode_varint(value);
    writer.write_all(&bytes)?;
    Ok(bytes.len())
}

/// Read a varint from a reader
#[inline]
pub(crate) fn read_varint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        let mut byte_buf = [0u8; 1];
        reader.read_exact(&mut byte_buf)?;
        let byte = byte_buf[0];
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Varint too long",
            ));
        }
    }
    Ok(value)
}

// ============================================================================
// DISTANCE SERIALIZATION
// ============================================================================

/// Write Distance to binary format
pub(crate) fn write_distance<W: Write>(writer: &mut W, distance: &Distance) -> io::Result<()> {
    writer.write_all(&[distance.to_u8()])?;
    match distance {
        Distance::Edit => {}
        Distance::GapAffine {
            mismatch,
            gap_opening,
            gap_extension,
        } => {
            writer.write_all(&mismatch.to_le_bytes())?;
            writer.write_all(&gap_opening.to_le_bytes())?;
            writer.write_all(&gap_extension.to_le_bytes())?;
        }
        Distance::GapAffine2p {
            mismatch,
            gap_opening1,
            gap_extension1,
            gap_opening2,
            gap_extension2,
        } => {
            writer.write_all(&mismatch.to_le_bytes())?;
            writer.write_all(&gap_opening1.to_le_bytes())?;
            writer.write_all(&gap_extension1.to_le_bytes())?;
            writer.write_all(&gap_opening2.to_le_bytes())?;
            writer.write_all(&gap_extension2.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read Distance from binary format
pub(crate) fn read_distance<R: Read>(reader: &mut R) -> io::Result<Distance> {
    let mut code = [0u8; 1];
    reader.read_exact(&mut code)?;
    let mut distance = Distance::from_u8(code[0])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    match &mut distance {
        Distance::Edit => {}
        Distance::GapAffine {
            mismatch,
            gap_opening,
            gap_extension,
        } => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *mismatch = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_opening = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_extension = i32::from_le_bytes(buf);
        }
        Distance::GapAffine2p {
            mismatch,
            gap_opening1,
            gap_extension1,
            gap_opening2,
            gap_extension2,
        } => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *mismatch = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_opening1 = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_extension1 = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_opening2 = i32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            *gap_extension2 = i32::from_le_bytes(buf);
        }
    }
    Ok(distance)
}

// ============================================================================
// HELPERS
// ============================================================================

/// Parse string to u64
pub(crate) fn parse_usize(s: &str, field: &str) -> io::Result<u64> {
    s.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid {}: {}", field, s),
        )
    })
}

/// Parse string to u8
pub(crate) fn parse_u8(s: &str, field: &str) -> io::Result<u8> {
    s.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid {}: {}", field, s),
        )
    })
}

/// Open PAF reader (supports stdin, plain, and gzipped files)
pub(crate) fn open_paf_reader(input_path: &str) -> io::Result<Box<dyn BufRead>> {
    if input_path == "-" {
        Ok(Box::new(BufReader::new(io::stdin())))
    } else if input_path.ends_with(".gz") || input_path.ends_with(".bgz") {
        let file = File::open(input_path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to open input file '{}': {}", input_path, e),
            )
        })?;
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        let file = File::open(input_path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to open input file '{}': {}", input_path, e),
            )
        })?;
        Ok(Box::new(BufReader::new(file)))
    }
}
