//! Advanced hybrid compression strategies
//!
//! This module implements advanced compression techniques combining multiple approaches:
//! - FastPFOR: Patched Frame-of-Reference with exceptions
//! - Cascaded: Multi-level compression chains
//! - Simple8bFull: Complete Simple8b with all packing modes
//! - SelectiveRLE: RLE preprocessing with bitmap positions

use crate::utils::*;
use std::io::{self, Read};

/// FastPFOR: Frame-of-Reference with patched exceptions
/// Groups integers into 128-value blocks, finding optimal bit-width that captures 95-98% of values
/// Stores exceptions separately for better compression
pub fn encode_fastpfor(vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(vals.len() * 2);

    // Process in 128-value blocks
    for chunk in vals.chunks(128) {
        // Find min value for this block
        let min_val = *chunk.iter().min().unwrap_or(&0);

        // Calculate offsets
        let offsets: Vec<u64> = chunk.iter().map(|&v| v - min_val).collect();

        // Find optimal bit-width (captures 95% of values)
        let mut sorted_offsets = offsets.clone();
        sorted_offsets.sort_unstable();
        let p95_idx = (sorted_offsets.len() as f64 * 0.95) as usize;
        let p95_val = sorted_offsets.get(p95_idx).copied().unwrap_or(0);

        let base_bits = if p95_val == 0 { 1 } else { 64 - p95_val.leading_zeros() };
        let base_bits = base_bits.min(32) as u8; // Cap at 32 bits

        // Find exceptions (values that don't fit in base_bits)
        let threshold = if base_bits >= 64 { u64::MAX } else { (1u64 << base_bits) - 1 };
        let mut exceptions = Vec::new();

        for (idx, &offset) in offsets.iter().enumerate() {
            if offset > threshold {
                exceptions.push((idx as u16, offset - threshold - 1));
            }
        }

        // Write block header
        write_varint(&mut buf, chunk.len() as u64)?;  // Block size
        write_varint(&mut buf, min_val)?;
        buf.push(base_bits);
        write_varint(&mut buf, exceptions.len() as u64)?;

        // Write base values (bit-packed)
        for &offset in &offsets {
            let base_val = offset.min(threshold);
            match base_bits {
                1..=8 => buf.push(base_val as u8),
                9..=16 => {
                    buf.push((base_val & 0xFF) as u8);
                    buf.push(((base_val >> 8) & 0xFF) as u8);
                },
                _ => { write_varint(&mut buf, base_val)?; },
            }
        }

        // Write exceptions
        for (idx, extra) in exceptions {
            buf.push((idx & 0xFF) as u8);
            buf.push(((idx >> 8) & 0xFF) as u8);
            write_varint(&mut buf, extra)?;
        }
    }

    Ok(buf)
}

pub fn decode_fastpfor(data: &[u8]) -> io::Result<Vec<u64>> {
    let mut result = Vec::new();
    let mut reader = &data[..];

    while !reader.is_empty() {
        // Read block header
        let block_size = read_varint(&mut reader)? as usize;  // Block size

        if reader.is_empty() { break; }
        let min_val = read_varint(&mut reader)?;

        if reader.is_empty() { break; }
        let mut base_bits_buf = [0u8; 1];
        reader.read_exact(&mut base_bits_buf)?;
        let base_bits = base_bits_buf[0];

        let exception_count = read_varint(&mut reader)?;

        let threshold = if base_bits >= 64 { u64::MAX } else { (1u64 << base_bits) - 1 };

        // Read base values
        let mut block_vals = Vec::with_capacity(block_size);
        for _ in 0..block_size {
            if reader.is_empty() { break; }

            let base_val = match base_bits {
                1..=8 => {
                    let mut buf = [0u8; 1];
                    reader.read_exact(&mut buf)?;
                    buf[0] as u64
                },
                9..=16 => {
                    let mut buf = [0u8; 2];
                    reader.read_exact(&mut buf)?;
                    buf[0] as u64 | ((buf[1] as u64) << 8)
                },
                _ => {
                    read_varint(&mut reader)?
                }
            };

            block_vals.push(min_val + base_val);
        }

        // Apply exceptions
        for _ in 0..exception_count {
            if reader.len() < 2 { break; }
            let mut idx_buf = [0u8; 2];
            reader.read_exact(&mut idx_buf)?;
            let idx = idx_buf[0] as usize | ((idx_buf[1] as usize) << 8);

            let extra = read_varint(&mut reader)?;

            if idx < block_vals.len() {
                block_vals[idx] = block_vals[idx] + extra + 1;
            }
        }

        result.extend(block_vals);
    }

    Ok(result)
}

/// Cascaded compression: Dictionary → RLE → base compression
/// Analyzes data and applies optimal cascade chain
pub fn encode_cascaded(vals: &[u64]) -> io::Result<Vec<u8>> {
    // Step 1: Analyze cardinality
    use std::collections::HashMap;
    let mut freq: HashMap<u64, usize> = HashMap::new();
    for &v in vals {
        *freq.entry(v).or_insert(0) += 1;
    }

    let cardinality = freq.len();
    let cardinality_ratio = cardinality as f64 / vals.len() as f64;

    let mut buf = Vec::new();

    if cardinality_ratio < 0.01 {
        // Very low cardinality: Dictionary → RLE
        buf.push(1); // Mode 1

        // Build dictionary of top values
        let mut freq_vec: Vec<_> = freq.into_iter().collect();
        freq_vec.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        let dict_size = freq_vec.len().min(256);

        write_varint(&mut buf, dict_size as u64)?;
        for i in 0..dict_size {
            write_varint(&mut buf, freq_vec[i].0)?;
        }

        // Create reverse mapping
        let dict_map: HashMap<u64, u8> = freq_vec.iter().take(dict_size)
            .enumerate()
            .map(|(i, &(val, _))| (val, i as u8))
            .collect();

        // Encode as dictionary indices with RLE
        let mut codes = Vec::new();
        for &v in vals {
            codes.push(*dict_map.get(&v).unwrap_or(&0));
        }

        // Simple RLE on codes
        let mut i = 0;
        while i < codes.len() {
            let val = codes[i];
            let mut run = 1;
            while i + run < codes.len() && codes[i + run] == val && run < 255 {
                run += 1;
            }

            buf.push(val);
            buf.push(run as u8);
            i += run;
        }
    } else {
        // Higher cardinality: use delta encoding
        buf.push(2); // Mode 2

        // Delta encode
        if vals.is_empty() {
            return Ok(buf);
        }

        write_varint(&mut buf, vals[0])?;
        for i in 1..vals.len() {
            // Calculate delta as signed value
            let delta = (vals[i] as i64).wrapping_sub(vals[i - 1] as i64);
            // Zigzag encode: (n << 1) ^ (n >> 63)
            let zigzag = ((delta << 1) ^ (delta >> 63)) as u64;
            write_varint(&mut buf, zigzag)?;
        }
    }

    Ok(buf)
}

pub fn decode_cascaded(data: &[u8]) -> io::Result<Vec<u64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mode = data[0];
    let mut reader = &data[1..];
    let mut result = Vec::new();

    match mode {
        1 => {
            // Dictionary + RLE mode
            let dict_size = read_varint(&mut reader)?;

            let mut dictionary = Vec::new();
            for _ in 0..dict_size {
                dictionary.push(read_varint(&mut reader)?);
            }

            // Decode RLE
            while reader.len() >= 2 {
                let mut val_run_buf = [0u8; 2];
                reader.read_exact(&mut val_run_buf)?;
                let val_idx = val_run_buf[0] as usize;
                let run = val_run_buf[1] as usize;

                let val = dictionary.get(val_idx).copied().unwrap_or(0);
                for _ in 0..run {
                    result.push(val);
                }
            }
        },
        2 => {
            // Delta mode
            if reader.is_empty() {
                return Ok(result);
            }

            let first = read_varint(&mut reader)?;
            result.push(first);

            while !reader.is_empty() {
                let zigzag = read_varint(&mut reader)?;
                // Correct zigzag decode: (n >> 1) ^ -(n & 1)
                let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                let prev = *result.last().unwrap();
                result.push((prev as i64).wrapping_add(delta) as u64);
            }
        },
        _ => {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown cascaded mode"));
        }
    }

    Ok(result)
}

/// SelectiveRLE: Detects runs of 8+ identical values before compression
pub fn encode_selective_rle(vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut i = 0;

    while i < vals.len() {
        // Check for run
        let val = vals[i];
        let mut run_len = 1;
        while i + run_len < vals.len() && vals[i + run_len] == val && run_len < 65535 {
            run_len += 1;
        }

        if run_len >= 8 {
            // Encode as run: marker (0) + value + run_len
            buf.push(0); // RLE marker
            write_varint(&mut buf, val)?;
            write_varint(&mut buf, run_len as u64)?;
            i += run_len;
        } else {
            // Encode as single value: marker (1) + value
            buf.push(1); // Single value marker
            write_varint(&mut buf, val)?;
            i += 1;
        }
    }

    Ok(buf)
}

pub fn decode_selective_rle(data: &[u8]) -> io::Result<Vec<u64>> {
    let mut result = Vec::new();
    let mut reader = &data[..];

    while !reader.is_empty() {
        // Read marker byte
        let mut marker = [0u8; 1];
        reader.read_exact(&mut marker)?;

        if marker[0] == 0 {
            // RLE run: marker (0) + value + run_len
            let val = read_varint(&mut reader)?;
            let run_len = read_varint(&mut reader)?;

            for _ in 0..run_len {
                result.push(val);
            }
        } else if marker[0] == 1 {
            // Single value: marker (1) + value
            let val = read_varint(&mut reader)?;
            result.push(val);
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Invalid SelectiveRLE marker: {}", marker[0])));
        }
    }

    Ok(result)
}

/// Simple8bFull: Enhanced Simple8 with more packing modes
/// Includes additional selectors for better compression
pub fn encode_simple8b_full(vals: &[u64]) -> io::Result<Vec<u64>> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < vals.len() {
        // Find best selector for next values
        let remaining = vals.len() - i;

        // Try different packing modes (selector determines how many ints and how many bits each)
        // Selector 0: 60 1-bit values
        // Selector 1: 30 2-bit values
        // Selector 2: 20 3-bit values
        // Selector 3: 15 4-bit values
        // Selector 4: 12 5-bit values
        // Selector 5: 10 6-bit values
        // Selector 6: 8 7-bit values
        // Selector 7: 7 8-bit values
        // Selector 8: 6 10-bit values
        // Selector 9: 5 12-bit values
        // Selector 10: 4 15-bit values
        // Selector 11: 3 20-bit values
        // Selector 12: 2 30-bit values
        // Selector 13: 1 60-bit value
        // Selector 14: RLE (value + count)
        // Selector 15: Uncompressed

        let modes = [
            (60, 1), (30, 2), (20, 3), (15, 4), (12, 5), (10, 6),
            (8, 7), (7, 8), (6, 10), (5, 12), (4, 15), (3, 20), (2, 30), (1, 60)
        ];

        let mut best_selector = 15u64; // Default: uncompressed
        let mut best_count = 1;

        for (selector, (count, bits)) in modes.iter().enumerate() {
            if *count > remaining { continue; }

            let max_val = if *bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let can_pack = vals[i..i+count].iter().all(|&v| v <= max_val);

            if can_pack {
                best_selector = selector as u64;
                best_count = *count;
                break;
            }
        }

        // Pack word: selector in top 4 bits, data in remaining 60 bits
        let mut word = best_selector << 60;

        if best_selector < 14 {
            let (count, bits) = modes[best_selector as usize];
            for j in 0..count.min(remaining) {
                word |= vals[i + j] << (j * bits);
            }
            i += count.min(remaining);
        } else {
            // Uncompressed: store one value
            word |= vals[i] & ((1u64 << 60) - 1);
            i += 1;
        }

        result.push(word);
    }

    Ok(result)
}

pub fn decode_simple8b_full(words: &[u64]) -> io::Result<Vec<u64>> {
    let mut result = Vec::new();

    let modes = [
        (60, 1), (30, 2), (20, 3), (15, 4), (12, 5), (10, 6),
        (8, 7), (7, 8), (6, 10), (5, 12), (4, 15), (3, 20), (2, 30), (1, 60)
    ];

    for &word in words {
        let selector = word >> 60;
        let data = word & ((1u64 << 60) - 1);

        if selector < 14 {
            let (count, bits) = modes[selector as usize];
            let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };

            for j in 0..count {
                let val = (data >> (j * bits)) & mask;
                result.push(val);
            }
        } else {
            // Uncompressed
            result.push(data);
        }
    }

    Ok(result)
}
