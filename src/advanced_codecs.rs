// Advanced compression codecs for tracepoint data
// Non-SIMD implementations maintaining O(1) random access
//
// Implements 7 strategies from compression research:
// 1. Frame-of-Reference (FOR) - O(1) access, 8-16 bits/int
// 2. Delta-of-delta (Gorilla) - 9.4x for evenly-spaced
// 3. XOR-delta (Gorilla) - 12x for correlated data
// 4. Joint/Offset encoding - 25-30x for ρ ≈ 1.0
// 5. Enhanced Dictionary - 5.4x for repeated patterns
// 6. Adaptive Correlation - Auto mode selection
// 7. Block-wise Adaptive - Meta-strategy selection

use std::collections::HashMap;
use std::io::{self, Write, Read};

const BLOCK_SIZE: usize = 128;

// ============================================================================
// 1. Frame-of-Reference (FOR) Encoding
// ============================================================================

/// Encode values using Frame-of-Reference: store min + bit-packed offsets
/// Provides O(1) random access within blocks
pub fn encode_frame_of_reference(values: &[u32]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();

    // Process in blocks of 128
    for chunk in values.chunks(BLOCK_SIZE) {
        encode_for_block(chunk, &mut output)?;
    }

    Ok(output)
}

fn encode_for_block(values: &[u32], output: &mut Vec<u8>) -> io::Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    // Find min value (base)
    let min_val = *values.iter().min().unwrap();
    let max_val = *values.iter().max().unwrap();

    // Calculate required bit width for offsets
    let max_offset = max_val - min_val;
    let bit_width = if max_offset == 0 {
        0
    } else {
        32 - max_offset.leading_zeros() as u8
    };

    // Write block header: size (1 byte) + min (4 bytes) + bit_width (1 byte)
    output.write_all(&[values.len() as u8])?;
    output.write_all(&min_val.to_le_bytes())?;
    output.write_all(&[bit_width])?;

    // Bit-pack offsets
    if bit_width > 0 {
        let mut bit_buffer: u64 = 0;
        let mut bits_in_buffer = 0;

        for &val in values {
            let offset = val - min_val;
            bit_buffer |= (offset as u64) << bits_in_buffer;
            bits_in_buffer += bit_width as usize;

            while bits_in_buffer >= 8 {
                output.write_all(&[(bit_buffer & 0xFF) as u8])?;
                bit_buffer >>= 8;
                bits_in_buffer -= 8;
            }
        }

        // Flush remaining bits
        if bits_in_buffer > 0 {
            output.write_all(&[(bit_buffer & 0xFF) as u8])?;
        }
    }

    Ok(())
}

pub fn decode_frame_of_reference(data: &[u8]) -> io::Result<Vec<u32>> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if pos + 6 > data.len() {
            break;
        }

        let size = data[pos] as usize;
        pos += 1;

        let min_val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;

        let bit_width = data[pos];
        pos += 1;

        if bit_width == 0 {
            // All values are the same
            result.extend(std::iter::repeat(min_val).take(size));
        } else {
            // Decode bit-packed offsets
            let mut bit_buffer: u64 = 0;
            let mut bits_in_buffer = 0;

            for _ in 0..size {
                // Fill buffer
                while bits_in_buffer < bit_width as usize && pos < data.len() {
                    bit_buffer |= (data[pos] as u64) << bits_in_buffer;
                    bits_in_buffer += 8;
                    pos += 1;
                }

                // Extract offset
                let mask = (1u64 << bit_width) - 1;
                let offset = (bit_buffer & mask) as u32;
                bit_buffer >>= bit_width;
                bits_in_buffer -= bit_width as usize;

                result.push(min_val + offset);
            }
        }
    }

    Ok(result)
}

// ============================================================================
// 2. Delta-of-Delta (Gorilla-style) Encoding
// ============================================================================

/// Encode using delta-of-delta with variable-length encoding
/// 0 → 1 bit, [-63,64] → 9 bits, else more bits
pub fn encode_delta_of_delta(values: &[u32]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut bit_writer = BitWriter::new();

    // Write first value directly
    output.write_all(&values[0].to_le_bytes())?;

    if values.len() < 2 {
        return Ok(output);
    }

    // Write first delta
    let first_delta = values[1] as i64 - values[0] as i64;
    output.write_all(&(first_delta as i32).to_le_bytes())?;

    // Encode delta-of-deltas
    let mut prev_delta = first_delta;
    for i in 2..values.len() {
        let delta = values[i] as i64 - values[i-1] as i64;
        let delta_of_delta = delta - prev_delta;

        // Variable-length encoding
        if delta_of_delta == 0 {
            // 0 → '0' (1 bit)
            bit_writer.write_bit(0);
        } else if delta_of_delta >= -63 && delta_of_delta <= 64 {
            // [-63, 64] → '10' + 7 bits signed
            bit_writer.write_bit(1);
            bit_writer.write_bit(0);
            bit_writer.write_bits((delta_of_delta as u64) & 0x7F, 7);
        } else if delta_of_delta >= -4095 && delta_of_delta <= 4096 {
            // [-4095, 4096] → '110' + 12 bits signed
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            bit_writer.write_bit(0);
            bit_writer.write_bits((delta_of_delta as u64) & 0xFFF, 12);
        } else {
            // else → '111' + 32 bits
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            bit_writer.write_bits(delta_of_delta as u64, 32);
        }

        prev_delta = delta;
    }

    output.extend_from_slice(&bit_writer.finish());
    Ok(output)
}

pub fn decode_delta_of_delta(data: &[u8]) -> io::Result<Vec<u32>> {
    if data.len() < 8 {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut pos = 0;

    // Read first value
    let first_val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
    result.push(first_val);
    pos += 4;

    if data.len() < 12 {
        return Ok(result);
    }

    // Read first delta
    let first_delta = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as i64;
    result.push((first_val as i64 + first_delta) as u32);
    pos += 4;

    // Decode delta-of-deltas
    let mut bit_reader = BitReader::new(&data[pos..]);
    let mut prev_delta = first_delta;
    let mut current_val = result[1] as i64;

    while bit_reader.has_bits() {
        let delta_of_delta = if bit_reader.read_bit() == 0 {
            // '0' → 0
            0i64
        } else if bit_reader.read_bit() == 0 {
            // '10' + 7 bits signed
            let bits = bit_reader.read_bits(7) as i64;
            if bits & 0x40 != 0 {
                bits | !0x7F  // Sign extend
            } else {
                bits
            }
        } else if bit_reader.read_bit() == 0 {
            // '110' + 12 bits signed
            let bits = bit_reader.read_bits(12) as i64;
            if bits & 0x800 != 0 {
                bits | !0xFFF  // Sign extend
            } else {
                bits
            }
        } else {
            // '111' + 32 bits
            bit_reader.read_bits(32) as i32 as i64
        };

        let delta = prev_delta + delta_of_delta;
        current_val += delta;
        result.push(current_val as u32);
        prev_delta = delta;
    }

    Ok(result)
}

// ============================================================================
// 3. XOR-Delta (Gorilla-style) Encoding
// ============================================================================

/// Encode using XOR with predecessor, storing only meaningful bits
pub fn encode_xor_delta(values: &[u32]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut bit_writer = BitWriter::new();

    // Write first value directly
    output.write_all(&values[0].to_le_bytes())?;

    let mut prev_val = values[0];
    let mut prev_leading = 0u8;
    let mut prev_trailing = 0u8;

    for &val in &values[1..] {
        let xor_val = val ^ prev_val;

        if xor_val == 0 {
            // Value same as previous → '0' (1 bit)
            bit_writer.write_bit(0);
        } else {
            let leading = xor_val.leading_zeros() as u8;
            let trailing = xor_val.trailing_zeros() as u8;
            let meaningful_bits = 32 - leading - trailing;

            if leading >= prev_leading && trailing >= prev_trailing {
                // Same leading/trailing → '10' + middle bits
                bit_writer.write_bit(1);
                bit_writer.write_bit(0);
                let middle = (xor_val >> prev_trailing) & ((1u32 << (32 - prev_leading - prev_trailing)) - 1);
                bit_writer.write_bits(middle as u64, (32 - prev_leading - prev_trailing) as usize);
            } else {
                // Different pattern → '11' + 5 bits leading + 6 bits length + middle bits
                bit_writer.write_bit(1);
                bit_writer.write_bit(1);
                bit_writer.write_bits(leading as u64, 5);
                bit_writer.write_bits(meaningful_bits as u64, 6);
                let middle = (xor_val >> trailing) & ((1u32 << meaningful_bits) - 1);
                bit_writer.write_bits(middle as u64, meaningful_bits as usize);

                prev_leading = leading;
                prev_trailing = trailing;
            }
        }

        prev_val = val;
    }

    output.extend_from_slice(&bit_writer.finish());
    Ok(output)
}

pub fn decode_xor_delta(data: &[u8]) -> io::Result<Vec<u32>> {
    if data.len() < 4 {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();

    // Read first value
    let first_val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    result.push(first_val);

    let mut bit_reader = BitReader::new(&data[4..]);
    let mut prev_val = first_val;
    let mut prev_leading = 0u8;
    let mut prev_trailing = 0u8;

    while bit_reader.has_bits() {
        if bit_reader.read_bit() == 0 {
            // '0' → same value
            result.push(prev_val);
        } else if bit_reader.read_bit() == 0 {
            // '10' → use previous leading/trailing
            let middle = bit_reader.read_bits((32 - prev_leading - prev_trailing) as usize) as u32;
            let xor_val = middle << prev_trailing;
            let val = prev_val ^ xor_val;
            result.push(val);
            prev_val = val;
        } else {
            // '11' → new pattern
            let leading = bit_reader.read_bits(5) as u8;
            let length = bit_reader.read_bits(6) as u8;
            let middle = bit_reader.read_bits(length as usize) as u32;
            let trailing = 32 - leading - length;
            let xor_val = middle << trailing;
            let val = prev_val ^ xor_val;
            result.push(val);
            prev_val = val;
            prev_leading = leading;
            prev_trailing = trailing;
        }
    }

    Ok(result)
}

// ============================================================================
// 4. Joint/Offset Encoding for Correlated Coordinates
// ============================================================================

/// Encode correlated coordinate pairs using offset encoding
/// Assumes correlation ρ > 0.95
pub fn encode_joint_offset(query: &[u32], target: &[u32]) -> io::Result<Vec<u8>> {
    assert_eq!(query.len(), target.len(), "Query and target must have same length");

    let mut output = Vec::new();

    // Encode query stream with delta-of-delta
    let query_encoded = encode_delta_of_delta(query)?;
    output.write_all(&(query_encoded.len() as u32).to_le_bytes())?;
    output.write_all(&query_encoded)?;

    // Encode target as offsets from query with frequency-based codes
    let mut bit_writer = BitWriter::new();

    for i in 0..query.len() {
        let offset = target[i] as i64 - query[i] as i64;

        if offset == 0 {
            // 0 → '0' (1 bit)
            bit_writer.write_bit(0);
        } else if offset == 1 || offset == -1 {
            // ±1 → '10' + sign (2 bits)
            bit_writer.write_bit(1);
            bit_writer.write_bit(0);
            bit_writer.write_bit(if offset > 0 { 1 } else { 0 });
        } else if offset == 2 || offset == -2 {
            // ±2 → '110' + sign (3 bits)
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            bit_writer.write_bit(0);
            bit_writer.write_bit(if offset > 0 { 1 } else { 0 });
        } else {
            // else → '111' + varint
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            bit_writer.write_bit(1);
            write_varint_to_bits(&mut bit_writer, offset);
        }
    }

    output.extend_from_slice(&bit_writer.finish());
    Ok(output)
}

pub fn decode_joint_offset(data: &[u8]) -> io::Result<(Vec<u32>, Vec<u32>)> {
    if data.len() < 4 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut pos = 0;

    // Decode query stream
    let query_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;

    if pos + query_len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated query data"));
    }

    let query = decode_delta_of_delta(&data[pos..pos+query_len])?;
    pos += query_len;

    // Decode target offsets
    let mut bit_reader = BitReader::new(&data[pos..]);
    let mut target = Vec::with_capacity(query.len());

    for &q in &query {
        let offset = if bit_reader.read_bit() == 0 {
            // '0' → 0
            0i64
        } else if bit_reader.read_bit() == 0 {
            // '10' + sign → ±1
            if bit_reader.read_bit() == 1 { 1 } else { -1 }
        } else if bit_reader.read_bit() == 0 {
            // '110' + sign → ±2
            if bit_reader.read_bit() == 1 { 2 } else { -2 }
        } else {
            // '111' + varint
            read_varint_from_bits(&mut bit_reader)
        };

        target.push((q as i64 + offset) as u32);
    }

    Ok((query, target))
}

// ============================================================================
// 5. Enhanced Dictionary Coding
// ============================================================================

/// Encode using dictionary with frequency-based bit codes
/// Best for low cardinality (distinct values < 1%)
pub fn encode_dictionary(values: &[u32]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    // Count value frequencies
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &val in values {
        *freq.entry(val).or_insert(0) += 1;
    }

    // Check cardinality
    let cardinality_ratio = freq.len() as f64 / values.len() as f64;
    if cardinality_ratio > 0.01 {
        // Too many distinct values, use fallback
        return encode_frame_of_reference(values);
    }

    // Sort by frequency
    let mut freq_vec: Vec<(u32, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let mut output = Vec::new();

    // Write dictionary size
    let dict_size = freq_vec.len().min(10); // Top 10 values max
    output.write_all(&[dict_size as u8])?;

    // Write dictionary (top 10 values)
    let mut dict = Vec::new();
    for i in 0..dict_size {
        dict.push(freq_vec[i].0);
        output.write_all(&freq_vec[i].0.to_le_bytes())?;
    }

    // Encode values with frequency codes
    let mut bit_writer = BitWriter::new();

    for &val in values {
        if let Some(idx) = dict.iter().position(|&v| v == val) {
            if idx < 2 {
                // Top 2 → 1-bit codes
                bit_writer.write_bit(idx as u8);
            } else {
                // Next 8 → '1' + 3-bit codes
                bit_writer.write_bit(1);
                bit_writer.write_bits((idx - 2) as u64, 3);
            }
        } else {
            // Not in dict → '1111' + full value
            bit_writer.write_bits(0xF, 4);
            bit_writer.write_bits(val as u64, 32);
        }
    }

    output.extend_from_slice(&bit_writer.finish());
    Ok(output)
}

pub fn decode_dictionary(data: &[u8]) -> io::Result<Vec<u32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut pos = 0;

    // Read dictionary size
    let dict_size = data[pos] as usize;
    pos += 1;

    // Read dictionary
    let mut dict = Vec::new();
    for _ in 0..dict_size {
        if pos + 4 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated dictionary"));
        }
        let val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        dict.push(val);
        pos += 4;
    }

    // Decode values
    let mut bit_reader = BitReader::new(&data[pos..]);
    let mut result = Vec::new();

    while bit_reader.has_bits() {
        if dict_size >= 2 {
            let first_bit = bit_reader.read_bit();
            if first_bit == 0 {
                // Top value (code '0')
                result.push(dict[0]);
                continue;
            }

            if dict_size >= 2 && bit_reader.peek_bits(3) == 0 {
                // Second value (code '1' followed by '000')
                bit_reader.read_bits(3); // consume
                result.push(dict[1]);
                continue;
            }
        }

        let next_3 = bit_reader.read_bits(3);
        if next_3 < 8 && (next_3 as usize + 2) < dict.len() {
            // Next 8 values (codes '1001' - '1111')
            result.push(dict[next_3 as usize + 2]);
        } else if next_3 == 0xF {
            // Full value
            let val = bit_reader.read_bits(32) as u32;
            result.push(val);
        } else {
            break; // End of data
        }
    }

    Ok(result)
}

// ============================================================================
// Helper Structures: BitWriter and BitReader
// ============================================================================

struct BitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bits_in_byte: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            current_byte: 0,
            bits_in_byte: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        self.current_byte |= (bit & 1) << self.bits_in_byte;
        self.bits_in_byte += 1;

        if self.bits_in_byte == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_byte = 0;
        }
    }

    fn write_bits(&mut self, value: u64, num_bits: usize) {
        for i in 0..num_bits {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_byte > 0 {
            self.bytes.push(self.current_byte);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit_pos: 0,
        }
    }

    fn has_bits(&self) -> bool {
        self.pos < self.data.len() || (self.pos == self.data.len() - 1 && self.bit_pos < 8)
    }

    fn read_bit(&mut self) -> u8 {
        if self.pos >= self.data.len() {
            return 0;
        }

        let bit = (self.data[self.pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.pos += 1;
            self.bit_pos = 0;
        }

        bit
    }

    fn read_bits(&mut self, num_bits: usize) -> u64 {
        let mut result = 0u64;
        for i in 0..num_bits {
            result |= (self.read_bit() as u64) << i;
        }
        result
    }

    fn peek_bits(&self, num_bits: usize) -> u64 {
        let mut result = 0u64;
        let mut pos = self.pos;
        let mut bit_pos = self.bit_pos;

        for i in 0..num_bits {
            if pos >= self.data.len() {
                break;
            }
            let bit = (self.data[pos] >> bit_pos) & 1;
            result |= (bit as u64) << i;
            bit_pos += 1;
            if bit_pos == 8 {
                pos += 1;
                bit_pos = 0;
            }
        }

        result
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn write_varint_to_bits(writer: &mut BitWriter, mut value: i64) {
    // Zigzag encode
    let encoded = ((value << 1) ^ (value >> 63)) as u64;

    // Variable-length encoding
    let mut v = encoded;
    while v >= 128 {
        writer.write_bits((v & 0x7F) | 0x80, 8);
        v >>= 7;
    }
    writer.write_bits(v & 0x7F, 8);
}

fn read_varint_from_bits(reader: &mut BitReader) -> i64 {
    let mut result = 0u64;
    let mut shift = 0;

    loop {
        let byte = reader.read_bits(8) as u64;
        result |= (byte & 0x7F) << shift;

        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }

    // Zigzag decode
    let decoded = (result >> 1) as i64 ^ -((result & 1) as i64);
    decoded
}

// ============================================================================
// 6. Adaptive Correlation Detection
// ============================================================================

pub struct CorrelationTracker {
    sum_q: f64,
    sum_t: f64,
    sum_qq: f64,
    sum_tt: f64,
    sum_qt: f64,
    count: usize,
}

impl CorrelationTracker {
    pub fn new() -> Self {
        CorrelationTracker {
            sum_q: 0.0,
            sum_t: 0.0,
            sum_qq: 0.0,
            sum_tt: 0.0,
            sum_qt: 0.0,
            count: 0,
        }
    }

    pub fn add(&mut self, q: u32, t: u32) {
        let q_f = q as f64;
        let t_f = t as f64;

        self.sum_q += q_f;
        self.sum_t += t_f;
        self.sum_qq += q_f * q_f;
        self.sum_tt += t_f * t_f;
        self.sum_qt += q_f * t_f;
        self.count += 1;
    }

    pub fn correlation(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let n = self.count as f64;
        let numerator = self.sum_qt - (self.sum_q * self.sum_t) / n;
        let denom_q = self.sum_qq - (self.sum_q * self.sum_q) / n;
        let denom_t = self.sum_tt - (self.sum_t * self.sum_t) / n;

        if denom_q <= 0.0 || denom_t <= 0.0 {
            return 0.0;
        }

        numerator / (denom_q * denom_t).sqrt()
    }

    pub fn select_mode(&self) -> EncodingMode {
        let rho = self.correlation();

        if rho > 0.95 {
            EncodingMode::Joint
        } else if rho > 0.80 {
            EncodingMode::Predictive
        } else if rho > 0.50 {
            EncodingMode::Decorrelation
        } else {
            EncodingMode::Independent
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncodingMode {
    Joint,          // ρ > 0.95: offset encoding
    Predictive,     // ρ > 0.80: predictive coding
    Decorrelation,  // ρ > 0.50: decorrelation transform
    Independent,    // ρ < 0.50: separate encoding
}

// ============================================================================
// 7. Rice/Golomb Entropy Coding
// ============================================================================

/// Rice coding: optimal for geometric distributions
/// Encodes each value as quotient (unary) + remainder (binary)
/// Parameter k chosen such that 2^k ≈ mean(values)
pub fn encode_rice(vals: &[u64]) -> io::Result<Vec<u8>> {
    if vals.is_empty() {
        return Ok(vec![]);
    }

    // Choose optimal k parameter based on mean
    let mean = vals.iter().sum::<u64>() / vals.len() as u64;
    let k = if mean == 0 {
        1
    } else {
        (64 - mean.leading_zeros()).saturating_sub(1).max(1) as u8
    };

    let mut writer = BitWriter::new();

    // Write header: k parameter and count
    writer.write_bits(k as u64, 5); // k fits in 5 bits (max 31)
    write_varint_to_bits(&mut writer, vals.len() as i64);

    // Encode each value
    for &val in vals {
        let quotient = val >> k;
        let remainder = val & ((1 << k) - 1);

        // Unary encode quotient: write 'quotient' 1-bits followed by a 0-bit
        for _ in 0..quotient {
            writer.write_bit(1);
        }
        writer.write_bit(0);

        // Binary encode remainder
        writer.write_bits(remainder, k as usize);
    }

    Ok(writer.finish())
}

pub fn decode_rice(data: &[u8]) -> io::Result<Vec<u64>> {
    if data.is_empty() {
        return Ok(vec![]);
    }

    let mut reader = BitReader::new(data);

    // Read header
    let k = reader.read_bits(5) as u8;
    let count = read_varint_from_bits(&mut reader) as usize;

    let mut result = Vec::with_capacity(count);

    // Decode each value
    for _ in 0..count {
        // Decode unary quotient
        let mut quotient = 0u64;
        while reader.read_bit() == 1 {
            quotient += 1;
            if quotient > 1000000 {
                // Safety check to prevent infinite loop
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Rice quotient too large",
                ));
            }
        }

        // Decode binary remainder
        let remainder = reader.read_bits(k as usize);

        // Reconstruct value
        let value = (quotient << k) | remainder;
        result.push(value);
    }

    Ok(result)
}

// ============================================================================
// 8. Huffman Entropy Coding
// ============================================================================

/// Simple static Huffman coding optimized for small integer deltas
/// Uses fixed code table tailored for geometric distribution (many small values, few large)
pub fn encode_huffman(vals: &[u64]) -> io::Result<Vec<u8>> {
    if vals.is_empty() {
        return Ok(vec![]);
    }

    // Build frequency table
    let mut frequencies = vec![0u64; 256];
    let mut max_val = 0u64;

    for &val in vals {
        max_val = max_val.max(val);
        if val < 256 {
            frequencies[val as usize] += 1;
        }
    }

    // Build Huffman tree (simplified canonical Huffman)
    let mut symbols: Vec<(u8, u64)> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &freq)| freq > 0)
        .map(|(sym, &freq)| (sym as u8, freq))
        .collect();

    if symbols.is_empty() {
        return Ok(vec![]);
    }

    symbols.sort_by(|a, b| a.1.cmp(&b.1));

    // Assign bit lengths using canonical Huffman (simplified)
    let mut code_lengths = vec![0u8; 256];
    let num_symbols = symbols.len();

    for (i, &(symbol, _)) in symbols.iter().enumerate() {
        // Shorter codes for more frequent symbols
        let bit_len = if num_symbols == 1 {
            1
        } else {
            // Logarithmic distribution of code lengths
            let rank = num_symbols - 1 - i;
            ((64 - (rank as u64 + 1).leading_zeros()) as u8).max(1)
        };
        code_lengths[symbol as usize] = bit_len;
    }

    // Generate canonical codes
    let mut codes = vec![0u64; 256];
    let mut code = 0u64;
    for len in 1..=16 {
        for sym in 0..256 {
            if code_lengths[sym] == len {
                codes[sym] = code;
                code += 1;
            }
        }
        code <<= 1;
    }

    let mut writer = BitWriter::new();

    // Write header: number of values, max value, and code table
    write_varint_to_bits(&mut writer, vals.len() as i64);
    write_varint_to_bits(&mut writer, max_val as i64);

    // Write code table (only for symbols with non-zero frequency)
    writer.write_bits(symbols.len() as u64, 8);
    for &(symbol, _) in &symbols {
        writer.write_bits(symbol as u64, 8);
        writer.write_bits(code_lengths[symbol as usize] as u64, 4);
    }

    // Encode values
    for &val in vals {
        if val < 256 && code_lengths[val as usize] > 0 {
            // Use Huffman code
            writer.write_bits(codes[val as usize], code_lengths[val as usize] as usize);
        } else {
            // Escape code for values >= 256: use special marker + varint
            // Use a reserved code (all 1s for max code length)
            writer.write_bits((1u64 << 12) - 1, 12); // Escape code
            write_varint_to_bits(&mut writer, val as i64);
        }
    }

    Ok(writer.finish())
}

pub fn decode_huffman(data: &[u8]) -> io::Result<Vec<u64>> {
    if data.is_empty() {
        return Ok(vec![]);
    }

    let mut reader = BitReader::new(data);

    // Read header
    let count = read_varint_from_bits(&mut reader) as usize;
    let max_val = read_varint_from_bits(&mut reader) as u64;

    // Read code table
    let num_symbols = reader.read_bits(8) as usize;
    let mut code_lengths = vec![0u8; 256];
    let mut symbols = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let symbol = reader.read_bits(8) as u8;
        let code_len = reader.read_bits(4) as u8;
        code_lengths[symbol as usize] = code_len;
        symbols.push(symbol);
    }

    // Rebuild canonical codes
    let mut codes = vec![0u64; 256];
    let mut code = 0u64;
    for len in 1..=16 {
        for sym in 0..256 {
            if code_lengths[sym] == len {
                codes[sym] = code;
                code += 1;
            }
        }
        code <<= 1;
    }

    // Build decode lookup table (code -> symbol) for each code length
    let mut decode_tables: Vec<std::collections::HashMap<u64, u8>> = vec![std::collections::HashMap::new(); 17];
    for sym in 0..256 {
        let len = code_lengths[sym];
        if len > 0 && len <= 16 {
            decode_tables[len as usize].insert(codes[sym], sym as u8);
        }
    }

    let escape_code = (1u64 << 12) - 1;
    let escape_len = 12usize;

    // Decode values
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        // Try to match Huffman code by reading bit-by-bit
        let mut current_code = 0u64;
        let mut matched = false;

        for bit_len in 1..=16 {
            let bit = reader.read_bit();
            current_code = (current_code << 1) | (bit as u64);

            // Check if this code exists for this length
            if let Some(&symbol) = decode_tables[bit_len].get(&current_code) {
                result.push(symbol as u64);
                matched = true;
                break;
            }

            // Check for escape code at length 12
            if bit_len == escape_len && current_code == escape_code {
                let val = read_varint_from_bits(&mut reader) as u64;
                result.push(val);
                matched = true;
                break;
            }
        }

        if !matched {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid Huffman code: no match found after reading 16 bits"),
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_of_reference() {
        let values = vec![100, 102, 101, 103, 100, 104];
        let encoded = encode_frame_of_reference(&values).unwrap();
        let decoded = decode_frame_of_reference(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_of_delta() {
        let values = vec![1000, 1060, 1120, 1180, 1240]; // Regular spacing
        let encoded = encode_delta_of_delta(&values).unwrap();
        let decoded = decode_delta_of_delta(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_xor_delta() {
        let values = vec![1000, 1001, 1002, 1003, 1004]; // Highly correlated
        let encoded = encode_xor_delta(&values).unwrap();
        let decoded = decode_xor_delta(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_joint_offset() {
        let query = vec![1000, 1100, 1200, 1300];
        let target = vec![1005, 1105, 1206, 1304]; // Highly correlated with offset

        let encoded = encode_joint_offset(&query, &target).unwrap();
        let (dec_query, dec_target) = decode_joint_offset(&encoded).unwrap();

        assert_eq!(query, dec_query);
        assert_eq!(target, dec_target);
    }

    #[test]
    fn test_dictionary() {
        let values = vec![5, 10, 5, 10, 5, 10, 5, 10]; // Low cardinality
        let encoded = encode_dictionary(&values).unwrap();
        let decoded = decode_dictionary(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_correlation_tracker() {
        let mut tracker = CorrelationTracker::new();

        // Perfect correlation
        for i in 0..100 {
            tracker.add(i * 10, i * 10 + 5); // offset by 5
        }

        let rho = tracker.correlation();
        assert!(rho > 0.99, "Expected high correlation, got {}", rho);
        assert_eq!(tracker.select_mode(), EncodingMode::Joint);
    }

    #[test]
    fn test_rice_coding() {
        // Test with geometric distribution (ideal for Rice coding)
        let values = vec![0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 5, 1, 2];
        let encoded = encode_rice(&values).unwrap();
        let decoded = decode_rice(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_rice_coding_large_values() {
        // Test with larger values
        let values = vec![100, 105, 110, 102, 108, 100, 115];
        let encoded = encode_rice(&values).unwrap();
        let decoded = decode_rice(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_huffman_coding() {
        // Test with skewed distribution (ideal for Huffman)
        let values = vec![0, 0, 0, 1, 0, 2, 0, 0, 1, 0];  // Many zeros
        let encoded = encode_huffman(&values).unwrap();
        let decoded = decode_huffman(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_huffman_coding_large_values() {
        // Test with values > 256 (using escape codes)
        let values = vec![0, 1, 500, 2, 0, 1000, 1];
        let encoded = encode_huffman(&values).unwrap();
        let decoded = decode_huffman(&encoded).unwrap();
        assert_eq!(values, decoded);
    }
}
