#!/usr/bin/env python3
"""
Normalize PAF files for comparison by:
1. Rounding float values to 3 decimal places (handles f32 precision noise)
2. Sorting optional fields (after field 12) by tag name

This allows comparison of PAF files where optional field order may differ
(e.g., TPA outputs tp:Z: at the end).
"""
import sys
import re

def normalize_line(line):
    """
    Normalize PAF line by:
    1. Rounding float fields to 3 decimal places with epsilon adjustment
    2. Sorting optional fields (13+) alphabetically by tag name
    """
    line = line.rstrip('\n\r')
    fields = line.split('\t')

    # PAF has 12 mandatory fields (0-11), rest are optional tags
    if len(fields) <= 12:
        # No optional fields, just normalize floats
        return normalize_floats(line) + '\n'

    mandatory = fields[:12]
    optional = fields[12:]

    # Sort optional fields by tag name (first 2 chars before :)
    optional.sort(key=lambda x: x[:2] if len(x) >= 2 else x)

    # Reconstruct line
    normalized_line = '\t'.join(mandatory + optional)

    # Now normalize floats
    return normalize_floats(normalized_line) + '\n'

def normalize_floats(line):
    """Normalize float fields by converting to f32 precision, then rounding to 3 decimals.

    TPA stores floats as f32, so we simulate this by first converting to f32
    (using struct pack/unpack), then rounding to 3 decimal places. This ensures
    both original and decompressed files are compared at the same precision level.
    """
    import struct

    def round_float(match):
        prefix = match.group(1)
        value_str = match.group(2)
        try:
            value = float(value_str)
            # Convert to f32 precision by round-tripping through struct
            f32_value = struct.unpack('f', struct.pack('f', value))[0]
            # Round to 3 decimal places
            rounded = round(f32_value, 3)
            return f"{prefix}{rounded:.3f}"
        except Exception:
            return match.group(0)

    # Match floats with optional leading zero: 0.123, .123, 123, 123.456
    return re.sub(r'(:f:)(\.[0-9]+|[0-9]+\.?[0-9]*)', round_float, line)

def main():
    """Process stdin line by line, normalizing float values."""
    # Use binary mode with explicit UTF-8 to avoid locale issues
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    for line in stdin:
        # Decode from UTF-8, normalize, encode back
        try:
            text = line.decode('utf-8')
            normalized = normalize_line(text)
            stdout.write(normalized.encode('utf-8'))
        except UnicodeDecodeError:
            # If line isn't valid UTF-8, pass through unchanged
            stdout.write(line)

    stdout.flush()

if __name__ == '__main__':
    main()
