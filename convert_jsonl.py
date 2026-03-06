import json
import argparse
import sys
from datetime import timedelta

def parse_time_str(time_str: str) -> timedelta:
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, s = map(int, parts)
    else:
        h, m, s = 0, 0, int(parts[0])
    return timedelta(hours=h, minutes=m, seconds=s)

def convert_jsonl(input_file, output_file=None, offset_str=None):
    base_offset = parse_time_str(offset_str) if offset_str else timedelta()
    out = open(output_file, "w", encoding="utf-8") if output_file else sys.stdout
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Example timestamp format: "[0:00:00 - 0:00:01.170000]"
                    ts_raw = data.get("timestamp", "")
                    text = data.get("text", "")
                    
                    # Extract just the start timestamp
                    start_ts = ""
                    if ts_raw.startswith("[") and "-" in ts_raw:
                        start_ts = ts_raw[1:].split("-")[0].strip()
                        # Optional: truncate microseconds
                        if "." in start_ts:
                            start_ts = start_ts.split(".")[0]
                    
                    if start_ts:
                        ts_delta = parse_time_str(start_ts)
                        total_ts = ts_delta + base_offset
                        out.write(f"[{total_ts}] {text}\n")
                    else:
                        out.write(f"{ts_raw} {text}\n")
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line}", file=sys.stderr)
    finally:
        if output_file:
            out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL transcription to [hh:mm:ss] text")
    parser.add_argument("input", help="Path to input .jsonl file")
    parser.add_argument("-o", "--output", help="Path to output .txt file (optional)", default=None)
    parser.add_argument("--offset", help="Start time offset (e.g., 18:00:00)", default=None)
    args = parser.parse_args()
    
    convert_jsonl(args.input, args.output, args.offset)
