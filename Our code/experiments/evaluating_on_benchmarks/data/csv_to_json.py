"""Convert CSV columns to JSON with column keys and value arrays.
Also normalizes MCQ-style fields like stringified dicts and numpy arrays.
"""

import csv
import json
import sys
import ast
from pathlib import Path


def safe_parse_choices(value):
    """
    Convert malformed 'choices' strings into proper Python objects.
    Handles:
    - stringified dicts
    - numpy arrays inside dicts
    - already-clean dicts/lists
    """
    if value is None or value == "":
        return None

    # If it's already structured (rare but possible)
    if isinstance(value, (dict, list)):
        return value

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return None

    # Convert numpy arrays -> lists if present
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if hasattr(v, "tolist"):
                parsed[k] = v.tolist()
        return parsed

    return parsed


def csv_to_json(csv_file, json_file=None):
    """
    Convert CSV file to JSON format where each column becomes a key with an array of values.
    Also fixes malformed MCQ-style 'choices' fields.
    """
    csv_path = Path(csv_file)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return False

    if json_file is None:
        json_file = csv_path.with_suffix('.json')

    try:
        data = {}

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Initialize columns
            for fieldname in fieldnames:
                data[fieldname] = []

            # Populate
            for row in reader:
                for fieldname in fieldnames:
                    value = row[fieldname]

                    # հատուկ handling for MCQ choices
                    if fieldname == "choices":
                        value = safe_parse_choices(value)

                    data[fieldname].append(value)

        # Write JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Successfully converted {csv_file} → {json_file}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def process_all_csvs(directory=None):
    """Process all CSV files in a directory."""
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return False

    csv_files = list(directory.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return False

    print(f"Found {len(csv_files)} CSV file(s)\n")

    success_count = 0
    for csv_file in sorted(csv_files):
        if csv_to_json(csv_file):
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(csv_files)} files OK")
    return success_count == len(csv_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        success = process_all_csvs()
    elif sys.argv[1] == "--dir":
        directory = sys.argv[2] if len(sys.argv) > 2 else None
        success = process_all_csvs(directory)
    else:
        csv_file = sys.argv[1]
        json_file = sys.argv[2] if len(sys.argv) > 2 else None
        success = csv_to_json(csv_file, json_file)

    sys.exit(0 if success else 1)