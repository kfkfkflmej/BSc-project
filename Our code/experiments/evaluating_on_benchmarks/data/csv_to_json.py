"""Convert CSV columns to JSON with column keys and value arrays."""

import csv
import json
import sys
from pathlib import Path


def csv_to_json(csv_file, json_file=None):
    """
    Convert CSV file to JSON format where each column becomes a key with an array of values.
    
    Args:
        csv_file: Path to input CSV file
        json_file: Path to output JSON file (default: input filename with .json extension)
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return False
    
    if json_file is None:
        json_file = csv_path.with_suffix('.json')
    
    try:
        # Read CSV and organize by columns
        data = {}
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Initialize column arrays
            for fieldname in fieldnames:
                data[fieldname] = []
            
            # Populate arrays
            for row in reader:
                for fieldname in fieldnames:
                    data[fieldname].append(row[fieldname])
        
        # Write to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully converted {csv_file} to {json_file}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def process_all_csvs(directory=None):
    """
    Process all CSV files in a directory.
    
    Args:
        directory: Path to directory containing CSV files (default: current directory)
    """
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
    
    print(f"Found {len(csv_files)} CSV file(s) in {directory}\n")
    
    success_count = 0
    for csv_file in sorted(csv_files):
        if csv_to_json(csv_file):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(csv_files)} files processed successfully")
    return success_count == len(csv_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: process all CSVs in current directory
        success = process_all_csvs()
    elif sys.argv[1] == "--dir":
        # Process all CSVs in specified directory
        directory = sys.argv[2] if len(sys.argv) > 2 else None
        success = process_all_csvs(directory)
    else:
        # Process single file
        csv_file = sys.argv[1]
        json_file = sys.argv[2] if len(sys.argv) > 2 else None
        success = csv_to_json(csv_file, json_file)
    
    sys.exit(0 if success else 1)
