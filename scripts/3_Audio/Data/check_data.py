#!/usr/bin/env python3
"""
Script to check CSV files in processed_audio_opensmile folder.

This script:
1. Shows total number of participants
2. Shows number of participants per condition (CR, CRADK, SHAM, ADK)
3. For each participant: number of CSV files and shape [rows, columns] for each file
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def check_data(processed_audio_dir, metadata_csv):
    """
    Check CSV files for all participants.
    
    Args:
        processed_audio_dir: Path to processed_audio_opensmile directory
        metadata_csv: Path to merged_RCT_info.csv with condition information
    """
    processed_dir = Path(processed_audio_dir)
    
    if not processed_dir.exists():
        print(f"Error: Directory not found: {processed_dir}")
        return
    
    # Load metadata to get condition information
    metadata_df = pd.read_csv(metadata_csv)
    # Create mapping from ID to condition (handle zero-padding)
    id_to_condition = {}
    for _, row in metadata_df.iterrows():
        participant_id = str(row['ID']).zfill(3)  # Zero-pad to 3 digits
        id_to_condition[participant_id] = row['condition']
    
    print("=" * 80)
    print("CSV File Check for processed_audio_opensmile")
    print("=" * 80)
    print(f"Scanning directory: {processed_dir}\n")
    
    # Get all participant folders (should be zero-padded IDs)
    participant_folders = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    
    # Count participants per condition
    condition_counts = defaultdict(int)
    for participant_dir in participant_folders:
        participant_id = participant_dir.name
        condition = id_to_condition.get(participant_id, 'Unknown')
        condition_counts[condition] += 1
    
    # Print summary at the start
    print(f"Total number of participants: {len(participant_folders)}\n")
    print("Number of participants per condition:")
    for condition in ['CR', 'CRADK', 'SHAM', 'ADK']:
        count = condition_counts.get(condition, 0)
        print(f"  {condition}: {count}")
    print()
    print("=" * 80)
    print()
    
    # Check each participant
    for participant_dir in participant_folders:
        participant_id = participant_dir.name
        csv_files = sorted(list(participant_dir.glob("*.csv")))
        
        condition = id_to_condition.get(participant_id, 'Unknown')
        
        print(f"Participant {participant_id} (Condition: {condition})")
        print(f"  Number of CSV files: {len(csv_files)}")
        
        if csv_files:
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    rows, cols = df.shape
                    print(f"    [{rows}, {cols}] {csv_file.name}")
                except Exception as e:
                    print(f"    ERROR reading {csv_file.name}: {e}")
        else:
            print("    No CSV files found")
        
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check CSV files in processed_audio_opensmile folder"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/processed_audio_opensmile",
        help="Path to processed_audio_opensmile directory"
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/merged_RCT_info.csv",
        help="Path to merged_RCT_info.csv"
    )
    
    args = parser.parse_args()
    
    check_data(args.processed_dir, args.metadata_csv)
