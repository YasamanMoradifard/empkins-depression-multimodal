import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

def find_duplicate_path_videos(base_dir):
    """
    Find duplicate path_video values across all CSV files in condition folders.
    
    Args:
        base_dir: Base directory containing condition folders (ADK, All, CR, CRADK, SHAM)
    
    Returns:
        Dictionary with duplicate path_video values and their counts per file
    """
    # Dictionary to track path_video -> {filename: count}
    path_video_tracker = defaultdict(lambda: defaultdict(int))
    
    # List of condition folders
    condition_folders = ['ADK', 'CR', 'CRADK', 'SHAM']
    
    # Process each condition folder
    for condition in condition_folders:
        condition_path = os.path.join(base_dir, condition)
        
        if not os.path.exists(condition_path):
            print(f"Warning: Folder {condition} not found, skipping...")
            continue
        
        # Process train, validation, and test CSV files
        for split in ['train', 'validation', 'test']:
            csv_file = os.path.join(condition_path, f"{condition}_{split}.csv")
            
            if not os.path.exists(csv_file):
                print(f"Warning: File {csv_file} not found, skipping...")
                continue
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Check if path_video column exists
                if 'path_video' not in df.columns:
                    print(f"Warning: 'path_video' column not found in {csv_file}, skipping...")
                    continue
                
                # Count occurrences of each path_video in this file
                path_counts = df['path_video'].value_counts()
                
                # Track each path_video value
                for path_video, count in path_counts.items():
                    filename = os.path.basename(csv_file)
                    path_video_tracker[path_video][filename] = count
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
    
    # Find duplicates (path_video values that appear in multiple files)
    duplicates = {}
    for path_video, file_counts in path_video_tracker.items():
        if len(file_counts) > 1:  # Appears in more than one file
            duplicates[path_video] = dict(file_counts)
    
    return duplicates


def print_duplicate_report(duplicates):
    """
    Print a formatted report of duplicate path_video values.
    """
    if not duplicates:
        print("No duplicate path_video values found across files.")
        return
    
    print(f"\n{'='*80}")
    print(f"DUPLICATE PATH_VIDEO REPORT")
    print(f"{'='*80}")
    print(f"Total number of duplicate path_video values: {len(duplicates)}\n")
    
    # Sort by total occurrences (sum across all files)
    sorted_duplicates = sorted(
        duplicates.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )
    
    for idx, (path_video, file_counts) in enumerate(sorted_duplicates, 1):
        total_count = sum(file_counts.values())
        print(f"\n{idx}. Path: {path_video}")
        print(f"   Total occurrences across all files: {total_count}")
        print(f"   Appears in {len(file_counts)} file(s):")
        for filename, count in sorted(file_counts.items()):
            print(f"      - {filename}: {count} time(s)")


def save_duplicate_report(duplicates, output_file):
    """
    Save duplicate report to a CSV file.
    """
    if not duplicates:
        print("No duplicates to save.")
        return
    
    # Prepare data for DataFrame
    report_data = []
    for path_video, file_counts in duplicates.items():
        row = {'path_video': path_video, 'total_occurrences': sum(file_counts.values()), 
               'num_files': len(file_counts)}
        # Add count for each file
        for filename, count in file_counts.items():
            row[filename] = count
        report_data.append(row)
    
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(output_file, index=False)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    # Base directory containing condition folders
    base_dir = "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests"
    
    print("Scanning CSV files for duplicate path_video values...")
    duplicates = find_duplicate_path_videos(base_dir)
    
    # Print report to console
    print_duplicate_report(duplicates)
    
    # Save report to CSV
    output_file = os.path.join(base_dir, "duplicate_path_videos_report.csv")
    save_duplicate_report(duplicates, output_file)