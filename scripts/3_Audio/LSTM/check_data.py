#!/usr/bin/env python3
"""
Quick diagnostic script to check what data files exist and what phases/conditions are available.
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Import config
from utils.config import PROCESSED_AUDIO_DIR, MERGED_METADATA_CSV

def check_available_data():
    """Check what data files exist and what phases/conditions are available."""
    
    processed_root = Path(PROCESSED_AUDIO_DIR)
    metadata_path = Path(MERGED_METADATA_CSV)
    
    print("="*80)
    print("DATA AVAILABILITY CHECK")
    print("="*80)
    
    # Check if directories exist
    print(f"\n1. Checking paths...")
    print(f"   Processed audio dir: {processed_root}")
    print(f"   Exists: {processed_root.exists()}")
    print(f"   Metadata CSV: {metadata_path}")
    print(f"   Exists: {metadata_path.exists()}")
    
    if not processed_root.exists():
        print(f"\nERROR: Processed audio directory not found: {processed_root}")
        return
    
    if not metadata_path.exists():
        print(f"\nERROR: Metadata CSV not found: {metadata_path}")
        return
    
    # Load metadata
    print(f"\n2. Loading metadata...")
    meta = pd.read_csv(metadata_path)
    print(f"   Total participants in metadata: {len(meta)}")
    print(f"   Available conditions: {sorted(meta['condition'].unique().tolist())}")
    
    # Check what subject directories exist
    print(f"\n3. Checking subject directories...")
    subject_dirs = [d for d in processed_root.iterdir() if d.is_dir()]
    print(f"   Found {len(subject_dirs)} subject directories")
    
    # Analyze filenames
    print(f"\n4. Analyzing CSV filenames...")
    all_phases = set()
    all_conditions = set()
    phase_condition_counts = defaultdict(int)
    subject_file_counts = defaultdict(int)
    
    for subj_dir in subject_dirs[:20]:  # Check first 20 subjects
        csv_files = list(subj_dir.glob("*.csv"))
        subject_file_counts[subj_dir.name] = len(csv_files)
        
        for csv_file in csv_files:
            filename = csv_file.name
            filename_lower = filename.lower()
            
            # Parse: ID_diagnose_condition_phase_aufgabe.csv
            # Phase can contain underscores (training_pos, training_neg)
            parts = filename_lower.replace('.csv', '').split('_')
            
            if len(parts) >= 5:  # Need at least: ID, diagnose, condition, phase, aufgabe
                file_condition = parts[2] if len(parts) > 2 else None
                file_phase = '_'.join(parts[3:-1])  # Everything between condition and aufgabe
                
                if file_condition:
                    all_conditions.add(file_condition)
                if file_phase:
                    all_phases.add(file_phase)
                
                if file_condition and file_phase:
                    phase_condition_counts[(file_condition, file_phase)] += 1
    
    print(f"\n   Found phases: {sorted(all_phases)}")
    print(f"\n   Found conditions: {sorted(all_conditions)}")
    print(f"\n   Phase-Condition combinations (sample from first 20 subjects):")
    for (cond, phase), count in sorted(phase_condition_counts.items()):
        print(f"     {cond} + {phase}: {count} files")
    
    # Check specific condition
    print(f"\n5. Checking CRADK condition specifically...")
    cradk_subjects = meta[meta['condition'] == 'CRADK']
    print(f"   CRADK subjects in metadata: {len(cradk_subjects)}")
    
    cradk_with_files = 0
    cradk_phases = defaultdict(int)
    for _, row in cradk_subjects.iterrows():
        pid = str(int(row['ID']))
        subj_dir = processed_root / pid
        if subj_dir.exists():
            csv_files = list(subj_dir.glob("*.csv"))
            if csv_files:
                cradk_with_files += 1
                for csv_file in csv_files:
                    filename_lower = csv_file.name.lower()
                    parts = filename_lower.replace('.csv', '').split('_')
                    if len(parts) >= 5:  # Need at least: ID, diagnose, condition, phase, aufgabe
                        file_condition = parts[2] if len(parts) > 2 else None
                        file_phase = '_'.join(parts[3:-1])  # Everything between condition and aufgabe
                        if file_condition == 'cradk' and file_phase:
                            cradk_phases[file_phase] += 1
    
    print(f"   CRADK subjects with files: {cradk_with_files}")
    print(f"   Phases found in CRADK files: {dict(cradk_phases)}")
    
    # Check for training_pos specifically
    print(f"\n6. Checking for 'training_pos' phase in CRADK...")
    training_pos_count = 0
    for _, row in cradk_subjects.iterrows():
        pid = str(int(row['ID']))
        subj_dir = processed_root / pid
        if subj_dir.exists():
            for csv_file in subj_dir.glob("*.csv"):
                filename_lower = csv_file.name.lower()
                parts = filename_lower.replace('.csv', '').split('_')
                if len(parts) >= 5:  # Need at least: ID, diagnose, condition, phase, aufgabe
                    file_condition = parts[2] if len(parts) > 2 else None
                    file_phase = '_'.join(parts[3:-1])  # Everything between condition and aufgabe
                    if file_condition == 'cradk' and file_phase == 'training_pos':
                        training_pos_count += 1
                        if training_pos_count <= 3:  # Show first 3 examples
                            print(f"   Example file: {csv_file.name}")
    
    print(f"   Total 'training_pos' files for CRADK: {training_pos_count}")
    
    if training_pos_count == 0:
        print(f"\n   WARNING: No files found with phase='training_pos' for CRADK!")
        print(f"   Try using phase='all' to see all available data, or check the actual phase names above.")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    if training_pos_count == 0:
        print("  Use --phase all to load all phases, or check the available phases listed above.")
    else:
        print("  Data looks good! The issue might be elsewhere.")
    print("="*80)

if __name__ == '__main__':
    check_available_data()

