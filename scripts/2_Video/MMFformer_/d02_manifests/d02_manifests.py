"""
Script to create manifest files for d02 dataset with train/validation/test splits.

This script:
1. Reads the merged RCT info CSV file
2. Creates folder structure: d02_manifests/{All, CR, ADK, CRADK, SHAM}/
3. Splits participants into train (70%), validation (10%), test (20%)
4. Maintains class balance (Depressed/Healthy) in each split
5. Ensures no data leakage (each participant in only one split)
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_training_assignments(path: str) -> dict:
    """
    Load training1_type / training2_type per participant_id from training_assignments.csv.
    Returns a dict: {participant_id: {'training1_type': 'positive'/'negative'/..., 'training2_type': ...}}
    """
    ta_df = pd.read_csv(path)
    mapping: dict[int, dict[str, str]] = {}
    for _, row in ta_df.iterrows():
        try:
            pid = int(row["participant_id"])
        except Exception:
            continue
        mapping[pid] = {
            "training1_type": str(row.get("training1_type", "")).strip().lower(),
            "training2_type": str(row.get("training2_type", "")).strip().lower(),
        }
    return mapping


class Tee:
    """
    A class that writes to both console and file simultaneously.
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def stratified_split(participants_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Split participants into train, validation, and test sets with stratification.
    
    Args:
        participants_df: DataFrame with participant data
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df: Three dataframes with split participants
    """
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        participants_df,
        test_size=(val_ratio + test_ratio),
        stratify=participants_df['Diagnose'] if len(participants_df['Diagnose'].unique()) > 1 else None,
        random_state=random_state
    )
    
    # Second split: separate val from test
    # Adjust the test_size for the second split
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio),
        stratify=temp_df['Diagnose'] if len(temp_df['Diagnose'].unique()) > 1 else None,
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def create_manifests_for_condition(df, condition, output_base_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Create train/validation/test manifest files for a specific condition.
    
    Args:
        df: Full dataframe with all participants
        condition: Condition name (e.g., 'CR', 'ADK', 'CRADK', 'SHAM')
        output_base_dir: Base directory for output files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # Filter participants by condition
    condition_df = df[df['condition'] == condition].copy()
    
    if len(condition_df) == 0:
        print(f"Warning: No participants found for condition {condition}")
        return
    
    print(f"\nProcessing condition: {condition}")
    print(f"  Total participants: {len(condition_df)}")
    total_files_all = condition_df['total_files'].sum() if 'total_files' in condition_df.columns else 0
    print(f"  Total files: {total_files_all}")
    print(f"  Depressed: {len(condition_df[condition_df['Diagnose'] == 'Depressed'])}")
    print(f"  Healthy: {len(condition_df[condition_df['Diagnose'] == 'Healthy'])}")
    
    # Split into train, validation, and test
    train_df, val_df, test_df = stratified_split(
        condition_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create output directory
    output_dir = Path(output_base_dir) / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_path = output_dir / f"{condition}_train.csv"
    val_path = output_dir / f"{condition}_validation.csv"
    test_path = output_dir / f"{condition}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Calculate total files for each split
    train_files = train_df['total_files'].sum() if 'total_files' in train_df.columns else 0
    val_files = val_df['total_files'].sum() if 'total_files' in val_df.columns else 0
    test_files = test_df['total_files'].sum() if 'total_files' in test_df.columns else 0
    
    print(f"  Train: {len(train_df)} participants, {train_files} files (Depressed: {len(train_df[train_df['Diagnose'] == 'Depressed'])}, Healthy: {len(train_df[train_df['Diagnose'] == 'Healthy'])})")
    print(f"  Validation: {len(val_df)} participants, {val_files} files (Depressed: {len(val_df[val_df['Diagnose'] == 'Depressed'])}, Healthy: {len(val_df[val_df['Diagnose'] == 'Healthy'])})")
    print(f"  Test: {len(test_df)} participants, {test_files} files (Depressed: {len(test_df[test_df['Diagnose'] == 'Depressed'])}, Healthy: {len(test_df[test_df['Diagnose'] == 'Healthy'])})")
    
    # Verify no overlap
    train_ids = set(train_df['ID'])
    val_ids = set(val_df['ID'])
    test_ids = set(test_df['ID'])
    assert len(train_ids & val_ids) == 0, "Data leakage detected: train and validation overlap!"
    assert len(train_ids & test_ids) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_ids & test_ids) == 0, "Data leakage detected: validation and test overlap!"
    print(f"  ✓ No data leakage detected")


def create_all_manifests(df, output_base_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Create train/validation/test manifest files for all conditions combined.
    
    Args:
        df: Full dataframe with all participants
        output_base_dir: Base directory for output files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    print(f"\nProcessing condition: All (all conditions combined)")
    print(f"  Total participants: {len(df)}")
    total_files_all = df['total_files'].sum() if 'total_files' in df.columns else 0
    print(f"  Total files: {total_files_all}")
    print(f"  Depressed: {len(df[df['Diagnose'] == 'Depressed'])}")
    print(f"  Healthy: {len(df[df['Diagnose'] == 'Healthy'])}")
    
    # Split all participants into train, validation, and test
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create output directory
    output_dir = Path(output_base_dir) / "All"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_path = output_dir / "All_train.csv"
    val_path = output_dir / "All_validation.csv"
    test_path = output_dir / "All_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Calculate total files for each split
    train_files = train_df['total_files'].sum() if 'total_files' in train_df.columns else 0
    val_files = val_df['total_files'].sum() if 'total_files' in val_df.columns else 0
    test_files = test_df['total_files'].sum() if 'total_files' in test_df.columns else 0
    
    print(f"  Train: {len(train_df)} participants, {train_files} files (Depressed: {len(train_df[train_df['Diagnose'] == 'Depressed'])}, Healthy: {len(train_df[train_df['Diagnose'] == 'Healthy'])})")
    print(f"  Validation: {len(val_df)} participants, {val_files} files (Depressed: {len(val_df[val_df['Diagnose'] == 'Depressed'])}, Healthy: {len(val_df[val_df['Diagnose'] == 'Healthy'])})")
    print(f"  Test: {len(test_df)} participants, {test_files} files (Depressed: {len(test_df[test_df['Diagnose'] == 'Depressed'])}, Healthy: {len(test_df[test_df['Diagnose'] == 'Healthy'])})")
    
    # Verify no overlap
    train_ids = set(train_df['ID'])
    val_ids = set(val_df['ID'])
    test_ids = set(test_df['ID'])
    assert len(train_ids & val_ids) == 0, "Data leakage detected: train and validation overlap!"
    assert len(train_ids & test_ids) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_ids & test_ids) == 0, "Data leakage detected: validation and test overlap!"
    print(f"  ✓ No data leakage detected")


def extract_phase_and_aufgabe_from_filename(filename):
    """
    Extract phase, training number, and aufgabe number from npz filename (case-insensitive).
    
    Args:
        filename: Name of the npz file (e.g., '4_10_training1_neg_aufgabe_10.npz')
    
    Returns:
        tuple: (phase, training_num, aufgabe) where:
               - phase is e.g., 'training_neg', 'training_pos', 'induction1', 'induction2', or 'unknown'
               - training_num is 1 or 2 for trainings, None for inductions or unknown
               - aufgabe is an integer (1 for inductions, extracted number for trainings)
    """
    # Convert to lowercase for case-insensitive matching
    filename_lower = filename.lower()
    
    # Extract aufgabe number
    aufgabe = 1  # Default for inductions
    aufgabe_match = re.search(r'_aufgabe_(\d+)', filename_lower)
    if aufgabe_match:
        aufgabe = int(aufgabe_match.group(1))
    
    # Extract training number and phase
    training_num = None
    if 'training1_neg' in filename_lower:
        return ('training_neg', 1, aufgabe)
    elif 'training2_neg' in filename_lower:
        return ('training_neg', 2, aufgabe)
    elif 'training1_pos' in filename_lower:
        return ('training_pos', 1, aufgabe)
    elif 'training2_pos' in filename_lower:
        return ('training_pos', 2, aufgabe)
    elif 'induction1' in filename_lower:
        return ('induction1', None, 1)  # Always 1 for inductions
    elif 'induction2' in filename_lower:
        return ('induction2', None, 1)  # Always 1 for inductions
    else:
        # Fallback: we only know it is a training but not pos/neg.
        # We try to extract training number and leave polarity resolution
        # to the training_assignments mapping.
        if 'training' in filename_lower:
            train_match = re.search(r'training([12])', filename_lower)
            if train_match:
                training_num = int(train_match.group(1))
            return ('unknown', training_num, aufgabe)
        return ('unknown', None, aufgabe)


def scan_npz_files(npz_folder_path, training_assignments: dict | None = None):
    """
    Scan the npz folder and create a mapping of participant ID to their files.
    
    Args:
        npz_folder_path: Path to the folder containing npz files
    
    Returns:
        files_dict: Dictionary mapping participant_id -> list of (filename, phase, training_num, aufgabe) tuples
    """
    files_dict = {}
    npz_path = Path(npz_folder_path)
    
    if not npz_path.exists():
        print(f"Warning: NPZ folder not found: {npz_folder_path}")
        return files_dict
    
    print(f"\nScanning NPZ files in: {npz_folder_path}")
    
    # Get all .npz files
    npz_files = list(npz_path.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")
    
    for npz_file in npz_files:
        filename = npz_file.name
        # Extract participant ID from filename (format: {ID}_{number}_{phase_info}.npz)
        match = re.match(r'^(\d+)_', filename)
        if match:
            participant_id = int(match.group(1))
            phase, training_num, aufgabe = extract_phase_and_aufgabe_from_filename(filename)

            # If this is a generic training1/training2 without explicit pos/neg,
            # use training_assignments.csv to decide between training_pos / training_neg.
            if (
                phase == 'unknown'
                and training_num in (1, 2)
                and training_assignments is not None
                and participant_id in training_assignments
            ):
                ta = training_assignments[participant_id]
                if training_num == 1:
                    t_type = ta.get("training1_type", "")
                else:
                    t_type = ta.get("training2_type", "")
                t_type = str(t_type).strip().lower()
                if t_type == "positive":
                    phase = "training_pos"
                elif t_type == "negative":
                    phase = "training_neg"

            if participant_id not in files_dict:
                files_dict[participant_id] = []

            files_dict[participant_id].append((filename, phase, training_num, aufgabe))
    
    print(f"Found files for {len(files_dict)} participants")
    return files_dict


def add_file_paths_to_manifests(manifest_base_dir, npz_folder_path, training_assignments: dict | None = None):
    """
    Add path_video, phase, and Aufgabe columns to all manifest CSV files.
    
    This function:
    1. Scans the npz folder to find all files for each participant
    2. For each manifest CSV file, expands rows to include all files for each participant
    3. Adds 'path_video', 'phase', and 'Aufgabe' columns
    
    Args:
        manifest_base_dir: Base directory containing manifest subfolders
        npz_folder_path: Path to the folder containing npz files
    """
    # Scan npz files
    files_dict = scan_npz_files(npz_folder_path, training_assignments=training_assignments)
    
    if not files_dict:
        print("No files found. Skipping manifest updates.")
        return
    
    manifest_base = Path(manifest_base_dir)
    npz_path = Path(npz_folder_path)
    
    # Find all CSV files in subfolders
    csv_files = list(manifest_base.glob("*/*.csv"))
    
    print(f"\nUpdating {len(csv_files)} manifest CSV files...")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # Read the manifest CSV
        df = pd.read_csv(csv_file)
        
        # Create expanded dataframe with one row per file
        expanded_rows = []
        
        for _, row in df.iterrows():
            participant_id = int(row['ID'])
            
            # Find all files for this participant
            if participant_id in files_dict:
                for filename, phase, training_num, aufgabe in files_dict[participant_id]:
                    # Create a new row with file information
                    new_row = row.copy()
                    new_row['path_video'] = str(npz_path / filename)
                    new_row['phase'] = phase
                    new_row['Aufgabe'] = aufgabe
                    
                    expanded_rows.append(new_row)
            else:
                # If no files found, keep the original row with empty columns
                new_row = row.copy()
                new_row['path_video'] = ''
                new_row['phase'] = ''
                new_row['Aufgabe'] = ''
                expanded_rows.append(new_row)
        
        # Create expanded dataframe
        expanded_df = pd.DataFrame(expanded_rows)
        
        # Reorder columns to put new columns at the end
        # Handle case where columns might already exist (if script is run multiple times)
        new_cols = ['path_video', 'phase', 'Aufgabe']
        existing_cols = [col for col in expanded_df.columns if col not in new_cols]
        
        # Ensure all new columns exist
        for col in new_cols:
            if col not in expanded_df.columns:
                expanded_df[col] = ''
        
        # Reorder: existing columns first, then new columns
        expanded_df = expanded_df[existing_cols + new_cols]
        
        # Save updated CSV
        expanded_df.to_csv(csv_file, index=False)
        
        print(f"  Updated: {len(df)} participants -> {len(expanded_df)} rows")
        print(f"  Files added: {len(expanded_df) - len(df)}")


def main():
    """
    Main function to create all manifest files.
    
    Configuration:
    - INPUT_FILE: Path to the merged RCT info CSV file
    - OUTPUT_DIR: Base directory for output manifest files
    - TRAIN_RATIO: Proportion for training set (default: 0.7)
    - VAL_RATIO: Proportion for validation set (default: 0.1)
    - TEST_RATIO: Proportion for test set (default: 0.2)
    - RANDOM_STATE: Random seed for reproducibility (default: 42)
    """
    # ========== CONFIGURATION ==========
    # Path to the merged RCT info CSV file
    INPUT_FILE = '/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests/merged_RCT_info.csv'
    
    # Base directory for output manifest files
    OUTPUT_DIR = '/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests'
    
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    
    # Random seed for reproducibility (change this if you want different splits)
    RANDOM_STATE = 42
    
    # Log file path (will be saved in OUTPUT_DIR)
    LOG_FILE = Path(OUTPUT_DIR) / f"d02_manifest_creation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Path to the folder containing npz files
    NPZ_FOLDER = '/home/vault/empkins/tpD/D02/processed_data/paper_implementation/opendbm_avec2019_npz_down2'

    # Path to training assignments (per-participant training1_type / training2_type)
    TRAINING_ASSIGNMENTS_FILE = '/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/Data/training_assignments.csv'
    
    # Whether to add file paths to manifests (set to True to enable)
    ADD_FILE_PATHS = True
    # ===================================
    
    # Setup logging to both console and file
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout to both console and log file
    tee = Tee(LOG_FILE)
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"Manifest Creation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Log file: {LOG_FILE}")
        print()
        
        # Set random seed for reproducibility
        np.random.seed(RANDOM_STATE)
        
        # Read the merged RCT info file
        print(f"Reading data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)

        # Read training assignments to resolve generic training1/training2 polarity
        print(f"Reading training assignments from: {TRAINING_ASSIGNMENTS_FILE}")
        training_assignments = load_training_assignments(TRAINING_ASSIGNMENTS_FILE)
        
        print(f"\nTotal participants in dataset: {len(df)}")
        print(f"Conditions: {df['condition'].unique()}")
        print(f"Diagnoses: {df['Diagnose'].unique()}")
        
        # Get unique conditions (excluding 'All')
        conditions = sorted(df['condition'].unique())
        print(f"\nConditions to process: {conditions}")
        
        # Create manifests for each condition
        for condition in conditions:
            create_manifests_for_condition(
                df,
                condition,
                OUTPUT_DIR,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO
            )
        
        # Create manifests for all conditions combined
        create_all_manifests(
            df,
            OUTPUT_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO
        )
        
        print(f"\n✓ All manifest files created successfully in: {OUTPUT_DIR}")
        
        # Add file paths and phases to manifests
        if ADD_FILE_PATHS:
            print("\n" + "=" * 80)
            print("Adding file paths, phases, and Aufgabe to manifest files...")
            print("=" * 80)
            add_file_paths_to_manifests(OUTPUT_DIR, NPZ_FOLDER, training_assignments)
            print("\n✓ File paths, phases, and Aufgabe added to all manifest files")
        
        print(f"\n✓ Log file saved to: {LOG_FILE}")
        print("=" * 80)
    
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✓ Log file saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()