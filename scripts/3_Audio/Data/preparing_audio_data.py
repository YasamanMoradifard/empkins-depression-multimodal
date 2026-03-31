"""
Comprehensive script to process audio data CSV files.

This script:
1. Reads participant metadata from merged_RCT_info.csv
2. Processes all CSV files for each participant
3. Groups files by phase and aufgabe
4. Removes 'name' and 'frameTime' columns from each CSV
5. Prefixes feature columns with method name (e.g., F0_compare16, F0_egemaps)
6. Concatenates CSV files horizontally (side-by-side) for each phase/aufgabe group
7. Adds metadata columns (ID, diagnose, condition, phase, aufgabe)
8. Removes duplicate columns after concatenation
9. Saves processed files in data/{participant_id}/ folder
"""

import pandas as pd  # type: ignore
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import logging
from datetime import datetime as dt

# Setup logging to both console and file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"runlog_{dt.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging to file: {log_file}")


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse CSV filename to extract phase, aufgabe, method, and timestamps.
    
    Expected formats:
    - Induction: YYYY-MM-DD_HH-MM_Belastungsphase_YYYY-MM-DD_HH-MM_<method>.csv
    - Training: YYYY-MM-DD_HH-MM_Training_<N>_Aufgabe_<M>_YYYY-MM-DD_HH-MM_<method>.csv
    
    Args:
        filename: CSV filename
        
    Returns:
        Dictionary with parsed information:
        - phase_type: 'Belastungsphase' or 'Training'
        - training_number: '1' or '2' for Training files, None for Belastungsphase
        - aufgabe: Aufgabe number as string, or None for Belastungsphase
        - method: opensmile method (compare16, egemaps, gemaps, is09, is13)
        - start_timestamp: First timestamp string
        - end_timestamp: Second timestamp string
    """
    result = {
        'phase_type': None,
        'training_number': None,
        'aufgabe': None,
        'method': None,
        'start_timestamp': None,
        'end_timestamp': None
    }
    
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Extract timestamps (format: YYYY-MM-DD_HH-MM)
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})'
    timestamps = re.findall(timestamp_pattern, name)
    if len(timestamps) >= 2:
        result['start_timestamp'] = timestamps[0]
        result['end_timestamp'] = timestamps[1]
    
    # Check for Belastungsphase
    if 'Belastungsphase' in name:
        result['phase_type'] = 'Belastungsphase'
        result['aufgabe'] = None  # Induction doesn't have aufgabe
    
    # Check for Training
    elif 'Training' in name:
        result['phase_type'] = 'Training'
        # Extract Training number
        training_match = re.search(r'Training_(\d+)', name)
        if training_match:
            result['training_number'] = training_match.group(1)
        
        # Extract Aufgabe number
        aufgabe_match = re.search(r'Aufgabe_(\d+)', name)
        if aufgabe_match:
            result['aufgabe'] = aufgabe_match.group(1)
    
    # Extract method (last part after last underscore)
    parts = name.split('_')
    if len(parts) > 0:
        # Methods: compare16, egemaps, gemaps, is09, is13
        possible_methods = ['compare16', 'egemaps', 'gemaps', 'is09', 'is13', 'is02']
        last_part = parts[-1]
        if last_part in possible_methods:
            result['method'] = last_part
    
    return result


def determine_induction_phase(belastungsphase_files: List[str], metadata: Dict) -> Dict[str, str]:
    """
    Determine which Belastungsphase files belong to induction1 vs induction2.
    
    Groups files by start timestamp - files with same start time belong to same phase.
    Earlier timestamp = induction1, later timestamp = induction2.
    
    Args:
        belastungsphase_files: List of Belastungsphase filenames
        metadata: Dictionary with participant metadata
        
    Returns:
        Dictionary mapping filename to phase ('induction1' or 'induction2')
    """
    file_phases = {}
    
    # Parse timestamps and group by start time
    timestamp_groups = defaultdict(list)
    for filename in belastungsphase_files:
        parsed = parse_filename(filename)
        if parsed['start_timestamp']:
            try:
                # Parse timestamp (use different variable name to avoid conflict with imported dt)
                timestamp_dt = datetime.strptime(parsed['start_timestamp'], '%Y-%m-%d_%H-%M')
                timestamp_groups[timestamp_dt].append(filename)
            except ValueError:
                logger.warning(f"Could not parse timestamp from {filename}")
    
    # Sort timestamps and assign phases
    sorted_timestamps = sorted(timestamp_groups.keys())
    
    if len(sorted_timestamps) == 0:
        logger.warning("No valid timestamps found for Belastungsphase files")
        return file_phases
    
    # Assign phases: first timestamp group = induction1, second = induction2
    for i, timestamp in enumerate(sorted_timestamps):
        phase = 'induction1' if i == 0 else 'induction2'
        for filename in timestamp_groups[timestamp]:
            file_phases[filename] = phase
    
    return file_phases


def determine_training_phase(training_number: str, metadata: Dict) -> str:
    """
    Determine if Training_1 or Training_2 is training_pos or training_neg.
    
    Args:
        training_number: '1' or '2'
        metadata: Dictionary with participant metadata (from merged_RCT_info.csv row)
        
    Returns:
        'training_pos' or 'training_neg'
    """
    if training_number == '1':
        training_type = metadata.get('training1_type', '').lower()
    elif training_number == '2':
        training_type = metadata.get('training2_type', '').lower()
    else:
        logger.warning(f"Unknown training number: {training_number}")
        return 'training_unknown'
    
    if training_type == 'positive':
        return 'training_pos'
    elif training_type == 'negative':
        return 'training_neg'
    else:
        logger.warning(f"Unknown training type: {training_type} for training {training_number}")
        return f'training_{training_type}'


def process_participant_files(base_path: str, participant_id: int, metadata: Dict, output_base_dir: str = "data", save_to_disk: bool = False):
    """
    Process all CSV files for a single participant.
    
    Args:
        base_path: Base directory containing participant folders
        participant_id: Participant ID (integer)
        metadata: Dictionary with participant metadata
        output_base_dir: Base directory for output files (only used if save_to_disk=True)
        save_to_disk: If True, save processed dataframes to CSV files. If False, return them in memory.
    
    Returns:
        List of dictionaries, each containing:
        - 'dataframe': Processed pandas DataFrame
        - 'phase': Phase name (e.g., 'training_pos', 'induction1')
        - 'aufgabe': Aufgabe number
        - 'participant_id': Participant ID
    """
    # Zero-pad participant ID
    padded_id = str(participant_id).zfill(3)
    
    # Construct paths
    participant_dir = Path(base_path) / padded_id / "timeseries_opensmile_features_vad"
    output_dir = Path(output_base_dir) / padded_id
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Get all CSV files
    csv_files = list(participant_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files for participant {participant_id}")

    
    # Group files by phase and aufgabe
    file_groups = defaultdict(list)
    
    # First pass: parse all files and determine induction phases
    belastungsphase_files = []
    for csv_file in csv_files:
        parsed = parse_filename(csv_file.name)
        if parsed['phase_type'] == 'Belastungsphase':
            belastungsphase_files.append(csv_file.name)
    
    # Determine induction phases
    induction_phases = {}
    if belastungsphase_files:
        induction_phases = determine_induction_phase(belastungsphase_files, metadata)
    
    # Second pass: group all files
    for csv_file in csv_files:
        parsed = parse_filename(csv_file.name)
        
        if parsed['phase_type'] == 'Belastungsphase':
            phase = induction_phases.get(csv_file.name, 'induction_unknown')
            group_key = f"{phase}_Aufgabe_1"  # Induction always has aufgabe=1
            file_groups[group_key].append(csv_file)
        elif parsed['phase_type'] == 'Training':
            # Create group key: Training_<N>_Aufgabe_<M>
            if parsed['training_number'] and parsed['aufgabe']:
                group_key = f"Training_{parsed['training_number']}_Aufgabe_{parsed['aufgabe']}"
                file_groups[group_key].append(csv_file)
    
    # Instead of saving, collect results
    results = []
    
    # Process each group (move the processing logic inside the loop)
    for group_key, files in file_groups.items():
        logger.info(f"Processing group: {group_key} ({len(files)} files)")
        
        # Group files by method (should have 5 files: one per method)
        files_by_method = defaultdict(list)
        for csv_file in files:
            parsed = parse_filename(csv_file.name)
            if parsed['method']:
                files_by_method[parsed['method']].append(csv_file)
            else:
                logger.warning(f"Could not determine method for {csv_file.name}")
        
        if len(files_by_method) == 0:
            logger.warning(f"No files with valid methods found for group {group_key}")
            continue
        
        # Determine phase and aufgabe from group key
        if group_key.startswith('induction'):
            # Extract phase from group key (e.g., "induction1_Aufgabe_1")
            phase = group_key.split('_')[0]  # "induction1" or "induction2"
            aufgabe = '1'
        elif group_key.startswith('Training'):
            # Extract training number and aufgabe (e.g., "Training_1_Aufgabe_5")
            parts = group_key.split('_')
            training_num = parts[1] if len(parts) > 1 else None
            aufgabe = parts[3] if len(parts) > 3 else '1'
            phase = determine_training_phase(training_num, metadata) if training_num else 'training_unknown'
        else:
            phase = 'unknown'
            aufgabe = '1'
        
        # Step 1: Read all CSV files first (before renaming) to identify common columns
        dataframes_by_method = {}
        
        # Read and process csv files one by one amd sttore them in one dictionary
        for method, method_files in files_by_method.items():
            csv_file = method_files[0]
            
            try:
                # Read CSV file
                logger.info(f"  Reading file: {csv_file.name}")
                # Try to detect delimiter (semicolon or comma)
                try:
                    # First, try reading with semicolon delimiter
                    df = pd.read_csv(csv_file, sep=';')
                    # If that fails or gives weird results, try comma
                    if df.shape[1] == 1:
                        df = pd.read_csv(csv_file, sep=',')
                except Exception:
                    df = pd.read_csv(csv_file, sep=',')
                
                # Log feature information before processing
                feature_cols = [c for c in df.columns]
                logger.info(f"Number of Features BEFORE processing: {len(feature_cols)} columns")
                
                # Remove columns that contain 'name' or 'frameTime' in their names
                # Handle various formats: 'name;frameTime;...', 'name', 'frameTime', etc.
                columns_to_remove = []
                for col in df.columns:
                    col_str = str(col).strip()
                    col_lower = col_str.lower()
                    
                    # Check if column name is exactly 'name' or 'frameTime' (with any whitespace)
                    if col_lower in ['name', 'frametime']:
                        columns_to_remove.append(col)
                    # Check if column name starts with 'name;' or contains ';name;' or ';frameTime;'
                    elif ';' in col_str:
                        parts = col_str.split(';')
                        # Check if first or second part is 'name' or 'frameTime'
                        if len(parts) > 0 and parts[0].lower().strip() in ['name', 'frametime']:
                            columns_to_remove.append(col)
                        elif len(parts) > 1 and parts[1].lower().strip() in ['name', 'frametime']:
                            columns_to_remove.append(col)
                
                if columns_to_remove:
                    df = df.drop(columns=columns_to_remove)
                    logger.info(f"    Removed columns containing 'name' or 'frameTime': {len(columns_to_remove)} columns")
                
                # Store dataframe with method as key (before renaming)
                dataframes_by_method[method] = df
                logger.info(f"    Processed {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} columns (after removing name/frameTime)")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue
        
        if len(dataframes_by_method) == 0:
            logger.warning(f"No dataframes to process for group {group_key}")
            continue
        
        # Step 2: Align row counts first (required for fair NaN comparison and concatenation)
        row_counts = [df.shape[0] for df in dataframes_by_method.values()]
        if len(set(row_counts)) > 1:
            min_rows = min(row_counts)
            logger.info(f"  Aligning dataframes to {min_rows} rows (minimum row count)")
            # Truncate all dataframes to minimum row count
            for method in dataframes_by_method.keys():
                df = dataframes_by_method[method]
                if df.shape[0] > min_rows:
                    dataframes_by_method[method] = df.iloc[:min_rows].copy()
                    logger.debug(f"    Truncated {method} from {df.shape[0]} to {min_rows} rows")
        else:
            min_rows = row_counts[0] if len(row_counts) > 0 else 0
            logger.info(f"  All dataframes have same row count: {min_rows}")
        
        # Step 3: Normalize column names (lowercase, strip, remove spaces) and log them
        logger.info("  Normalizing column names (lowercase, strip, remove spaces)...")
        for method, df in dataframes_by_method.items():
            logger.info(f"    {method}: {len(df.columns)} columns before normalization")
            df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "", regex=True)
            logger.debug(f"    {method} column names (first 10): {list(df.columns)[:10]}")
        
        # Step 4: Find columns that appear in 2 or more dataframes (not just all)
        # Count how many dataframes contain each column
        column_counts = defaultdict(list)  # column_name -> list of methods that have it
        
        for method, df in dataframes_by_method.items():
            for col in df.columns:
                column_counts[col].append(method)
        
        # Find columns that appear in 2 or more files
        duplicate_columns = {col: methods for col, methods in column_counts.items() if len(methods) >= 2}
        
        logger.info(f"  Found {len(duplicate_columns)} columns that appear in 2 or more files")
        if len(duplicate_columns) > 0:
            logger.info(f"    Duplicate columns (first 20): {list(duplicate_columns.keys())[:20]}")
            # Log which methods have each duplicate column
            for col, methods in list(duplicate_columns.items())[:10]:
                logger.debug(f"      '{col}' appears in: {methods}")
        
        # Step 5: For each duplicate column, keep the version with fewer NaNs and drop from others
        if len(duplicate_columns) > 0:
            columns_removed_count = 0
            for col, methods_with_col in duplicate_columns.items():
                # Calculate NaN counts for this column across all methods that have it
                nan_counts = {}
                for method in methods_with_col:
                    df = dataframes_by_method[method]
                    if col in df.columns:
                        nan_counts[method] = df[col].isna().sum()
                
                if len(nan_counts) == 0:
                    continue
                
                # Find method with minimum NaN count
                best_method = min(nan_counts.items(), key=lambda x: x[1])[0]
                logger.debug(f"    Column '{col}': keeping from {best_method} (NaNs: {nan_counts[best_method]})")
                
                # Drop this column from all other methods
                for method in methods_with_col:
                    if method != best_method and col in dataframes_by_method[method].columns:
                        dataframes_by_method[method] = dataframes_by_method[method].drop(columns=[col])
                        columns_removed_count += 1
                        logger.debug(f"      Dropped '{col}' from {method} (NaNs: {nan_counts[method]})")
            
            logger.info(f"  Removed {columns_removed_count} duplicate column instances (kept best version from each duplicate)")
        
        # Step 6: Rename columns to include method suffix and prepare for concatenation
        all_dataframes = []
        for method, df in dataframes_by_method.items():
            # Rename columns to include method suffix
            # Format: original_column_name_method (e.g., F0_compare16, F0_egemaps)
            df.columns = [f"{col}_{method}" for col in df.columns]
            all_dataframes.append(df)
            logger.info(f"    Renamed columns for {method}: {df.shape[1]} columns")
        
        if len(all_dataframes) == 0:
            logger.warning(f"No dataframes to concatenate for group {group_key}")
            continue
        
        # Log total features before concatenation
        total_features_before = sum(len(df.columns) for df in all_dataframes)
        logger.info(f"  Total features across all files BEFORE concatenation: {total_features_before}")
        
        # Concatenate horizontally (axis=1) instead of vertically (axis=0)
        concatenated_df = pd.concat(all_dataframes, axis=1)
        logger.info(f"  Horizontally concatenated dataframe: {concatenated_df.shape[0]} rows, {concatenated_df.shape[1]} columns")
        
        # Remove any remaining 'name' or 'frameTime' columns after concatenation
        cols_to_remove_final = []
        for col in concatenated_df.columns:
            col_str = str(col).strip().lower()
            # Check for columns that contain name/frameTime patterns
            if col_str.startswith('name') or col_str.startswith('frametime'):
                # Check if it's a compound column like "name;frameTime;..." or just "name" or "frameTime"
                if ';' in col_str:
                    parts = col_str.split(';')
                    if len(parts) > 0 and parts[0] in ['name', 'frametime']:
                        cols_to_remove_final.append(col)
                else:
                    # Simple column name
                    if col_str in ['name', 'frametime']:
                        cols_to_remove_final.append(col)
            elif ';name;' in col_str or ';frametime;' in col_str:
                cols_to_remove_final.append(col)
        
        if cols_to_remove_final:
            concatenated_df = concatenated_df.drop(columns=cols_to_remove_final)
            logger.info(f"  Removed {len(cols_to_remove_final)} columns containing 'name'/'frameTime' after concatenation")
            logger.debug(f"  Removed columns: {cols_to_remove_final[:10]}")
        
        # Check for any remaining duplicate column names (should be none since we handled duplicates before concatenation)
        duplicate_names = [col for col in concatenated_df.columns if concatenated_df.columns.tolist().count(col) > 1]
        if duplicate_names:
            original_col_count = len(concatenated_df.columns)
            logger.error(f"  ERROR: Found {len(set(duplicate_names))} duplicate column names after concatenation: {list(set(duplicate_names))[:10]}")
            logger.error(f"  This should not happen - duplicates should have been removed before concatenation!")
            # Keep only first occurrence of each duplicate column
            seen = set()
            cols_to_keep = []
            for col in concatenated_df.columns:
                if col not in seen:
                    seen.add(col)
                    cols_to_keep.append(col)
            concatenated_df = concatenated_df[cols_to_keep]
            removed_count = original_col_count - len(cols_to_keep)
            logger.warning(f"  Removed {removed_count} duplicate columns (kept first occurrence)")
        else:
            logger.info(f"  ✓ No duplicate column names found (duplicates were handled before concatenation)")
        
        # Add metadata columns (only once, not per method)
        concatenated_df['ID'] = participant_id
        concatenated_df['diagnose'] = metadata.get('Diagnose', 'Unknown')
        concatenated_df['condition'] = metadata.get('condition', 'Unknown')
        concatenated_df['phase'] = phase
        concatenated_df['aufgabe'] = int(aufgabe)
        
        # Reorder columns: metadata first, then audio features
        metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
        feature_cols = [c for c in concatenated_df.columns if c not in metadata_cols]
        concatenated_df = concatenated_df[metadata_cols + feature_cols]
        
        # Log feature information after processing
        logger.info(f"  Features AFTER processing: {len(feature_cols)} audio feature columns")
        logger.info(f"  Total columns (including metadata): {len(concatenated_df.columns)}")
        #logger.debug(f"  Feature names (first 20): {feature_cols[:20]}{'...' if len(feature_cols) > 20 else ''}")
        
        # Show breakdown by method
        method_counts = {}
        for col in feature_cols:
            for method in ['compare16', 'egemaps', 'gemaps', 'is09', 'is13', 'is02']:
                if col.endswith(f'_{method}'):
                    method_counts[method] = method_counts.get(method, 0) + 1
                    break
        logger.info(f"  Features by method: {method_counts}")
        
        # Get values for output filename
        aufgabe_num = int(concatenated_df['aufgabe'].iloc[0]) if len(concatenated_df) > 0 else 1
        phase_name = str(concatenated_df['phase'].iloc[0]) if len(concatenated_df) > 0 else 'unknown'
        diagnose_lower = str(metadata.get('Diagnose', 'Unknown')).lower()
        condition = str(metadata.get('condition', 'Unknown'))
        
        # Create output filename: {ID}_{diagnose}_{condition}_{phase}_{aufgabe}.csv
        output_filename = f"{participant_id}_{diagnose_lower}_{condition}_{phase_name}_{aufgabe_num}.csv"
        output_path = output_dir / output_filename
        
        # After creating concatenated_df (around line 473), instead of saving:
        result_dict = {
            'dataframe': concatenated_df,
            'phase': phase,
            'aufgabe': int(aufgabe),
            'participant_id': participant_id,
            'group_key': group_key
        }
        results.append(result_dict)
        
        # Only save if requested
        if save_to_disk:
            concatenated_df.to_csv(output_path, index=False)
            logger.info(f"  ✓ Saved: {output_path}")
    
    return results  # Return list of processed dataframes


def process_all_participants(base_path: str, metadata_path: str, output_base_dir: str = "data"):
    """
    Process all participants' data.
    
    Args:
        base_path: Base directory containing participant folders
        metadata_path: Path to merged_RCT_info.csv
        output_base_dir: Base directory for output files
    """
    logger.info(f"\n{'='*80}")
    logger.info("STARTING DATA PROCESSING")
    logger.info(f"{'='*80}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Metadata path: {metadata_path}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}\n")
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {len(metadata_df)} participants")
    
    # Track overall statistics
    stats = {
        'total_participants': len(metadata_df),
        'processed_participants': 0,
        'failed_participants': 0,
        'total_files_processed': 0,
        'total_groups_processed': 0
    }
    
    # Process each participant
    for _, row in metadata_df.iterrows():
        participant_id = int(row['ID'])
        metadata_dict = row.to_dict()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing participant {participant_id}")
        logger.info(f"{'='*80}")
        
        try:
            # Call process_participant_files with save_to_disk=True
            processed_dataframes = process_participant_files(base_path, participant_id, metadata_dict, output_base_dir, save_to_disk=True)
            stats['processed_participants'] += 1
            stats['total_files_processed'] += sum(df['dataframe'].shape[0] for df in processed_dataframes)
            stats['total_groups_processed'] += len(processed_dataframes)
        except Exception as e:
            logger.error(f"Error processing participant {participant_id}: {e}", exc_info=True)
            stats['failed_participants'] += 1
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Summary Statistics:")
    logger.info(f"  Total participants: {stats['total_participants']}")
    logger.info(f"  Successfully processed: {stats['processed_participants']}")
    logger.info(f"  Failed: {stats['failed_participants']}")
    logger.info(f"{'='*80}")
    logger.info(f"Detailed log saved to: {log_file}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    # Configuration
    BASE_PATH = "/home/vault/empkins/tpD/D02/processed_data/processed_audio_opensmile"
    METADATA_PATH = "/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/merged_RCT_info.csv"  # Relative to script location
    OUTPUT_BASE_DIR = "/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/OpenSmile_data"
    
    # TEST MODE: Set to a participant ID to test on just one participant, or None to process all
    TEST_PARTICIPANT_ID = None  # Change to a number like 4, 13, etc. to test on one participant
    
    # If metadata path is not found, try absolute path
    script_dir = Path(__file__).parent
    metadata_abs_path = script_dir / METADATA_PATH
    if not metadata_abs_path.exists():
        metadata_abs_path = input("Enter full path to merged_RCT_info.csv: ").strip()
    
    # Test mode: process single participant
    if TEST_PARTICIPANT_ID is not None:
        for participant_id in TEST_PARTICIPANT_ID:
            logger.info(f"\n{'='*80}")
            logger.info("TEST MODE: Processing single participant")
            logger.info(f"{'='*80}")
            
            # Load metadata
            metadata_df = pd.read_csv(metadata_abs_path)
            
            # Find the participant
            participant_row = metadata_df[metadata_df['ID'] == participant_id]
            if len(participant_row) == 0:
                logger.error(f"Participant {participant_id} not found in metadata!")
            else:
                metadata_dict = participant_row.iloc[0].to_dict()
                logger.info(f"Processing participant {participant_id}")
                logger.info(f"{'='*80}\n")
                
                try:
                    # Call process_participant_files with save_to_disk=True
                    processed_dataframes = process_participant_files(BASE_PATH, participant_id, metadata_dict, OUTPUT_BASE_DIR, save_to_disk=True)
                    logger.info(f"\n{'='*80}")
                    logger.info(f"✓ Successfully processed participant {participant_id}")
                    logger.info(f"{'='*80}")
                except Exception as e:
                    logger.error(f"Error processing participant {participant_id}: {e}", exc_info=True)
    else:
        # Normal mode: process all participants
        process_all_participants(BASE_PATH, str(metadata_abs_path), OUTPUT_BASE_DIR)
