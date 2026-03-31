#!/usr/bin/env python3
"""
Audio Data Exploration Script

This script explores CSV files for a given ID, condition, phase, and aufgabe.
It provides:
1. Dimensions (rows, columns) of the CSV file
2. Feature analysis grouped by method (compare16, egemaps, gemaps, is09, is13)
3. Common features between methods (by name and by values)
4. Statistics about NaN/Inf values per feature
5. Basic descriptive statistics

Usage:
    python Audio_Data_Exploration.py --id 4 --condition CRADK --phase training_pos --aufgabe 1
    python Audio_Data_Exploration.py --id 4 --condition CRADK --phase training_pos --aufgabe 1 --id2 38
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple, Optional
import sys
import re
from collections import defaultdict
from datetime import datetime


class Tee:
    """
    Class to write output to both stdout and a file simultaneously.
    """
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse CSV filename to extract phase, aufgabe, method, and timestamps.
    
    Expected formats:
    - Induction: YYYY-MM-DD_HH-MM_Belastungsphase_YYYY-MM-DD_HH-MM_<method>.csv
    - Training: YYYY-MM-DD_HH-MM_Training_<N>_Aufgabe_<M>_YYYY-MM-DD_HH-MM_<method>.csv
    """
    result = {
        'phase_type': None,
        'training_number': None,
        'aufgabe': None,
        'method': None,
        'start_timestamp': None,
        'end_timestamp': None
    }
    
    name = filename.replace('.csv', '')
    
    # Extract timestamps
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})'
    timestamps = re.findall(timestamp_pattern, name)
    if len(timestamps) >= 2:
        result['start_timestamp'] = timestamps[0]
        result['end_timestamp'] = timestamps[1]
    
    # Check for Belastungsphase
    if 'Belastungsphase' in name:
        result['phase_type'] = 'Belastungsphase'
        result['aufgabe'] = None
    
    # Check for Training
    elif 'Training' in name:
        result['phase_type'] = 'Training'
        training_match = re.search(r'Training_(\d+)', name)
        if training_match:
            result['training_number'] = training_match.group(1)
        
        aufgabe_match = re.search(r'Aufgabe_(\d+)', name)
        if aufgabe_match:
            result['aufgabe'] = aufgabe_match.group(1)
    
    # Extract method
    parts = name.split('_')
    if len(parts) > 0:
        possible_methods = ['compare16', 'egemaps', 'gemaps', 'is09', 'is13', 'is02']
        last_part = parts[-1]
        if last_part in possible_methods:
            result['method'] = last_part
    
    return result


def determine_induction_phase(belastungsphase_files: List[str]) -> Dict[str, str]:
    """Determine which Belastungsphase files belong to induction1 vs induction2."""
    file_phases = {}
    timestamp_groups = defaultdict(list)
    
    for filename in belastungsphase_files:
        parsed = parse_filename(filename)
        if parsed['start_timestamp']:
            try:
                timestamp_dt = datetime.strptime(parsed['start_timestamp'], '%Y-%m-%d_%H-%M')
                timestamp_groups[timestamp_dt].append(filename)
            except ValueError:
                pass
    
    sorted_timestamps = sorted(timestamp_groups.keys())
    
    for i, timestamp in enumerate(sorted_timestamps):
        phase = 'induction1' if i == 0 else 'induction2'
        for filename in timestamp_groups[timestamp]:
            file_phases[filename] = phase
    
    return file_phases


def determine_training_phase(training_number: str, metadata: Dict) -> str:
    """Determine if Training_1 or Training_2 is training_pos or training_neg."""
    if training_number == '1':
        training_type = metadata.get('training1_type', '').lower()
    elif training_number == '2':
        training_type = metadata.get('training2_type', '').lower()
    else:
        return 'training_unknown'
    
    if training_type == 'positive':
        return 'training_pos'
    elif training_type == 'negative':
        return 'training_neg'
    else:
        return f'training_{training_type}'


def find_raw_csv_files(data_dir: Path, participant_id: int, condition: str, phase: str, aufgabe: int) -> Dict[str, Path]:
    """
    Find the 5 raw CSV files (one per method) matching the given parameters.
    
    Args:
        data_dir: Base directory containing merged_RCT_info.csv (for metadata)
        participant_id: Participant ID (integer)
        condition: Condition (CR, CRADK, ADK, SHAM) - used for validation only
        phase: Phase (training_pos, training_neg, induction1, induction2)
        aufgabe: Aufgabe number (integer)
    
    Returns:
        Dictionary mapping method names to file paths
    
    Raises:
        FileNotFoundError: If files not found
    """
    # Zero-pad participant ID
    padded_id = str(participant_id).zfill(3)
    
    # Load metadata
    metadata_path = data_dir / "merged_RCT_info.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_df = pd.read_csv(metadata_path)
    participant_row = metadata_df[metadata_df['ID'] == participant_id]
    
    if len(participant_row) == 0:
        raise ValueError(f"Participant ID {participant_id} not found in metadata")
    
    # Convert row to dict
    metadata = participant_row.iloc[0].to_dict()
    
    # Validate condition
    if metadata['condition'] != condition:
        raise ValueError(f"Participant {participant_id} has condition {metadata['condition']}, not {condition}")
    
    # Construct path to raw data folder
    raw_data_dir = Path("/home/vault/empkins/tpD/D02/processed_data/processed_audio_opensmile")
    participant_dir = raw_data_dir / padded_id / "timeseries_opensmile_features"
    
    if not participant_dir.exists():
        raise FileNotFoundError(f"Participant directory not found: {participant_dir}")
    
    # Get all CSV files
    csv_files = list(participant_dir.glob("*.csv"))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {participant_dir}")
    
    # First, determine induction phases for Belastungsphase files
    belastungsphase_files = []
    for csv_file in csv_files:
        parsed = parse_filename(csv_file.name)
        if parsed['phase_type'] == 'Belastungsphase':
            belastungsphase_files.append(csv_file.name)
    
    induction_phases = {}
    if belastungsphase_files:
        induction_phases = determine_induction_phase(belastungsphase_files)
    
    # Find files matching the requested phase and aufgabe
    matching_files = {}
    methods = ['compare16', 'egemaps', 'gemaps', 'is09', 'is13']
    
    for csv_file in csv_files:
        parsed = parse_filename(csv_file.name)
        
        if not parsed['method'] or parsed['method'] not in methods:
            continue
        
        file_phase = None
        file_aufgabe = None
        
        if parsed['phase_type'] == 'Belastungsphase':
            file_phase = induction_phases.get(csv_file.name, None)
            file_aufgabe = 1  # Induction always has aufgabe=1
        elif parsed['phase_type'] == 'Training':
            if parsed['training_number']:
                file_phase = determine_training_phase(parsed['training_number'], metadata)
            if parsed['aufgabe']:
                file_aufgabe = int(parsed['aufgabe'])
        
        # Check if this file matches our criteria
        if file_phase == phase and file_aufgabe == aufgabe:
            method = parsed['method']
            if method not in matching_files:
                matching_files[method] = csv_file
    
    # Check if we found all 5 methods
    missing_methods = [m for m in methods if m not in matching_files]
    if missing_methods:
        raise FileNotFoundError(
            f"Could not find files for methods: {missing_methods}\n"
            f"Found files for: {list(matching_files.keys())}\n"
            f"Looking for: phase={phase}, aufgabe={aufgabe}"
        )
    
    return matching_files


def get_method_from_feature(feature_name: str) -> str:
    """
    Extract method name from feature name.
    
    Args:
        feature_name: Feature name (e.g., 'F0final_sma_egemaps', 'loudness_sma3_compare16')
    
    Returns:
        Method name (egemaps, compare16, gemaps, is09, is13, is02) or 'unknown'
    """
    methods = ['compare16', 'egemaps', 'gemaps', 'is09', 'is13', 'is02']
    for method in methods:
        if feature_name.endswith(f'_{method}'):
            return method
    return 'unknown'


def group_features_by_method(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group feature columns by their method.
    
    Args:
        df: DataFrame with feature columns
    
    Returns:
        Dictionary mapping method names to lists of feature names
    """
    metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    features_by_method = {}
    for col in feature_cols:
        method = get_method_from_feature(col)
        if method not in features_by_method:
            features_by_method[method] = []
        features_by_method[method].append(col)
    
    return features_by_method


def extract_base_feature_name(feature_name: str) -> str:
    """
    Extract base feature name without method suffix.
    
    Example:
        'F0final_sma_egemaps' -> 'F0final_sma'
        'loudness_sma3_compare16' -> 'loudness_sma3'
    """
    methods = ['compare16', 'egemaps', 'gemaps', 'is09', 'is13', 'is02']
    for method in methods:
        if feature_name.endswith(f'_{method}'):
            return feature_name[:-len(f'_{method}')]
    return feature_name


def find_common_features(features_by_method: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Find features that appear in multiple methods (by base name).
    
    Args:
        features_by_method: Dictionary mapping methods to feature lists
    
    Returns:
        Dictionary mapping base feature names to list of methods that have it
    """
    # Build reverse mapping: base_name -> methods
    base_to_methods = {}
    for method, features in features_by_method.items():
        for feature in features:
            base_name = extract_base_feature_name(feature)
            if base_name not in base_to_methods:
                base_to_methods[base_name] = []
            if method not in base_to_methods[base_name]:
                base_to_methods[base_name].append(method)
    
    # Keep only features that appear in 2+ methods
    common_features = {base: methods for base, methods in base_to_methods.items() 
                      if len(methods) >= 2}
    
    return common_features


def compare_feature_values(df: pd.DataFrame, base_feature: str, methods: List[str]) -> Dict:
    """
    Compare values of a feature across different methods.
    
    Args:
        df: DataFrame
        base_feature: Base feature name (without method suffix)
        methods: List of methods that have this feature
    
    Returns:
        Dictionary with comparison statistics
    """
    results = {
        'base_feature': base_feature,
        'methods': methods,
        'correlations': {},
        'mean_values': {},
        'std_values': {},
        'are_similar': False
    }
    
    # Get feature columns for each method
    feature_cols = {}
    for method in methods:
        full_name = f"{base_feature}_{method}"
        if full_name in df.columns:
            feature_cols[method] = df[full_name]
    
    if len(feature_cols) < 2:
        return results
    
    # Calculate correlations between methods
    for i, method1 in enumerate(methods):
        if method1 not in feature_cols:
            continue
        for method2 in methods[i+1:]:
            if method2 not in feature_cols:
                continue
            col1 = feature_cols[method1]
            col2 = feature_cols[method2]
            
            # Remove NaN/Inf for correlation
            mask = ~(pd.isna(col1) | pd.isna(col2) | np.isinf(col1) | np.isinf(col2))
            if mask.sum() > 10:  # Need at least 10 valid values
                corr = np.corrcoef(col1[mask], col2[mask])[0, 1]
                results['correlations'][f"{method1}_vs_{method2}"] = corr
    
    # Calculate mean and std for each method
    for method, col in feature_cols.items():
        valid_data = col[~(pd.isna(col) | np.isinf(col))]
        if len(valid_data) > 0:
            results['mean_values'][method] = float(valid_data.mean())
            results['std_values'][method] = float(valid_data.std())
    
    # Determine if values are similar (high correlation > 0.8)
    if results['correlations']:
        max_corr = max(results['correlations'].values())
        results['are_similar'] = max_corr > 0.8
    
    return results


def find_common_features_across_files(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    Find features that appear in multiple files (by exact column name).
    
    Args:
        dataframes: Dictionary mapping method names to DataFrames
    
    Returns:
        Dictionary mapping feature names to list of methods that have it
    """
    # Build mapping: feature_name -> methods
    feature_to_methods = defaultdict(list)
    
    for method, df in dataframes.items():
        for col in df.columns:
            feature_to_methods[col].append(method)
    
    # Keep only features that appear in 2+ methods
    common_features = {feat: methods for feat, methods in feature_to_methods.items() 
                      if len(methods) >= 2}
    
    return common_features


def compare_feature_values_across_files(dataframes: Dict[str, pd.DataFrame], feature_name: str, methods: List[str]) -> Dict:
    """
    Compare values of a feature across different files.
    
    Args:
        dataframes: Dictionary mapping method names to DataFrames
        feature_name: Feature name (exact column name)
        methods: List of methods that have this feature
    
    Returns:
        Dictionary with comparison statistics
    """
    results = {
        'feature': feature_name,
        'methods': methods,
        'correlations': {},
        'mean_values': {},
        'std_values': {},
        'are_similar': False
    }
    
    # Get feature columns for each method
    feature_cols = {}
    for method in methods:
        if method in dataframes and feature_name in dataframes[method].columns:
            col_data = dataframes[method][feature_name]
            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                feature_cols[method] = col_data
    
    if len(feature_cols) < 2:
        return results
    
    # Align dataframes by index (frame number) if possible
    # For now, assume they have the same number of rows and align by position
    
    # Calculate correlations between methods
    for i, method1 in enumerate(methods):
        if method1 not in feature_cols:
            continue
        for method2 in methods[i+1:]:
            if method2 not in feature_cols:
                continue
            col1 = feature_cols[method1]
            col2 = feature_cols[method2]
            
            # Align by index if possible, otherwise by position
            if len(col1) == len(col2):
                # Remove NaN/Inf for correlation
                try:
                    mask = ~(pd.isna(col1) | pd.isna(col2))
                    # Only check for Inf if columns are numeric
                    if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                        mask = mask & ~(np.isinf(col1) | np.isinf(col2))
                    
                    if mask.sum() > 10:  # Need at least 10 valid values
                        corr = np.corrcoef(col1[mask], col2[mask])[0, 1]
                        if not np.isnan(corr):
                            results['correlations'][f"{method1}_vs_{method2}"] = corr
                except (TypeError, ValueError):
                    # Skip if correlation calculation fails
                    pass
    
    # Calculate mean and std for each method
    for method, col in feature_cols.items():
        try:
            mask = ~pd.isna(col)
            if pd.api.types.is_numeric_dtype(col):
                mask = mask & ~np.isinf(col)
            valid_data = col[mask]
            if len(valid_data) > 0:
                results['mean_values'][method] = float(valid_data.mean())
                results['std_values'][method] = float(valid_data.std())
        except (TypeError, ValueError):
            # Skip if calculation fails
            pass
    
    # Determine if values are similar (high correlation > 0.8)
    if results['correlations']:
        max_corr = max(results['correlations'].values())
        results['are_similar'] = max_corr > 0.8
    
    return results


def analyze_nan_inf_file(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Analyze NaN and Inf values per feature in a single file.
    
    Args:
        df: DataFrame
        method: Method name (for labeling)
    
    Returns:
        DataFrame with NaN/Inf statistics per column
    """
    stats = []
    for col in df.columns:
        col_data = df[col]
        n_total = len(col_data)
        n_nan = pd.isna(col_data).sum()
        
        # Only check for Inf values if column is numeric
        n_inf = 0
        if pd.api.types.is_numeric_dtype(col_data):
            try:
                n_inf = np.isinf(col_data).sum()
            except (TypeError, ValueError):
                # If conversion fails, treat as 0 infinite values
                n_inf = 0
        
        n_valid = n_total - n_nan - n_inf
        
        stats.append({
            'feature': col,
            'method': method,
            'total_rows': n_total,
            'nan_count': n_nan,
            'inf_count': n_inf,
            'valid_count': n_valid,
            'nan_percent': (n_nan / n_total * 100) if n_total > 0 else 0,
            'inf_percent': (n_inf / n_total * 100) if n_total > 0 else 0,
            'valid_percent': (n_valid / n_total * 100) if n_total > 0 else 0
        })
    
    return pd.DataFrame(stats)


def analyze_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze NaN and Inf values per feature.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame with NaN/Inf statistics per column
    """
    metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    stats = []
    for col in feature_cols:
        col_data = df[col]
        n_total = len(col_data)
        n_nan = pd.isna(col_data).sum()
        
        # Only check for Inf values if column is numeric
        n_inf = 0
        if pd.api.types.is_numeric_dtype(col_data):
            try:
                n_inf = np.isinf(col_data).sum()
            except (TypeError, ValueError):
                # If conversion fails, treat as 0 infinite values
                n_inf = 0
        
        n_valid = n_total - n_nan - n_inf
        
        stats.append({
            'feature': col,
            'method': get_method_from_feature(col),
            'total_rows': n_total,
            'nan_count': n_nan,
            'inf_count': n_inf,
            'valid_count': n_valid,
            'nan_percent': (n_nan / n_total * 100) if n_total > 0 else 0,
            'inf_percent': (n_inf / n_total * 100) if n_total > 0 else 0,
            'valid_percent': (n_valid / n_total * 100) if n_total > 0 else 0
        })
    
    return pd.DataFrame(stats)


def explore_data(participant_id: int, condition: str, phase: str, aufgabe: int, 
                 data_dir: Path, verbose: bool = True) -> Dict:
    """
    Main exploration function.
    
    Args:
        participant_id: Participant ID
        condition: Condition (CR, CRADK, ADK, SHAM)
        phase: Phase (training_pos, training_neg, induction1, induction2)
        aufgabe: Aufgabe number
        data_dir: Base data directory
        verbose: Whether to print results
    
    Returns:
        Dictionary with exploration results
    """
    results = {
        'participant_id': participant_id,
        'condition': condition,
        'phase': phase,
        'aufgabe': aufgabe,
        'file_paths': None,
        'dimensions': None,
        'features_by_method': None,
        'common_features': None,
        'nan_inf_stats': None,
        'feature_comparisons': None
    }
    
    # Find and load raw CSV files
    try:
        file_paths = find_raw_csv_files(data_dir, participant_id, condition, phase, aufgabe)
        results['file_paths'] = {method: str(path) for method, path in file_paths.items()}
        
        if verbose:
            print("=" * 80)
            print(f"EXPLORING RAW DATA FOR PARTICIPANT {participant_id}")
            print("=" * 80)
            print(f"Condition: {condition}")
            print(f"Phase: {phase}")
            print(f"Aufgabe: {aufgabe}")
            print(f"\nFound {len(file_paths)} CSV files:")
            for method, path in sorted(file_paths.items()):
                print(f"  {method:12s}: {Path(path).name}")
            print()
        
        # Load all dataframes
        dataframes = {}
        for method, file_path in file_paths.items():
            try:
                # Try semicolon delimiter first (common in OpenSMILE output)
                df = pd.read_csv(file_path, sep=';')
                if df.shape[1] == 1:
                    df = pd.read_csv(file_path, sep=',')
            except Exception:
                df = pd.read_csv(file_path, sep=',')
            dataframes[method] = df
        
        # 1. Dimensions for each file
        dimensions_by_method = {}
        for method, df in dataframes.items():
            n_rows, n_cols = df.shape
            dimensions_by_method[method] = {
                'rows': n_rows,
                'columns': n_cols,
                'features': n_cols
            }
        
        results['dimensions'] = dimensions_by_method
        
        if verbose:
            print("1. DIMENSIONS (per file)")
            print("-" * 80)
            for method in sorted(dataframes.keys()):
                dims = dimensions_by_method[method]
                print(f"  {method:12s}: {dims['rows']:6,} rows × {dims['columns']:4d} columns (features)")
            print()
        
        # 2. Features per method
        features_by_method = {method: list(df.columns) for method, df in dataframes.items()}
        results['features_by_method'] = {k: len(v) for k, v in features_by_method.items()}
        
        # Calculate total features across all files
        total_features = sum(len(features) for features in features_by_method.values())
        
        if verbose:
            print("2. FEATURES BY METHOD")
            print("-" * 80)
            print(f"  Total features across all 5 files: {total_features}")
            print()
            for method, features in sorted(features_by_method.items()):
                print(f"  {method:12s}: {len(features):4d} features")
            print()
        
        # 3. Common features (by exact name)
        common_features = find_common_features_across_files(dataframes)
        
        # Count common features by method (how many common features each method participates in)
        common_features_by_method = defaultdict(int)
        for feat, methods in common_features.items():
            for method in methods:
                common_features_by_method[method] += 1
        
        # Count common features by number of files they appear in
        common_features_by_count = defaultdict(int)
        for feat, methods in common_features.items():
            n_files = len(methods)
            common_features_by_count[n_files] += 1
        
        results['common_features'] = {
            'count': len(common_features),
            'details': {feat: methods for feat, methods in common_features.items()},
            'by_method': dict(common_features_by_method),
            'by_file_count': dict(common_features_by_count)
        }
        
        if verbose:
            print("3. COMMON FEATURES (appear in 2+ files by exact name)")
            print("-" * 80)
            print(f"  Total common features: {len(common_features)}")
            print()
            
            # Show statistics by number of files
            print("  Common features by number of files:")
            for n_files in sorted(common_features_by_count.keys(), reverse=True):
                count = common_features_by_count[n_files]
                print(f"    Appear in {n_files} files: {count} features")
            print()
            
            # Show statistics by method
            print("  Common features by method (how many each method participates in):")
            for method in sorted(common_features_by_method.keys()):
                count = common_features_by_method[method]
                print(f"    {method:12s}: {count:4d} common features")
            print()
            
            # Show ALL common features
            if len(common_features) > 0:
                print(f"  All {len(common_features)} common features:")
                for feat, methods in sorted(common_features.items()):
                    methods_str = ', '.join(sorted(methods))
                    print(f"    {feat:50s} → files: {methods_str}")
            print()
        
        # 4. NaN/Inf statistics per file
        all_nan_inf_stats = []
        for method, df in dataframes.items():
            stats = analyze_nan_inf_file(df, method)
            all_nan_inf_stats.append(stats)
        
        nan_inf_stats = pd.concat(all_nan_inf_stats, ignore_index=True)
        results['nan_inf_stats'] = nan_inf_stats
        
        if verbose:
            print("4. NaN/Inf STATISTICS")
            print("-" * 80)
            
            # Summary by method
            summary_by_method = nan_inf_stats.groupby('method').agg({
                'nan_count': 'sum',
                'inf_count': 'sum',
                'valid_count': 'sum',
                'total_rows': 'first'
            }).reset_index()
            
            print("  Summary by method:")
            for _, row in summary_by_method.iterrows():
                method = row['method']
                n_features = len(features_by_method.get(method, []))
                total_cells = row['total_rows'] * n_features
                if total_cells > 0:
                    nan_pct = (row['nan_count'] / total_cells * 100)
                    inf_pct = (row['inf_count'] / total_cells * 100)
                    valid_pct = (row['valid_count'] / total_cells * 100)
                    print(f"    {method:12s}: NaN={nan_pct:6.2f}%, Inf={inf_pct:6.2f}%, Valid={valid_pct:6.2f}%")
            
            # Features with most NaN/Inf (across all files)
            nan_inf_stats_sorted = nan_inf_stats.sort_values('nan_percent', ascending=False)
            print(f"\n  Top 10 features with most NaN values (across all files):")
            for _, row in nan_inf_stats_sorted.head(10).iterrows():
                print(f"    [{row['method']:12s}] {row['feature']:40s}: {row['nan_percent']:6.2f}% NaN, {row['inf_percent']:6.2f}% Inf")
            print()
        
        # 5. Compare feature values for common features
        if verbose:
            print("5. FEATURE VALUE COMPARISONS (for common features)")
            print("-" * 80)
        
        feature_comparisons = []
        for feat_name, methods in list(common_features.items())[:]:  # Check first 10
            comparison = compare_feature_values_across_files(dataframes, feat_name, methods)
            feature_comparisons.append(comparison)
            
            if verbose:
                print(f"  {feat_name}:")
                print(f"    Files: {', '.join(methods)}")
                if comparison['correlations']:
                    for pair, corr in comparison['correlations'].items():
                        print(f"      Correlation ({pair}): {corr:.4f}")
                if comparison['mean_values']:
                    for method, mean_val in comparison['mean_values'].items():
                        print(f"      Mean ({method}): {mean_val:.4f}")
                print(f"    Values are similar (corr > 0.8): {comparison['are_similar']}")
                print()
        
        results['feature_comparisons'] = feature_comparisons
        
        if verbose:
            print("=" * 80)
            print("EXPLORATION COMPLETE")
            print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        results['error'] = str(e)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Explore audio data CSV files for a given participant, condition, phase, and aufgabe"
    )
    parser.add_argument(
        '--id',
        type=int,
        required=True,
        help='Participant ID (e.g., 4)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        choices=['CR', 'CRADK', 'ADK', 'SHAM'],
        help='Condition (CR, CRADK, ADK, SHAM)'
    )
    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['training_pos', 'training_neg', 'induction1', 'induction2'],
        help='Phase (training_pos, training_neg, induction1, induction2)'
    )
    parser.add_argument(
        '--aufgabe',
        type=int,
        required=True,
        help='Aufgabe number (integer, e.g., 1)'
    )
    parser.add_argument(
        '--id2',
        type=int,
        default=None,
        help='Optional second participant ID to compare'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data',
        help='Base directory containing OpenSmile_data and merged_RCT_info.csv'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory to save log file (default: same as data-dir)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Set up log file
    if args.log_dir is None:
        log_dir = data_dir
    else:
        log_dir = Path(args.log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"exploration_log_{args.id}_{args.condition}_{args.phase}_{args.aufgabe}_{timestamp}.txt"
    
    # Set up Tee to write to both stdout/stderr and log file
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    try:
        # Print header to log
        print("=" * 80)
        print("AUDIO DATA EXPLORATION LOG")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Participant ID: {args.id}")
        if args.id2 is not None:
            print(f"Participant ID 2: {args.id2}")
        print(f"Condition: {args.condition}")
        print(f"Phase: {args.phase}")
        print(f"Aufgabe: {args.aufgabe}")
        print(f"Log file: {log_filename}")
        print("=" * 80)
        print()
        
        # Explore first participant
        print("\n")
        results1 = explore_data(
            args.id, args.condition, args.phase, args.aufgabe, data_dir, verbose=True
        )
        
        # Explore second participant if provided
        if args.id2 is not None:
            print("\n\n")
            results2 = explore_data(
                args.id2, args.condition, args.phase, args.aufgabe, data_dir, verbose=True
            )
            
            # Compare dimensions
            if 'dimensions' in results1 and 'dimensions' in results2:
                print("\n" + "=" * 80)
                print("COMPARISON BETWEEN TWO PARTICIPANTS")
                print("=" * 80)
                print(f"\nParticipant {args.id}:")
                for method in sorted(results1['dimensions'].keys()):
                    dims = results1['dimensions'][method]
                    print(f"  {method:12s}: {dims['rows']:6,} rows × {dims['columns']:4d} features")
                print(f"\nParticipant {args.id2}:")
                for method in sorted(results2['dimensions'].keys()):
                    dims = results2['dimensions'][method]
                    print(f"  {method:12s}: {dims['rows']:6,} rows × {dims['columns']:4d} features")
                print("=" * 80)
        
        print("\n" + "=" * 80)
        print(f"Log saved to: {log_filename}")
        print("=" * 80)
        
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"\nLog saved to: {log_filename}")


if __name__ == "__main__":
    main()

