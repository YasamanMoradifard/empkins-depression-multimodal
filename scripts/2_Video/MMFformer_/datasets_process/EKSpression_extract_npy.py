"""
Feature extraction helper for the d02 (EKSpression) dataset.

This script extracts visual features from NPZ files and saves them:

    <output_root>/
        visual/<sample_id>_visual.npy   # Visual features (all features from NPZ)

The script reads paths from manifest CSV files in d02_manifests subfolders.
Visual features: All Visual features extracted from NPZ files.
Audio features: Processed separately using process_timeseries_audio() function.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

# Add scipy imports for filtering
try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available. Some filtering options will be disabled.")


def standardize(sequence: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Standardize a sequence to zero mean / unit variance along ``axis``. # Normalization

    Args:
        sequence: Input array shaped [time, features] or similar.
        axis: Axis along which to compute statistics.

    Returns:
        Standardized numpy array with same shape.
    """
    return preprocessing.scale(sequence, axis=axis)

def downsample(sequence: np.ndarray, every_n: int, method: str = "uniform") -> np.ndarray:
    """
    Down-sample the sequence using either uniform interval or averaging method.
    
    Args:
        sequence: Input array of shape (num_frames, num_features)
        every_n: Downsampling factor
        method: Downsampling method - "uniform" (keep every nth frame) or 
                "average" (average every n consecutive frames)
    
    Returns:
        Downsampled array
    """
    if every_n <= 1:
        return sequence
    
    if method == "average":
        # Average-based downsampling: average every n consecutive frames
        num_frames, num_features = sequence.shape
        
        # Calculate number of complete blocks
        num_blocks = num_frames // every_n
        remainder = num_frames % every_n
        
        # Reshape to (num_blocks, every_n, num_features) for complete blocks
        if num_blocks > 0:
            complete_blocks = sequence[:num_blocks * every_n].reshape(num_blocks, every_n, num_features)
            # Average along the middle axis (axis=1) to get (num_blocks, num_features)
            averaged_blocks = np.mean(complete_blocks, axis=1)
        else:
            averaged_blocks = np.array([]).reshape(0, num_features)
        
        # Handle remainder frames if any
        if remainder > 0:
            remainder_frames = sequence[num_blocks * every_n:]
            remainder_avg = np.mean(remainder_frames, axis=0, keepdims=True)
            
            if num_blocks > 0:
                # Concatenate averaged complete blocks with remainder average
                return np.concatenate([averaged_blocks, remainder_avg], axis=0)
            else:
                # Only remainder exists
                return remainder_avg
        
        return averaged_blocks
    
    else:  # method == "uniform" (default)
        # Uniform interval downsampling: take every nth frame
        return sequence[::every_n]


def apply_temporal_smoothing(
    sequence: np.ndarray,
    method: str = "moving_average",
    window_size: int = 5,
    poly_order: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply temporal smoothing to reduce jitter in landmark sequences.
    
    Args:
        sequence: Input array of shape (num_frames, num_features)
        method: Smoothing method - "moving_average", "savgol", "gaussian", or "none"
        window_size: Window size for moving average or Savitzky-Golay (must be odd for savgol)
        poly_order: Polynomial order for Savitzky-Golay filter (must be < window_size)
        sigma: Standard deviation for Gaussian filter
    
    Returns:
        Smoothed array with same shape as input
    """
    if method == "none" or method is None:
        return sequence
    
    num_frames, num_features = sequence.shape
    
    if method == "moving_average":
        # Simple moving average using convolution
        if window_size < 2:
            return sequence
        # Pad the sequence to handle boundaries
        pad_width = window_size // 2
        padded = np.pad(sequence, ((pad_width, pad_width), (0, 0)), mode='edge')
        # Create moving average kernel
        kernel = np.ones((window_size, 1)) / window_size
        # Apply 1D convolution along time axis for each feature
        smoothed = np.zeros_like(sequence)
        for i in range(num_features):
            smoothed[:, i] = np.convolve(padded[:, i], kernel[:, 0], mode='valid')
        return smoothed
    
    elif method == "savgol":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Savitzky-Golay filtering")
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        # Ensure window_size doesn't exceed sequence length
        window_size = min(window_size, num_frames if num_frames % 2 == 1 else num_frames - 1)
        if window_size < poly_order + 1:
            window_size = poly_order + 1
            if window_size % 2 == 0:
                window_size += 1
        
        smoothed = np.zeros_like(sequence)
        for i in range(num_features):
            smoothed[:, i] = savgol_filter(sequence[:, i], window_size, poly_order)
        return smoothed
    
    elif method == "gaussian":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Gaussian filtering")
        smoothed = np.zeros_like(sequence)
        for i in range(num_features):
            smoothed[:, i] = gaussian_filter1d(sequence[:, i], sigma=sigma)
        return smoothed
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Choose from: moving_average, savgol, gaussian, none")


def apply_polynomial_regression(
    sequence: np.ndarray,
    degree: int = 2,
    apply: bool = False,
) -> np.ndarray:
    """
    Apply polynomial regression to smooth landmark trajectories over time.
    
    Args:
        sequence: Input array of shape (num_frames, num_features)
        degree: Polynomial degree (2 for quadratic, 3 for cubic)
        apply: Whether to apply polynomial regression
    
    Returns:
        Smoothed array with same shape as input
    """
    if not apply or degree < 1:
        return sequence
    
    num_frames, num_features = sequence.shape
    
    # Create time axis
    t = np.arange(num_frames)
    
    # Fit polynomial for each feature
    smoothed = np.zeros_like(sequence)
    for i in range(num_features):
        # Fit polynomial
        coeffs = np.polyfit(t, sequence[:, i], degree)
        # Evaluate polynomial
        poly_func = np.poly1d(coeffs)
        smoothed[:, i] = poly_func(t)
    
    return smoothed


def apply_visual_filters(
    sequence: np.ndarray,
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 3,
    smoothing_sigma: float = 1.0,
    polynomial_degree: Optional[int] = None,
) -> np.ndarray:
    """
    Apply all visual filters to the sequence.
    
    Args:
        sequence: Input array of shape (num_frames, num_features)
        smoothing_method: Temporal smoothing method (moving_average, savgol, gaussian, none)
        smoothing_window: Window size for smoothing
        smoothing_poly_order: Polynomial order for Savitzky-Golay
        smoothing_sigma: Sigma for Gaussian filter
        polynomial_degree: Degree for polynomial regression (None to disable)
    
    Returns:
        Filtered array with same shape as input
    """
    # Apply temporal smoothing first
    if smoothing_method and smoothing_method != "none":
        sequence = apply_temporal_smoothing(
            sequence,
            method=smoothing_method,
            window_size=smoothing_window,
            poly_order=smoothing_poly_order,
            sigma=smoothing_sigma,
        )
    
    # Apply polynomial regression if requested
    if polynomial_degree is not None and polynomial_degree > 0:
        sequence = apply_polynomial_regression(
            sequence,
            degree=polynomial_degree,
            apply=True,
        )
    
    return sequence


def save_feature(
    array: np.ndarray,
    target_path: Path,
    standardize_axis: Optional[int],
    downsample_every: int,
) -> None:
    """
    Apply optional preprocessing and persist the feature.
    """
    if downsample_every > 1:
        array = downsample(array, downsample_every)
    if standardize_axis is not None:
        array = standardize(array, axis=standardize_axis)
    np.save(target_path, array.astype(np.float32))


def discover_npz_files(source_root: Path) -> Dict[str, Path]:
    """
    Walk ``source_root`` and return {sample_id: path_to_npz}.

    ``sample_id`` is derived from the stem (filename without extension).
    """
    mapping: Dict[str, Path] = {}
    for npz_path in source_root.rglob("*.npz"):
        mapping[npz_path.stem] = npz_path
    return mapping


def extract_video_from_npz(npz_path: Path) -> np.ndarray:
    """
    Extract 136 face landmark features (68 x-coordinates + 68 y-coordinates) from NPZ file.
    
    Args:
        npz_path: Path to NPZ file containing 'feature' and 'feature_names'
    
    Returns:
        Face landmark features array of shape (num_frames, 136)
        Order: x_0, x_1, ..., x_67, y_0, y_1, ..., y_67
    """
    with np.load(npz_path, allow_pickle=True) as data:
        features = data['feature']  # Shape: (num_frames, num_features)
        feature_names = data['feature_names']  # Shape: (num_features,)
        
        # Convert feature_names to list if it's a numpy array
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        
        # Build list of landmark feature names we want to extract
        landmark_names = []
        # Add x-coordinates: x_0, x_1, ..., x_67
        landmark_names.extend([f'x_{i}' for i in range(68)])
        # Add y-coordinates: y_0, y_1, ..., y_67
        landmark_names.extend([f'y_{i}' for i in range(68)])
        
        # Find indices of landmark features in feature_names
        landmark_indices = []
        for name in landmark_names:
            try:
                idx = feature_names.index(name)
                landmark_indices.append(idx)
            except ValueError:
                # If a landmark is missing, raise an error
                raise ValueError(
                    f"Landmark feature '{name}' not found in NPZ file {npz_path}. "
                    f"Available features: {feature_names[:10]}..."
                )
        
        # Extract only the landmark features
        landmark_features = features[:, landmark_indices].astype(np.float32)
        
        # Verify we got exactly 136 features
        if landmark_features.shape[1] != 136:
            raise ValueError(
                f"Expected 136 landmark features, but got {landmark_features.shape[1]}. "
                f"File: {npz_path}"
            )
        
        return landmark_features




def process_timeseries_audio(
    timeseries_root: Path,
    output_root: Path,
    output_subdir: str = "audio_timeseries",
) -> None:
    """
    Process TimeSeries audio features from participant folders.
    
    Structure expected:
    timeseries_root/
        <participant_id>/  (zero-padded, e.g., 004, 005)
            timeseries_opensmile_features/
                <phase_aufgabe_base>_compare16.csv
                <phase_aufgabe_base>_egemaps.csv
                <phase_aufgabe_base>_gemaps.csv
                <phase_aufgabe_base>_is09.csv
                <phase_aufgabe_base>_is13.csv
    
    For each phase/aufgabe combination, concatenates the 5 CSV files horizontally,
    drops non-numeric columns, and saves as NPY files to output_root/{output_subdir}/
    
    Args:
        timeseries_root: Root directory containing participant folders
        output_root: Root directory where output subdirectory will be created
        output_subdir: Name of the output subdirectory (default: "audio_timeseries")
    """
    audio_timeseries_dir = output_root / output_subdir
    audio_timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature set suffixes to look for
    feature_suffixes = ["_compare16.csv", "_egemaps.csv", "_gemaps.csv", "_is09.csv", "_is13.csv"]
    
    # Find all participant folders (zero-padded IDs)
    participant_dirs = sorted([d for d in timeseries_root.iterdir() if d.is_dir()])
    
    if not participant_dirs:
        print(f"[WARN] No participant folders found in {timeseries_root}")
        return
    
    print(f"[INFO] Processing {len(participant_dirs)} participant folders...")
    
    total_processed = 0
    total_skipped = 0
    
    for participant_dir in tqdm(participant_dirs, desc="Processing participants"):
        participant_id = participant_dir.name
        
        # Navigate to timeseries_opensmile_features folder
        timeseries_dir = participant_dir / "timeseries_opensmile_features"
        if not timeseries_dir.exists():
            print(f"[WARN] timeseries_opensmile_features not found for participant {participant_id}")
            continue
        
        # Find all CSV files
        csv_files = list(timeseries_dir.glob("*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found for participant {participant_id}")
            continue
        
        # Group CSV files by base name (phase/aufgabe combination)
        # Base name is everything before the feature suffix
        file_groups: Dict[str, List[Path]] = defaultdict(list)
        
        for csv_file in csv_files:
            filename = csv_file.stem  # filename without .csv extension
            # Try to match each feature suffix
            matched = False
            for suffix in feature_suffixes:
                suffix_base = suffix.replace(".csv", "")
                if filename.endswith(suffix_base):
                    base_name = filename[:-len(suffix_base)]
                    file_groups[base_name].append(csv_file)
                    matched = True
                    break
            
            if not matched:
                print(f"[WARN] CSV file doesn't match expected pattern: {csv_file.name}")
        
        # Process each group (phase/aufgabe combination)
        for base_name, csv_files_group in file_groups.items():
            # Sort files by feature suffix order to ensure consistent concatenation
            # Map each file to its suffix index for sorting
            def get_suffix_index(csv_file: Path) -> int:
                for idx, suffix in enumerate(feature_suffixes):
                    if csv_file.name.endswith(suffix):
                        return idx
                return len(feature_suffixes)  # Put unmatched files at the end
            
            csv_files_group.sort(key=get_suffix_index)
            
            # Check if we have all expected files and organize them in order
            ordered_files = []
            found_suffixes = set()
            for suffix in feature_suffixes:
                matching_file = None
                for csv_file in csv_files_group:
                    if csv_file.name.endswith(suffix):
                        matching_file = csv_file
                        found_suffixes.add(suffix)
                        break
                if matching_file:
                    ordered_files.append(matching_file)
            
            if len(found_suffixes) < len(feature_suffixes):
                missing = set(feature_suffixes) - found_suffixes
                print(f"[WARN] Missing feature files for {base_name} (participant {participant_id}): {missing}")
                total_skipped += 1
                continue
            
            try:
                # Load and concatenate all CSV files in the correct order
                dataframes = []
                for csv_file in ordered_files:
                    df = pd.read_csv(csv_file)
                    
                    # Drop non-numeric columns
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    # Skip 'name' and 'class' columns if they exist (they're usually non-numeric)
                    if 'name' in numeric_df.columns:
                        numeric_df = numeric_df.drop(columns=['name'])
                    if 'class' in numeric_df.columns:
                        numeric_df = numeric_df.drop(columns=['class'])
                    
                    dataframes.append(numeric_df)
                
                # Concatenate horizontally (axis=1)
                concatenated_df = pd.concat(dataframes, axis=1)
                
                # Additional check: drop any remaining non-numeric columns
                final_numeric_cols = []
                for col in concatenated_df.columns:
                    numeric_series = pd.to_numeric(concatenated_df[col], errors='coerce')
                    if not numeric_series.isna().any():
                        final_numeric_cols.append(col)
                
                final_df = concatenated_df[final_numeric_cols]
                
                # Convert to numpy array
                audio_features = final_df.values.astype(np.float32)
                
                # Create output filename: participant_id_base_name.npy
                output_filename = f"{participant_id}_{base_name}.npy"
                output_path = audio_timeseries_dir / output_filename
                
                # Save as NPY file
                np.save(output_path, audio_features)
                total_processed += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process {base_name} for participant {participant_id}: {e}")
                total_skipped += 1
                continue
    
    print(f"\n[INFO] Timeseries audio processing complete:")
    print(f"  Processed: {total_processed} phase/aufgabe combinations")
    print(f"  Skipped: {total_skipped}")
    print(f"  Output directory: {audio_timeseries_dir}")



def slugify(value: str) -> str:
    """Lowercase string that only keeps alphanumeric characters and underscores."""
    value = value.strip().lower()
    value = re.sub(r"[^\w]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


class TemplateArgs(dict):
    """Provide default empty string for missing template keys."""

    def __missing__(self, key):
        return ""


@dataclass
class SampleRecord:
    sample_id: str
    npz_path: Path
    metadata: Dict[str, str]


def load_manifest_rows(
    manifest_root: Path,
    subsets: Optional[Iterable[str]],
    splits: Iterable[str],
    id_template: str,
    path_column: str,
    replace_ext: Optional[str],
    npz_root: Optional[Path],
    conditions: Optional[Iterable[str]],
    phases: Optional[Iterable[str]],
) -> List[SampleRecord]:
    if not manifest_root:
        return []

    if subsets:
        subset_list = list(subsets)
    else:
        subset_list = sorted([p.name for p in manifest_root.iterdir() if p.is_dir()])

    split_list = list(splits)
    condition_set = {c.upper() for c in conditions} if conditions else None
    phase_set = {p for p in phases} if phases else None

    counters: Dict[str, int] = defaultdict(int)
    records: List[SampleRecord] = []

    def build_sample_id(row: Dict[str, str]) -> str:
        args = TemplateArgs({k: (row.get(k) or "") for k in row})
        formatted = id_template.format_map(args)
        base = slugify(formatted if formatted.strip() else row.get("stem", "sample"))
        idx = counters[base]
        counters[base] += 1
        if idx:
            base = f"{base}_{idx:02d}"
        return base

    if replace_ext:
        if ":" not in replace_ext:
            raise ValueError("--path-replace-ext must be in form OLD:NEW (e.g., .mp4:.npz)")
        old_ext, new_ext = replace_ext.split(":")
        if not old_ext.startswith("."):
            old_ext = f".{old_ext}"
        if not new_ext.startswith("."):
            new_ext = f".{new_ext}"
    else:
        old_ext = new_ext = ""

    def derive_npz_path(raw_path: str, stem: str) -> Path:
        path = Path(raw_path)
        if path.suffix.lower() != ".npz":
            if replace_ext and path.suffix.lower() == old_ext.lower():
                path = path.with_suffix(new_ext)
            elif npz_root is not None:
                path = Path(stem).with_suffix(".npz")
        if npz_root and not path.is_absolute():
            path = (npz_root / path).resolve()
        return path

    for subset in subset_list:
        for split in split_list:
            # Try both naming patterns: {subset}_{split}.csv and {split}.csv
            csv_path = manifest_root / subset / f"{subset}_{split}.csv"
            if not csv_path.exists():
                csv_path = manifest_root / subset / f"{split}.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                    row["subset"] = subset
                    row["split"] = split
                    cond = (row.get("condition") or subset or "").upper()
                    phase = row.get("phase", "")
                    if condition_set and cond not in condition_set:
                        continue
                    if phase_set and phase not in phase_set:
                        continue
                    raw_path = row.get(path_column) or row.get("path") or row.get("video_path") or row.get("path_video")
                    if not raw_path:
                        print(f"[WARN] Row missing '{path_column}' column: {row}")
                        continue
                    npz_path = derive_npz_path(raw_path, row.get("stem") or Path(raw_path).stem)
                    if not npz_path.exists():
                        print(f"[WARN] NPZ path not found ({npz_path}); skipping sample.")
                        continue
                    
                    sample_id = build_sample_id(row)
                    metadata = {
                        "label": row.get("label", "") or row.get("Diagnose", ""),
                        "pid": row.get("pid", "") or row.get("ID", ""),
                        "condition": row.get("condition", subset),
                        "phase": row.get("phase", ""),
                        "split": split,
                        "subset": subset,
                        "source_path": raw_path,
                    }
                    records.append(SampleRecord(sample_id, npz_path, metadata))
    return records


def run(
    output_root: Path,
    downsample_every: int = 1,
    standardize_axis: Optional[int] = 1,
    manifest_path: Optional[Path] = None,
    manifest_records: Optional[List[SampleRecord]] = None,
    log_file: Optional[Path] = None,
    # New filtering parameters
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 3,
    smoothing_sigma: float = 1.0,
    polynomial_degree: Optional[int] = None,
    # Downsampling method parameters
    downsample_method_visual: str = "uniform",
) -> None:
    """
    Extract all visual features from NPZ files and save them in visual/ subdirectory.
    
    Args:
        output_root: Directory where visual/ folder will be created
        downsample_every: Downsample frames (keep every Nth frame)
        standardize_axis: Axis for standardization (1 = per frame, None = disabled)
        manifest_path: Optional JSON file to save metadata
        manifest_records: List of SampleRecord objects from manifest CSV files
        log_file: Optional path to log file for tracking missing files and errors
        smoothing_method: Temporal smoothing method (moving_average, savgol, gaussian, none)
        smoothing_window: Window size for smoothing
        smoothing_poly_order: Polynomial order for Savitzky-Golay
        smoothing_sigma: Sigma for Gaussian filter
        polynomial_degree: Degree for polynomial regression (None to disable)
        downsample_method_visual: Downsampling method for visual features ("uniform" or "average")
    """
    output_root.mkdir(parents=True, exist_ok=True)
    visual_dir = output_root / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, str]] = {}

    if not manifest_records:
        raise ValueError("manifest_records must be provided. Use --manifests-root to load from CSV files.")

    # Initialize log file if requested
    log_fh = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_file.open("w", encoding="utf-8")
        log_fh.write(f"Feature Extraction Run Log\n")
        log_fh.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write(f"Output root: {output_root}\n")
        if smoothing_method:
            log_fh.write(f"Smoothing method: {smoothing_method}\n")
            log_fh.write(f"Smoothing window: {smoothing_window}\n")
        if polynomial_degree:
            log_fh.write(f"Polynomial degree: {polynomial_degree}\n")
        log_fh.write("=" * 80 + "\n\n")

    missing_npz_count = 0
    processing_errors_count = 0

    for rec in tqdm(manifest_records, desc="Extracting features"):
        ##IMPORTANT: Each sample is processed completely independently
        sample_id = rec.sample_id
        npz_path = rec.npz_path
        metadata = rec.metadata

        try:
            # Check if NPZ file exists
            if not npz_path.exists():
                missing_npz_count += 1
                error_msg = f"[MISSING NPZ] Sample ID: {sample_id}\n"
                error_msg += f"  NPZ path: {npz_path}\n"
                error_msg += f"  Expected path: {npz_path.resolve()}\n"
                error_msg += f"  Metadata: {metadata}\n"
                print(f"[WARN] NPZ file not found for {sample_id}: {npz_path}")
                if log_fh:
                    log_fh.write(error_msg + "\n")
                # NPZ is required - skip this sample
                continue
            
            # Extract all visual features from NPZ
            visual_features = extract_video_from_npz(npz_path)
            
            # Apply visual filters if specified
            visual_features = apply_visual_filters(
                visual_features,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                smoothing_poly_order=smoothing_poly_order,
                smoothing_sigma=smoothing_sigma,
                polynomial_degree=polynomial_degree,
            )
            
            # Apply preprocessing
            if downsample_every > 1:
                visual_features = downsample(visual_features, downsample_every, method=downsample_method_visual)
            
            if standardize_axis is not None:
                visual_features = standardize(visual_features, axis=standardize_axis)
            
            # Save visual features
            visual_output = visual_dir / f"{sample_id}_visual.npy"
            np.save(visual_output, visual_features.astype(np.float32))
            
            entry = {
                "npz": str(npz_path.resolve()),
                "visual": str(visual_output.resolve()),
                "visual_shape": list(visual_features.shape),
            }
            entry.update(metadata)
            manifest[sample_id] = entry
            
        except Exception as e:
            processing_errors_count += 1
            error_msg = f"[PROCESSING ERROR] Sample ID: {sample_id}\n"
            error_msg += f"  Error: {str(e)}\n"
            error_msg += f"  NPZ path: {npz_path}\n"
            error_msg += f"  Metadata: {metadata}\n"
            print(f"[ERROR] Failed to process {sample_id}: {e}")
            if log_fh:
                log_fh.write(error_msg + "\n")
            continue

    # Write summary to log file
    if log_fh:
        log_fh.write("=" * 80 + "\n")
        log_fh.write("SUMMARY\n")
        log_fh.write("=" * 80 + "\n")
        log_fh.write(f"Total samples processed: {len(manifest_records)}\n")
        log_fh.write(f"Successfully processed: {len(manifest)}\n")
        log_fh.write(f"Missing NPZ files: {missing_npz_count}\n")
        log_fh.write(f"Processing errors: {processing_errors_count}\n")
        log_fh.write(f"Skipped samples: {len(manifest_records) - len(manifest)}\n")
        log_fh.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.close()
        print(f"\n[INFO] Log file saved to: {log_file}")

    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract visual features from NPZ files and save them in visual/ folder. "
                    "Audio features are processed separately using --timeseries-audio-root."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for visual/ and audio/ folders.",
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        help="Root folder containing split CSV manifests (e.g., d02_manifests).",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        help="Specific subset folders to use (default: all found under manifests-root).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "validation", "test"),
        help="Split names to load from each subset directory (default: train, validation, test).",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional list of conditions (CR, ADK, ...) to include.",
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        help="Optional list of phases to include.",
    )
    parser.add_argument(
        "--path-column",
        default="path_video",
        help="CSV column that points to the NPZ file path (default: path_video).",
    )
    parser.add_argument(
        "--path-replace-ext",
        default=".mp4:.npz",
        help="When the path column points to videos, replace the extension using OLD:NEW.",
    )
    parser.add_argument(
        "--npz-root",
        type=Path,
        help="Optional base directory used when derived NPZ paths are relative.",
    )
    parser.add_argument(
        "--id-template",
        default="{ID}_{Diagnose}_{condition}_{phase}_{Aufgabe}",
        help="Template used to build sample ids. Fields come from manifest columns. "
             "Default: {ID}_{Diagnose}_{condition}_{phase}_{Aufgabe}",
    )
    parser.add_argument(
        "--downsample-every",
        type=int,
        default=30,
        help="Keep one frame out of N along the time axis (>=1).",
    )
    parser.add_argument(
        "--downsample-method-visual",
        type=str,
        choices=["uniform", "average"],
        default="uniform",
        help="Downsampling method for visual features: 'uniform' (take every nth frame) or 'average' (average every n frames). Default: uniform",
    )
    parser.add_argument(
        "--standardize-axis",
        type=int,
        default=1,
        help=(
            "Axis passed to sklearn.preprocessing.scale. "
            "Use -1 to disable standardization."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON file storing the mapping from sample IDs to outputs.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to log file for tracking missing files and errors.",
    )
    # New filtering arguments
    parser.add_argument(
        "--smoothing-method",
        type=str,
        choices=["none", "moving_average", "savgol", "gaussian"],
        default=None,
        help="Temporal smoothing method for visual features: moving_average, savgol (Savitzky-Golay), gaussian, or none (default: none)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for temporal smoothing (default: 5). Must be odd for Savitzky-Golay.",
    )
    parser.add_argument(
        "--smoothing-poly-order",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter (default: 3). Must be < window_size.",
    )
    parser.add_argument( # Not considered
        "--smoothing-sigma",
        type=float,
        default=1.0,
        help="Sigma parameter for Gaussian filter (default: 1.0).",
    )
    parser.add_argument(
        "--polynomial-degree",
        type=int,
        default=None,
        help="Apply polynomial regression with given degree (2=quadratic, 3=cubic). None to disable (default: None).",
    )
    parser.add_argument(
        "--timeseries-audio-root",
        type=Path,
        help="Root directory containing participant folders with timeseries_opensmile_features. "
             "If provided, will process and concatenate CSV files by phase/aufgabe.",
    )
    parser.add_argument(
        "--timeseries-vad-audio-root",
        type=Path,
        help="Root directory containing participant folders with timeseries_opensmile_features. "
             "If provided, will process and concatenate CSV files by phase/aufgabe.",
    )
    parser.add_argument(
        "--participant-ids",
        nargs="*",
        help="List of participant IDs to process (e.g., --participant-ids 114 115 121). "
             "Can also be a path to a CSV file with 'ID' or 'Normalized_ID' column.",
    )
    parser.add_argument(
        "--filter-phases",
        nargs="*",
        help="Only process specific phases (e.g., --filter-phases induction1 induction2). "
             "Useful for processing only Belastungsphase files.",
    )
    return parser.parse_args()


def load_participant_ids_from_csv(csv_path: Path) -> List[str]:
    """
    Load participant IDs from a CSV file.
    Looks for 'ID' or 'Normalized_ID' column.
    Returns unique participant IDs only.
    
    Returns:
        List of unique participant IDs as strings
    """
    try:
        df = pd.read_csv(csv_path)
        # Try to find ID column
        id_col = None
        for col in df.columns:
            if col.lower() in ('id', 'normalized_id', 'participant_id'):
                id_col = col
                break
        
        if id_col is None:
            raise ValueError(f"Could not find ID column in CSV: {csv_path}")
        
        # Get unique IDs and convert to strings
        unique_ids = df[id_col].dropna().unique()
        ids = []
        seen = set()
        for id_val in unique_ids:
            try:
                # Normalize ID (remove leading zeros)
                normalized = str(int(float(id_val)))
                if normalized not in seen:
                    ids.append(normalized)
                    seen.add(normalized)
            except (ValueError, TypeError):
                # Skip invalid IDs
                continue
        
        return ids
    except Exception as e:
        raise ValueError(f"Failed to load participant IDs from CSV {csv_path}: {e}")


def normalize_pid_for_filtering(pid: str) -> str:
    """Normalize participant ID by removing leading zeros for comparison."""
    try:
        return str(int(float(pid)))
    except (ValueError, TypeError):
        return str(pid).strip()


def filter_manifest_records_by_participant_ids(
    records: List[SampleRecord],
    participant_ids: List[str]
) -> List[SampleRecord]:
    """
    Filter manifest records to ONLY include specified participant IDs.
    All other participant IDs are excluded.
    
    Args:
        records: List of SampleRecord objects
        participant_ids: List of participant IDs (can be strings or numbers)
    
    Returns:
        Filtered list of SampleRecord objects containing ONLY the specified participant IDs
    """
    if not participant_ids:
        return records
    
    # Normalize participant IDs for comparison
    normalized_target_ids = {normalize_pid_for_filtering(str(pid)) for pid in participant_ids}
    
    filtered_records = []
    skipped_pids = set()
    
    for rec in records:
        # Extract participant ID from metadata
        pid = rec.metadata.get("pid", "") or rec.metadata.get("ID", "")
        if pid:
            normalized_pid = normalize_pid_for_filtering(str(pid))
            if normalized_pid in normalized_target_ids:
                filtered_records.append(rec)
            else:
                # Track which IDs were skipped (for validation)
                skipped_pids.add(normalized_pid)
        else:
            # If no PID found in metadata, skip it
            print(f"[WARN] Record {rec.sample_id} has no participant ID in metadata, skipping")
    
    # Validation: Report if any target IDs were not found
    found_pids = {normalize_pid_for_filtering(str(rec.metadata.get("pid", "") or rec.metadata.get("ID", ""))) 
                   for rec in filtered_records}
    missing_pids = normalized_target_ids - found_pids
    if missing_pids:
        print(f"[WARN] The following participant IDs from the filter list were not found in manifest records: {sorted(missing_pids)}")
    
    return filtered_records


def filter_manifest_records_by_phases(
    records: List[SampleRecord],
    phases: List[str]
) -> List[SampleRecord]:
    """
    Filter manifest records to only include specified phases.
    
    Args:
        records: List of SampleRecord objects
        phases: List of phase names to include
    
    Returns:
        Filtered list of SampleRecord objects
    """
    if not phases:
        return records
    
    phase_set = {p.lower() for p in phases}
    
    filtered_records = []
    for rec in records:
        phase = rec.metadata.get("phase", "").lower()
        if phase in phase_set:
            filtered_records.append(rec)
    
    return filtered_records


if __name__ == "__main__":
    args = parse_args()
    standardize_axis = args.standardize_axis if args.standardize_axis >= 0 else None

    manifest_records = None
    if args.manifests_root:
        manifest_records = load_manifest_rows(
            manifest_root=args.manifests_root,
            subsets=args.subsets,
            splits=args.splits,
            id_template=args.id_template,
            path_column=args.path_column,
            replace_ext=args.path_replace_ext,
            npz_root=args.npz_root,
            conditions=args.conditions,
            phases=args.phases,
        )
        
        # Filter by participant IDs if specified
        if args.participant_ids:
            participant_ids = []
            for item in args.participant_ids:
                # Check if it's a file path
                item_path = Path(item)
                if item_path.exists() and item_path.suffix.lower() == '.csv':
                    # Load IDs from CSV file
                    csv_ids = load_participant_ids_from_csv(item_path)
                    participant_ids.extend(csv_ids)
                    print(f"[INFO] Loaded {len(csv_ids)} unique participant IDs from {item_path}")
                    print(f"[INFO] Participant IDs to process: {sorted(set(participant_ids))}")
                else:
                    # Treat as direct ID
                    participant_ids.append(item)
            
            if participant_ids:
                # Remove duplicates while preserving order
                unique_participant_ids = list(dict.fromkeys(participant_ids))
                original_count = len(manifest_records)
                
                print(f"\n[INFO] FILTERING: Processing ONLY participant IDs: {sorted(unique_participant_ids)}")
                print(f"[INFO] All other participant IDs will be EXCLUDED from processing.")
                
                manifest_records = filter_manifest_records_by_participant_ids(
                    manifest_records, unique_participant_ids
                )
                
                print(f"[INFO] Filtered to {len(manifest_records)} records for {len(unique_participant_ids)} participant IDs (from {original_count} total)")
                print(f"[INFO] ✓ Only specified participant IDs will be processed. All others excluded.\n")
            else:
                print("[WARN] No participant IDs found after loading. Processing all records.")
        
        # Filter by phases if specified (separate from --phases which filters at manifest loading)
        if args.filter_phases:
            original_count = len(manifest_records)
            manifest_records = filter_manifest_records_by_phases(
                manifest_records, args.filter_phases
            )
            print(f"[INFO] Filtered to {len(manifest_records)} records for phases: {args.filter_phases} (from {original_count} total)")

    # Note: manifest_records can be None if only processing timeseries audio

    # Process timeseries audio if requested
    if args.timeseries_audio_root:
        print("\n[INFO] Processing timeseries audio features...")
        process_timeseries_audio(
            timeseries_root=args.timeseries_audio_root,
            output_root=args.output_root,
            output_subdir="audio_timeseries",
        )
        print("[INFO] Timeseries audio processing completed.\n")

    if args.timeseries_vad_audio_root:
        print("\n[INFO] Processing timeseries VAD audio features...")
        process_timeseries_audio(
            timeseries_root=args.timeseries_vad_audio_root,
            output_root=args.output_root,
            output_subdir="audio_timeseries_vad",
        )
        print("[INFO] Timeseries VAD audio processing completed.\n")
        
    # Set up log file path
    log_file = args.log_file

    if manifest_records:
        # Print filtering information
        if args.smoothing_method:
            print(f"\n[INFO] Applying temporal smoothing: {args.smoothing_method}")
            print(f"  Window size: {args.smoothing_window}")
            if args.smoothing_method == "savgol":
                print(f"  Polynomial order: {args.smoothing_poly_order}")
            elif args.smoothing_method == "gaussian":
                print(f"  Sigma: {args.smoothing_sigma}")
        if args.polynomial_degree:
            print(f"[INFO] Applying polynomial regression: degree {args.polynomial_degree}")
        
        run(
            output_root=args.output_root,
            downsample_every=max(1, args.downsample_every),
            standardize_axis=standardize_axis,
            manifest_path=args.manifest,
            manifest_records=manifest_records,
            log_file=log_file,
            smoothing_method=args.smoothing_method,
            smoothing_window=args.smoothing_window,
            smoothing_poly_order=args.smoothing_poly_order,
            smoothing_sigma=args.smoothing_sigma,
            polynomial_degree=args.polynomial_degree,
            downsample_method_visual=args.downsample_method_visual,
        )
    elif not (args.timeseries_audio_root or args.timeseries_vad_audio_root):
        raise ValueError(
            "No manifest records found and no timeseries saudio root provided. "
            "Please provide --manifests-root, --timeseries-audio-root, or --timeseries-vad-audio-root."
        )
# Runing Samples    




""" Video - No downsampling, no smoothing
python datasets_process/EKSpression_extract_npy.py \
--output-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy \
--manifests-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests \
--subsets SHAM \
--downsample-every -1 \
--standardize-axis 1 \
--splits train validation test \
--log-file /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy/extraction_log.txt
"""

""" Video - 30 downsampling avg, no smoothing
python datasets_process/EKSpression_extract_npy.py \
--output-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30 \
--manifests-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests \
--subsets All \
--downsample-every 30 \
--downsample-method-visual average \
--standardize-axis 1 \
--splits train validation test \
--log-file /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30/extraction_log.txt
"""


""" Video - 30 downsampling uniform, no smoothing
python datasets_process/EKSpression_extract_npy.py \
--output-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_uniform \
--manifests-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests \
--subsets All \
--downsample-every 30 \
--downsample-method-visual uniform \
--standardize-axis 1 \
--splits train validation test \
--log-file /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_uniform/extraction_log.txt
"""


""" Video - 30 downsampling uniform, Savitzky-Golay smoothing
python datasets_process/EKSpression_extract_npy.py \
--output-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_uniform_savgol \
--manifests-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests \
--subsets All \
--downsample-every 30 \
--downsample-method-visual uniform \
--smoothing-method savgol \
--smoothing-window 5 \
--smoothing-poly-order 3 \
--standardize-axis 1 \
--splits train validation test \
--log-file /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_uniform_savgol/extraction_log.txt
"""


""" Video - 30 downsampling, Savitzky-Golay smoothing
python datasets_process/EKSpression_extract_npy.py \
--output-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_savgol \
--manifests-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests \
--subsets All \
--downsample-every 30 \
--downsample-method-visual average \
--smoothing-method savgol \
--smoothing-window 5 \
--smoothing-poly-order 3 \
--standardize-axis 1 \
--splits train validation test \
--log-file /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy_downsampled_30_savgol/extraction_log.txt
"""