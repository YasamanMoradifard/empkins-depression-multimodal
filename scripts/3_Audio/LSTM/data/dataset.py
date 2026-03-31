"""
Data loading and PyTorch Dataset for audio time-series depression detection.

This module handles:
1. Loading subjects from pre-processed CSV files
2. PyTorch Dataset for loading and preprocessing CSV files per subject
3. Feature reduction (PCA/SelectKBest) for small datasets

Data structure:
- Processed per-participant CSVs live in: OpenSmile_data/{ID}/
- Participant metadata lives in: merged_RCT_info.csv
- CSV files are named: ID_diagnose_condition_phase_aufgabe.csv
  Example: 4_healthy_CR_induction1_1.csv

For a given (condition, phase), we:
  1) Select matching participants from merged_RCT_info.csv
  2) Find all their CSV files for that condition+phase in OpenSmile_data/{ID}/
  3) Build subject dictionaries with:
       - subject_id
       - file_paths (list of CSVs, one per file)
       - label (0=Healthy, 1=Depressed)
"""

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

# Type alias for subject dictionary
SubjectDict = Dict[str, Any]


# ============================================================================
# Data Loading Functions
# ============================================================================

def normalize_condition(condition: str) -> str:
    """
    Normalize condition name to match metadata values.
    
    Args:
        condition: Condition string (CR, CRADK, ADK, SHAM, All, or ALL)
    
    Returns:
        Normalized condition string
    """
    condition_upper = condition.upper()
    if condition_upper == 'ALL':
        return 'All'  # Special "all conditions" flag
    return condition  # Keep original case for CR, CRADK, ADK, SHAM


def diagnose_to_label(diagnose: str) -> int:
    """
    Convert Diagnose string to binary label (0=Healthy, 1=Depressed).
    
    Args:
        diagnose: Diagnosis string from metadata
    
    Returns:
        0 for Healthy, 1 for Depressed
    
    Raises:
        ValueError: If diagnose is NaN or unknown value
    """
    if pd.isna(diagnose):
        raise ValueError("Diagnose is NaN")

    diagnose_str = str(diagnose).strip().lower()
    if 'healthy' in diagnose_str:
        return 0
    if 'depressed' in diagnose_str:
        return 1
    raise ValueError(f"Unknown Diagnose value: {diagnose}")


def load_subjects_from_processed(condition: str,
                                 phase: str,
                                 processed_audio_dir: str,
                                 metadata_csv: str) -> List[SubjectDict]:
    """
    Load subjects from pre-processed CSV files in OpenSmile_data/{ID}/ folders.
    
    CSV files are expected to be named: ID_diagnose_condition_phase_aufgabe.csv
    Example: 4_healthy_CR_induction1_1.csv
    
    Args:
        condition: 'CR', 'CRADK', 'ADK', 'SHAM', 'All' or 'ALL'
        phase: 'training_pos', 'training_neg', 'induction1', 'induction2', 'all'
        processed_audio_dir: Root directory containing OpenSmile_data folder
                            (e.g., /Yasaman/Audio_data/Data/OpenSmile_data)
        metadata_csv: Path to merged_RCT_info.csv (for labels and conditions)
    
    Returns:
        List[SubjectDict] with 'subject_id', 'file_paths', and 'label'.
        Each SubjectDict contains:
        {
            'subject_id': str,  # e.g., "4"
            'file_paths': List[str],  # List of CSV file paths
            'label': int  # 0 for Healthy, 1 for Depressed
        }
    """
    # Load metadata
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    
    meta = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {len(meta)} participants from {metadata_path}")
    
    # Normalize condition and filter metadata
    normalized_condition = normalize_condition(condition)
    if normalized_condition != 'All':
        before = len(meta)
        meta = meta[meta['condition'] == normalized_condition].copy()
        logger.info(f"Filtered metadata to condition '{normalized_condition}': "
                    f"{len(meta)} participants (from {before})")
    
    if len(meta) == 0:
        logger.warning(f"No participants found for condition={condition}")
        return []
    
    # Setup paths
    processed_root = Path(processed_audio_dir)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed audio directory not found: {processed_root}")
    
    # Normalize phase for filename matching
    phase_lower = phase.lower()
    cond_lower = normalized_condition.lower()
    
    # Collect subjects and their files
    subjects_dict: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {'file_paths': [], 'label': None, 'subject_id': None}
    )
    
    for _, row in meta.iterrows():
        pid = row['ID']
        diagnose = row['Diagnose']
        
        # Convert diagnosis to label
        try:
            label = diagnose_to_label(diagnose)
        except ValueError as e:
            logger.warning(f"Skipping ID={pid}: {e}")
            continue
        
        subject_id = str(int(pid))
        subj_dir = processed_root / subject_id
        
        if not subj_dir.exists():
            logger.debug(f"Folder not found for ID={subject_id}: {subj_dir}")
            continue
        
        # Collect CSV files matching condition and phase
        file_paths: List[str] = []
        for csv_path in subj_dir.glob("*.csv"):
            filename = csv_path.name
            filename_lower = filename.lower()
            
            # Parse filename: ID_diagnose_condition_phase_aufgabe.csv
            # Examples:
            #   4_healthy_CR_induction1_1.csv → parts = ["4", "healthy", "CR", "induction1", "1"]
            #   4_healthy_CR_training_pos_14.csv → parts = ["4", "healthy", "CR", "training", "pos", "14"]
            # Note: phase can contain underscores (training_pos, training_neg)
            parts = filename_lower.replace('.csv', '').split('_')
            
            if len(parts) < 5:  # Need at least: ID, diagnose, condition, phase, aufgabe
                logger.debug(f"Skipping file with unexpected format: {filename}")
                continue
            
            # Extract components from filename
            # Format: {ID}_{diagnose}_{condition}_{phase}_{aufgabe}
            # - First 3 parts are always: ID, diagnose, condition
            # - Last part is always: aufgabe
            # - Everything in between is the phase (can have underscores)
            file_condition = parts[2] if len(parts) > 2 else None
            file_aufgabe = parts[-1]  # Last part is always aufgabe
            file_phase = '_'.join(parts[3:-1])  # Everything between condition and aufgabe is phase
            
            # Filter by condition (if not 'All')
            if normalized_condition != 'All':
                if file_condition != cond_lower:
                    continue
            
            # Filter by phase (if not 'all')
            if phase_lower != 'all':
                if file_phase != phase_lower:
                    continue
            
            file_paths.append(str(csv_path))
        
        if not file_paths:
            logger.debug(f"No matching files for ID={subject_id}, condition={condition}, phase={phase}")
            continue
        
        # Store subject data
        subj_entry = subjects_dict[subject_id]
        if subj_entry['subject_id'] is None:
            subj_entry['subject_id'] = subject_id
            subj_entry['label'] = label
        
        subj_entry['file_paths'].extend(sorted(file_paths))
    
    # Convert to list and filter out subjects with no files
    subjects: List[SubjectDict] = []
    for sid, data in subjects_dict.items():
        if len(data['file_paths']) > 0:
            subjects.append({
                'subject_id': data['subject_id'],
                'file_paths': sorted(set(data['file_paths'])),  # Remove duplicates and sort
                'label': data['label']
            })
        else:
            logger.warning(f"Skipping subject {sid}: no matching files found")
    
    logger.info(f"Loaded {len(subjects)} subjects for "
                f"condition='{condition}', phase='{phase}'")
    
    if subjects:
        n_depressed = sum(s['label'] for s in subjects)
        n_healthy = len(subjects) - n_depressed
        logger.info(f"Class distribution: {n_depressed} depressed, {n_healthy} healthy")
        
        # Log file count statistics
        file_counts = [len(s['file_paths']) for s in subjects]
        logger.info(f"Files per subject: min={min(file_counts)}, max={max(file_counts)}, "
                   f"mean={sum(file_counts)/len(file_counts):.1f}")
    
    return subjects


# ============================================================================
# PyTorch Dataset Class
# ============================================================================

class SubjectDataset(Dataset):
    """
    Dataset that loads per-subject file lists and returns file data.
    
    Each sample is a subject with multiple files. Each file is a time-series
    of shape (T, F) where T varies per file and F is the number of features.
    
    Args:
        subjects: List of SubjectDict from load_subjects_from_processed()
        normalize: Whether to normalize features per-file (zero mean, unit variance)
        handle_nan_inf: Whether to replace NaN/Inf values with feature mean per CSV
    """
    
    def __init__(self, 
                 subjects: List[Dict],
                 normalize: bool = True,
                 handle_nan_inf: bool = True,
                 downsample_factor: int = 1):
        """
        Args:
            subjects: List of SubjectDict from load_subjects_from_processed()
            normalize: Whether to normalize features per-file (zero mean, unit variance)
            handle_nan_inf: Whether to replace NaN/Inf values with feature mean per CSV
            downsample_factor: Downsampling factor (keep every Nth frame). 
                              Set to 1 to disable downsampling, 30 for 1/30 downsampling.
        """
        self.subjects = subjects
        self.normalize = normalize
        self.handle_nan_inf = handle_nan_inf
        self.downsample_factor = downsample_factor
        
        logger.info(f"Initialized dataset with {len(subjects)} subjects "
                   f"(normalize={normalize}, handle_nan_inf={handle_nan_inf}, "
                   f"downsample_factor={downsample_factor})")
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load all files for a subject.
        
        Returns:
            Dict with:
                - 'files': List[np.ndarray], each shape (T_i, F)
                - 'file_lengths': List[int], actual length of each file
                - 'label': int (0 or 1)
                - 'subject_id': str
        """
        subject = self.subjects[idx]
        file_paths = subject['file_paths']
        label = subject['label']
        subject_id = subject['subject_id']
        
        file_data = []
        file_lengths = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                # Exclude metadata columns (ID, diagnose, condition, phase, aufgabe)
                metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
                # Keep only numeric columns (features), excluding metadata
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in metadata_cols]
                data = df[feature_cols].values.astype(np.float32)
                
                # Apply downsampling (keep every Nth frame)
                if self.downsample_factor > 1 and len(data) > 0:
                    original_length = len(data)
                    data = data[::self.downsample_factor]  # Keep every Nth frame
                    if original_length != len(data):
                        logger.debug(f"Downsampled file from {original_length} to {len(data)} frames "
                                   f"(factor={self.downsample_factor})")
                
                # Skip empty files
                if len(data) == 0:
                    logger.warning(f"Empty CSV file (after downsampling): {file_path}, skipping")
                    continue
                
                # Handle NaN and Inf values (replace with mean per feature for this CSV)
                if self.handle_nan_inf:
                    data = data.copy()
                    n_features = data.shape[1]
                    for col_idx in range(n_features):
                        col = data[:, col_idx]
                        # Find valid values (not NaN and not Inf)
                        valid_mask = np.isfinite(col)
                        if valid_mask.sum() > 0:
                            # Compute mean of valid values
                            mean_value = np.mean(col[valid_mask])
                            # Replace NaN and Inf with mean
                            data[~valid_mask, col_idx] = mean_value
                        else:
                            # All values are NaN/Inf, replace with 0
                            logger.warning(f"Column {col_idx} has all NaN/Inf values, replacing with 0")
                            data[:, col_idx] = 0.0
                
                # Normalize per-file (zero mean, unit variance)
                if self.normalize:
                    mean = np.mean(data, axis=0, keepdims=True)
                    std = np.std(data, axis=0, keepdims=True)
                    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
                    data = (data - mean) / std
                
                file_data.append(data)
                file_lengths.append(len(data))
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}, skipping")
                continue
        
        if len(file_data) == 0:
            logger.error(f"No valid files loaded for subject {subject_id}")
            # Return dummy data - try to infer feature count from first file if available
            # Otherwise use a reasonable default (will be adjusted by model)
            # Note: 325 = 330 (total) - 5 (metadata columns: ID, diagnose, condition, phase, aufgabe)
            n_features = 188  # Fallback default (after excluding metadata)
            if file_paths:
                try:
                    df_test = pd.read_csv(file_paths[0])
                    # Exclude metadata columns when inferring feature count
                    metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
                    numeric_cols = df_test.select_dtypes(include=[np.number]).columns
                    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
                    n_features = len(feature_cols)
                except Exception:
                    pass  # Use default
            file_data = [np.zeros((1, n_features), dtype=np.float32)]
            file_lengths = [1]
        
        return {
            'files': file_data,
            'file_lengths': file_lengths,
            'label': label,
            'subject_id': subject_id
        }


class FeatureReducer:
    """
    Feature reduction using PCA or SelectKBest.
    
    Reduces the number of features to prevent overfitting on small datasets.
    MUST be fitted on training data only to avoid data leakage.
    """
    
    def __init__(self, method: str = 'pca', n_components: int = 30):
        """
        Args:
            method: 'pca' or 'select_k_best'
            n_components: Number of features/components to keep
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the reducer on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (required for select_k_best, optional for pca)
        """
        # Ensure n_components doesn't exceed actual features
        actual_n_components = min(self.n_components, X.shape[1], X.shape[0])
        
        if self.method == 'pca':
            self.reducer = PCA(n_components=actual_n_components)
            self.reducer.fit(X)
            explained_var = sum(self.reducer.explained_variance_ratio_) * 100
            logger.info(f"PCA fitted: {X.shape[1]} -> {actual_n_components} features "
                       f"(explains {explained_var:.1f}% variance)")
        elif self.method == 'select_k_best':
            if y is None:
                raise ValueError("select_k_best requires labels (y)")
            self.reducer = SelectKBest(f_classif, k=actual_n_components)
            self.reducer.fit(X, y)
            logger.info(f"SelectKBest fitted: {X.shape[1]} -> {actual_n_components} features")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted reducer."""
        if not self._fitted:
            raise RuntimeError("Reducer must be fitted before transform")
        return self.reducer.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


def fit_feature_reducer(dataset: 'SubjectDataset', 
                        method: str = 'pca',
                        n_components: int = 30) -> FeatureReducer:
    """
    Fit a feature reducer on all data in a dataset.
    
    Args:
        dataset: SubjectDataset instance
        method: 'pca' or 'select_k_best'
        n_components: Number of features to keep
    
    Returns:
        Fitted FeatureReducer
    """
    all_data = []
    all_labels = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label']
        for file_data in sample['files']:
            all_data.append(file_data)
            # Repeat label for each frame in the file
            all_labels.extend([label] * len(file_data))
    
    if len(all_data) == 0:
        raise ValueError("No data to fit reducer on")
    
    # Stack all data
    stacked_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    
    logger.info(f"Fitting feature reducer on {stacked_data.shape[0]} samples, "
               f"{stacked_data.shape[1]} features")
    
    reducer = FeatureReducer(method=method, n_components=n_components)
    reducer.fit(stacked_data, all_labels)
    
    return reducer


class ScaledDataset(Dataset):
    """
    Dataset wrapper that applies a fitted scaler to data on-the-fly.
    
    This ensures scalers are fitted only on training data to avoid data leakage.
    The scaler is applied when data is accessed, not when the dataset is created.
    """
    def __init__(self, base_dataset: SubjectDataset, scaler: object, 
                 feature_reducer: Optional[FeatureReducer] = None):
        """
        Args:
            base_dataset: Original dataset to wrap
            scaler: Fitted scaler (StandardScaler, MinMaxScaler, RobustScaler, etc.)
            feature_reducer: Optional fitted FeatureReducer for dimensionality reduction
        """
        self.base_dataset = base_dataset
        self.scaler = scaler
        self.feature_reducer = feature_reducer
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item and apply scaling (and optional feature reduction) on-the-fly.
        
        Args:
            idx: Index of the subject
            
        Returns:
            Dict with scaled file data, same structure as SubjectDataset
        """
        sample = self.base_dataset[idx]
        processed_files = []
        
        for file_data in sample['files']:
            # Apply scaler first
            scaled_file = self.scaler.transform(file_data)
            
            # Apply feature reduction if configured
            if self.feature_reducer is not None:
                scaled_file = self.feature_reducer.transform(scaled_file)
            
            processed_files.append(scaled_file.astype(np.float32))
        
        return {
            'files': processed_files,
            'file_lengths': sample['file_lengths'],
            'label': sample['label'],
            'subject_id': sample['subject_id']
        }
