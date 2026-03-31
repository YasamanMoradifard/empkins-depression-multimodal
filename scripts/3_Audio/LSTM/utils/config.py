"""
Configuration file for LSTM depression detection project.

See LSTM_ARCHITECTURE_GUIDE.md and LSTM_VISUAL_SUMMARY.md for architectural details.
"""

import os
from pathlib import Path

# ============================================================================
# Dataset Paths
# ============================================================================
# ----------------------------------------------------------------------------
# Option A: Use legacy manifest CSVs (one row per file, with path_audio, phase, etc.)
# ----------------------------------------------------------------------------
# MANIFESTS_DIR points to a directory containing condition-specific
# train/val/test CSV files (see comments below for expected structure).
#
#   manifests_dir/
#     ├── CR/
#     │   ├── CR_train.csv
#     │   ├── CR_validation.csv
#     │   └── CR_test.csv
#     ├── CRADK/
#     │   ├── CRADK_train.csv
#     │   ├── CRADK_validation.csv
#     │   └── CRADK_test.csv
#     ├── ADK/
#     ├── SHAM/
#     └── All/
#
# Each CSV file contains rows with: ID, condition, Diagnose, path_audio, phase, Aufgabe, etc.
# The path_audio column contains full paths to OpenSMILE feature CSV files.
MANIFESTS_DIR = os.getenv(
    'AUDIO_MANIFESTS_DIR',
    '/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests'
)

# ----------------------------------------------------------------------------
# Option B: Use processed per-participant CSVs from Audio_data/Data
# ----------------------------------------------------------------------------
# If USE_PROCESSED_AUDIO is True, the data loader will ignore MANIFESTS_DIR
# and instead:
#   1) Read participant metadata from MERGED_METADATA_CSV
#   2) For each matching subject ID, look in PROCESSED_AUDIO_DIR/{padded_id}/
#      for CSV files named: ID_diagnose_condition_phase_aufgabe.csv
#      (exactly the files created by Data/preparing_audio_data.py)
#
# You can override these paths with environment variables.
USE_PROCESSED_AUDIO = True

PROCESSED_AUDIO_DIR = os.getenv(
    'AUDIO_PROCESSED_DIR',
    '/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/OpenSmile_data'
)

MERGED_METADATA_CSV = os.getenv(
    'AUDIO_MERGED_METADATA_CSV',
    '/home/vault/empkins/tpD/D02/Students/Yasaman/Audio_data/Data/merged_RCT_info.csv'
)

# ============================================================================
# Model Architecture Hyperparameters
# ============================================================================
# Note: N_FEATURES is determined dynamically from CSV files at runtime
# This value includes metadata columns (ID, diagnose, condition, phase, aufgabe)
# The actual feature count (after excluding metadata) is ~325 and is auto-detected by dataset.py
N_FEATURES = 188  # Approximate/fallback value including metadata (actual is auto-detected from CSV)

# Model dimensions - SIGNIFICANTLY REDUCED for small dataset (~57 subjects)
# Rule of thumb: total params should be << 10x number of training samples
LSTM_HIDDEN_DIM = 32  # Reduced from 128 - much smaller for small dataset
FILE_EMBEDDING_DIM = 16  # Reduced from 64
SUBJECT_EMBEDDING_DIM = 16  # Reduced from 64
CLASSIFIER_HIDDEN_DIM = 8  # Reduced from 32
CLASSIFIER_HIDDEN_DIM2 = 4  # Reduced from 16

# Dropout - VERY HIGH for small dataset to prevent overfitting
DROPOUT_LSTM = 0.7  # Increased from 0.3 - very strong regularization
DROPOUT_CLASSIFIER = 0.7  # Increased from 0.5 - very strong regularization

# Aggregation method: 'attention', 'lstm', 'mean', 'max', 'mean_max'
# Using 'mean' instead of 'attention' to reduce parameters for small dataset
AGGREGATION_METHOD = 'mean'

# Use simple (single-layer) LSTM for small datasets
USE_SIMPLE_LSTM = True  # Set to False to use original 2-layer bidirectional LSTM

# Feature reduction settings
USE_FEATURE_REDUCTION = True  # Enable PCA/feature selection
FEATURE_REDUCTION_METHOD = 'pca'  # 'pca' or 'select_k_best'
N_FEATURES_REDUCED = 30  # Number of features after reduction (much smaller than 188)

# ============================================================================
# Training Hyperparameters
# ============================================================================
BATCH_SIZE = 2  # Very small for small dataset - helps with regularization
LEARNING_RATE = 1e-5  # Low learning rate for stability
WEIGHT_DECAY = 1e-3  # INCREASED - stronger L2 regularization for small dataset
NUM_EPOCHS = 150  # More epochs since we have strong regularization
EARLY_STOPPING_PATIENCE = 20  # Increased - allow more time with strong regularization
EARLY_STOPPING_MIN_DELTA = 0.001  # Very sensitive to small improvements
GRADIENT_CLIP_NORM = 1.0  # Tighter gradient clipping for stability
USE_MIXED_PRECISION = False  # Disabled for stability

# Learning rate scheduler settings
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by half when plateau
LR_SCHEDULER_PATIENCE = 7  # Wait 7 epochs before reducing
LR_SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate

# Loss function
USE_WEIGHTED_BCE = True  # Compute pos_weight from class distribution

# Label smoothing (helps with small datasets)
LABEL_SMOOTHING = 0.1  # Smooth labels: 0->0.1, 1->0.9

# ============================================================================
# Data Processing
# ============================================================================
# Downsampling: Reduce temporal resolution by keeping every Nth frame
# Set to 1 to disable downsampling
# Example: 30 means keep 1 frame per 30 frames (1 FPS from 30 FPS original)
# After 1/30 downsampling:
#   - Median: ~69 frames (2080/30)
#   - P75: ~82 frames (2446/30)
#   - P90: ~104 frames (3124/30)
#   - Mean: ~167 frames (5000/30)
DOWNSAMPLE_FACTOR = 1  # Set to 30 for 1/30 downsampling, 1 to disable

# MAX_SEQUENCE_LENGTH: Truncate sequences longer than this value (in frames).
# Based on data analysis:
#   - Median: ~2080 frames (before downsampling)
#   - P75: ~2446 frames  
#   - P90: ~3124 frames
#   - Mean: ~5000 frames
# Recommended: 3000 (captures ~90% of files) or None (no truncation)
# Note: If downsampling is enabled, adjust this value accordingly
MAX_SEQUENCE_LENGTH = None  # Truncate longer sequences
NORMALIZE_FEATURES = False
NORMALIZATION_MODE = 'per_file'  # 'per_file' or 'global'

# ============================================================================
# Cross-Validation
# ============================================================================
# Note: Pre-defined train/validation/test splits are available in manifests
# If using pre-defined splits, set USE_PREDEFINED_SPLITS = True
# Otherwise, cross-validation will create its own splits
USE_PREDEFINED_SPLITS = False  # Use train/val/test from manifests (recommended)
OUTER_CV_FOLDS = 5  # Only used if USE_PREDEFINED_SPLITS = False
INNER_CV_FOLDS = 3
RANDOM_SEED = 42

# Hyperparameter search grid (for inner CV)
# Reduced values for small dataset
HP_SEARCH_GRID = {
    'learning_rate': [1e-6, 1e-5, 5e-5],  # Lower LRs for stability
    'lstm_hidden_dim': [16, 32, 64],  # Smaller dimensions
    'dropout_lstm': [0.5, 0.7, 0.8],  # Higher dropout values
}

# ============================================================================
# Paths and Directories
# ============================================================================
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_INTERVAL = 10  # Log every N batches during training

# ============================================================================
# Gradient Monitoring
# ============================================================================
LOG_GRADIENTS = False  # Enable gradient statistics logging
GRADIENT_LOG_INTERVAL = 25  # Log gradients every N batches (0 = only first batch)
DIAGNOSE_GRADIENTS_EPOCHS = None # Epochs to run detailed gradient diagnosis

# ============================================================================
# Device
# ============================================================================
DEVICE = 'cuda'


