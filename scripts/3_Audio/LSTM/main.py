"""
Main script for LSTM depression detection with nested cross-validation.

This script implements:
- Nested cross-validation (outer and inner loops)
- Proper data preprocessing with imputation and scaling (avoiding data leakage)
- Hyperparameter configuration via command-line arguments
- Model training and evaluation
- Visualization of results (ROC curves, confusion matrices, training history)
- Final results as mean ± std across all test folds
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Local imports
from data.dataset import load_subjects_from_processed, SubjectDataset, ScaledDataset, FeatureReducer, fit_feature_reducer
from data.collate import collate_fn
from models.hierarchical_lstm import HierarchicalLSTMDepression
from training.train import train_model
from training.evaluate import evaluate_model
from utils.config import (
    PROCESSED_AUDIO_DIR, MERGED_METADATA_CSV, RESULTS_DIR,
    LSTM_HIDDEN_DIM, FILE_EMBEDDING_DIM, SUBJECT_EMBEDDING_DIM,
    CLASSIFIER_HIDDEN_DIM, CLASSIFIER_HIDDEN_DIM2,
    DROPOUT_LSTM, DROPOUT_CLASSIFIER,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    WEIGHT_DECAY, GRADIENT_CLIP_NORM, USE_MIXED_PRECISION,
    USE_WEIGHTED_BCE,
    HP_SEARCH_GRID, LOG_GRADIENTS, GRADIENT_LOG_INTERVAL,
    DIAGNOSE_GRADIENTS_EPOCHS, DOWNSAMPLE_FACTOR,
    USE_SIMPLE_LSTM, USE_FEATURE_REDUCTION, FEATURE_REDUCTION_METHOD,
    N_FEATURES_REDUCED, USE_LR_SCHEDULER, LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_MIN_LR, LABEL_SMOOTHING
)
from utils.metrics import compute_confusion_matrix
from utils.visualization import plot_training_metrics_separate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fit_scaler_on_dataset(dataset: SubjectDataset, scaler_type: str = 'standard') -> object:
    """
    Fit a scaler on all data in a dataset.
    
    Args:
        dataset: SubjectDataset instance
        scaler_type: 'standard', 'minmax', or 'robust'
    
    Returns:
        Fitted scaler
    """
    all_data = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        for file_data in sample['files']:
            all_data.append(file_data)
    
    if len(all_data) == 0:
        raise ValueError("No data to fit scaler on")
    
    # Stack all data
    stacked_data = np.vstack(all_data)
    
    # Create and fit scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    scaler.fit(stacked_data)
    logger.info(f"Fitted {scaler_type} scaler on {stacked_data.shape[0]} samples, {stacked_data.shape[1]} features")
    
    return scaler


def detect_n_features(subjects: List[Dict], sample_size: int = 5) -> int:
    """
    Detect number of features from sample CSV files.
    
    Args:
        subjects: List of SubjectDict
        sample_size: Number of files to sample for detection
    
    Returns:
        Number of features (excluding metadata columns)
    """
    
    metadata_cols = ['ID', 'diagnose', 'condition', 'phase', 'aufgabe']
    feature_counts = []
    
    files_checked = 0
    for subject in subjects:
        for file_path in subject['file_paths']:
            if files_checked >= sample_size:
                break
            try:
                df = pd.read_csv(file_path)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in metadata_cols]
                feature_counts.append(len(feature_cols))
                files_checked += 1
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        if files_checked >= sample_size:
            break
    
    if not feature_counts:
        logger.warning("Could not detect feature count, using default 188")
        return 188
    
    # Use most common feature count
    unique_counts, counts = np.unique(feature_counts, return_counts=True)
    n_features = unique_counts[np.argmax(counts)]
    logger.info(f"Detected {n_features} features (from {len(feature_counts)} sample files)")
    return int(n_features)



def cv_train_eval(
    subjects: List[Dict],
    n_folds: int = 3,
    scaler_type: str = 'standard',
    aggregation_method: str = 'attention',
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 120,
    early_stopping_patience: int = 10,
    lstm_hidden_dim: int = 128,
    file_embedding_dim: int = 64,
    subject_embedding_dim: int = 64,
    dropout_lstm: float = 0.3,
    dropout_classifier: float = 0.5,
    use_weighted_bce: bool = True,
    device: torch.device = None,
    results_dir: Path = RESULTS_DIR,
    random_seed: int = 42,
    downsample_factor: int = 1,
    use_simple_lstm: bool = True,
    use_feature_reduction: bool = False,
    feature_reduction_method: str = 'pca',
    n_features_reduced: int = 30,
    use_lr_scheduler: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 7,
    lr_scheduler_min_lr: float = 1e-7,
    label_smoothing: float = 0.0
) -> Tuple[Dict, List[Dict]]:
    """
    Perform single-level cross-validation training and evaluation.
    
    For each fold:
    1. Split subjects into train and test (subject-level StratifiedGroupKFold)
    2. Split train into train and val for early stopping
    3. Fit scaler ONLY on training data
    4. Train model on training data (with that scaler)
    5. Evaluate on test data (using the SAME scaler)
    
    This ensures no data leakage and consistent scaling between train and test.
    
    Args:.
        subjects: List of SubjectDict
        n_folds: Number of CV folds
        scaler_type: 'standard', 'minmax', or 'robust'
        aggregation_method: 'attention', 'lstm', 'mean', 'max', 'mean_max'
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        lstm_hidden_dim: LSTM hidden dimension
        file_embedding_dim: File embedding dimension
        subject_embedding_dim: Subject embedding dimension
        dropout_lstm: LSTM dropout rate
        dropout_classifier: Classifier dropout rate
        use_weighted_bce: Use weighted BCE loss
        device: torch.device to use
        results_dir: Directory to save results
        random_seed: Random seed
    
    Returns:
        Tuple of (final_results_dict, all_test_results_list)
    """
    # Ensure GPU is available - fail if not
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is required but not available. Please ensure GPU is available and PyTorch is installed with CUDA support.")
        device = torch.device('cuda')
    
    # Verify device is GPU
    if device.type != 'cuda':
        raise RuntimeError(f"GPU (CUDA) is required but device is set to {device.type}. Please ensure GPU is available.")
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is required but not available. Please ensure GPU is available and PyTorch is installed with CUDA support.")
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Detect number of features
    n_features = detect_n_features(subjects)
    
    # Prepare data for CV
    subject_ids = np.array([s['subject_id'] for s in subjects])
    labels = np.array([s['label'] for s in subjects])
    
    # Use StratifiedGroupKFold for subject-level splitting with class balance
    # Signature: split(X, y, groups) where X can be indices or None
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(len(subjects))
    
    all_test_results = []
    all_test_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(indices, labels, subject_ids)):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*80}")
        
        # Split into train and test
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        test_labels = labels[test_idx]
        train_labels = labels[train_idx]
        
        logger.info(f"Train: {len(train_subjects)} subjects ({np.sum(train_labels==0)} healthy, {np.sum(train_labels==1)} depressed)")
        logger.info(f"Test: {len(test_subjects)} subjects ({np.sum(test_labels==0)} healthy, {np.sum(test_labels==1)} depressed)")
        
        # Split train into train and val for early stopping
        # Use 67/33 split for train/val (better for small datasets - gives ~12-13 val subjects instead of 8)
        # This provides more stable validation metrics and better AUC granularity
        # StratifiedGroupKFold(n_splits=3) gives 3 folds: 2 parts train, 1 part val = 67/33
        # with class balance maintained
        train_val_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_seed + fold)
        train_subject_ids = np.array([s['subject_id'] for s in train_subjects])
        train_subject_labels = np.array([s['label'] for s in train_subjects])
        train_indices = np.arange(len(train_subjects))
        
        # Get first split (67% train, 33% val) with class balance
        inner_train_idx, inner_val_idx = next(train_val_sgkf.split(train_indices, train_subject_labels, train_subject_ids))
        
        inner_train_subjects = [train_subjects[i] for i in inner_train_idx]
        inner_val_subjects = [train_subjects[i] for i in inner_val_idx]
        
        logger.info(f"  Inner split: {len(inner_train_subjects)} train, {len(inner_val_subjects)} val")
        
        # Create datasets WITHOUT normalization (we'll use global scaler)
        train_dataset = SubjectDataset(inner_train_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        val_dataset = SubjectDataset(inner_val_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        test_dataset = SubjectDataset(test_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        
        # Fit scaler ONLY on training data
        logger.info(f"Fitting {scaler_type} scaler on training data...")
        scaler = fit_scaler_on_dataset(train_dataset, scaler_type=scaler_type)
        
        # Fit feature reducer ONLY on training data (if enabled)
        feature_reducer = None
        if use_feature_reduction:
            logger.info(f"Fitting {feature_reduction_method} feature reducer to {n_features_reduced} features...")
            feature_reducer = fit_feature_reducer(train_dataset, 
                                                   method=feature_reduction_method,
                                                   n_components=n_features_reduced)
        
        # Apply the SAME scaler (and reducer) to train, val, and test
        train_dataset_scaled = ScaledDataset(train_dataset, scaler, feature_reducer)
        val_dataset_scaled = ScaledDataset(val_dataset, scaler, feature_reducer)
        test_dataset_scaled = ScaledDataset(test_dataset, scaler, feature_reducer)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset_scaled,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset_scaled,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset_scaled,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Determine effective number of features (after optional reduction)
        effective_n_features = n_features_reduced if use_feature_reduction else n_features
        
        # Create model
        model = HierarchicalLSTMDepression(
            n_features=effective_n_features,
            lstm_hidden=lstm_hidden_dim,
            file_embedding=file_embedding_dim,
            subject_embedding=subject_embedding_dim,
            dropout=dropout_lstm,
            aggregation_method=aggregation_method,
            classifier_hidden=CLASSIFIER_HIDDEN_DIM,
            classifier_hidden2=CLASSIFIER_HIDDEN_DIM2,
            classifier_dropout=dropout_classifier,
            use_simple_lstm=use_simple_lstm
        ).to(device)
        
        # Log model info
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {n_params:,} trainable parameters")
        logger.info(f"Using {'simple' if use_simple_lstm else 'full'} LSTM architecture")
        
        # Train model
        fold_results_dir = results_dir / f"fold_{fold}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)
        
        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            use_mixed_precision=USE_MIXED_PRECISION,
            use_weighted_bce=use_weighted_bce,
            save_dir=fold_results_dir,
            seed=random_seed + fold,
            log_gradients=LOG_GRADIENTS,
            gradient_log_interval=GRADIENT_LOG_INTERVAL,
            diagnose_gradients_epochs=DIAGNOSE_GRADIENTS_EPOCHS,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            label_smoothing=label_smoothing
        )
        
        logger.info(f"Best validation AUC: {train_result['best_val_auc']:.4f}")
        
        # Plot training metrics for this fold
        logger.info(f"Generating training plots for fold {fold + 1}...")
        plot_training_metrics_separate(
            history=train_result['history'],
            save_dir=fold_results_dir,
            fold_name=f"Fold {fold + 1}"
        )
        
        # Load best model checkpoint
        if train_result['best_model_path'] is not None:
            checkpoint = torch.load(train_result['best_model_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {train_result['best_model_path']}")
        
        # Evaluate on test set (using the SAME scaler as training)
        logger.info(f"\n--- Evaluating on test set ---")
        test_results_dir = results_dir / f"fold_{fold}" / "test"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            save_predictions=True,
            output_dir=test_results_dir
        )
        
        all_test_results.append(test_metrics)
        all_test_metrics.append({
            'auc': test_metrics['auc'],
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
            'f1': test_metrics['f1'],
            'f1_weighted': test_metrics['f1_weighted'],
            'f1_macro': test_metrics['f1_macro']
        })
        
        logger.info(f"Test Results (Fold {fold + 1}):")
        logger.info(f"  ROC-AUC:     {test_metrics['auc']:.4f}")
        logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:      {test_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"  F1:          {test_metrics['f1']:.4f}")
        logger.info(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
        logger.info(f"  F1 Macro:    {test_metrics['f1_macro']:.4f}")
    
    # Compute mean and std across all test folds
    if len(all_test_metrics) == 0:
        raise RuntimeError("No test results were collected. Check if CV folds completed successfully.")
    
    # Build metrics array with all metrics
    metric_names = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'f1_weighted', 'f1_macro']
    metrics_array = np.array([[m[name] for name in metric_names] for m in all_test_metrics])
    
    final_results = {
        'mean': {name: float(np.mean(metrics_array[:, i])) for i, name in enumerate(metric_names)},
        'std': {name: float(np.std(metrics_array[:, i])) for i, name in enumerate(metric_names)},
        'all_folds': all_test_metrics
    }
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS (Mean ± Std across all test folds)")
    logger.info(f"{'='*80}")
    logger.info(f"ROC-AUC:     {final_results['mean']['auc']:.4f} ± {final_results['std']['auc']:.4f}")
    logger.info(f"Accuracy:    {final_results['mean']['accuracy']:.4f} ± {final_results['std']['accuracy']:.4f}")
    logger.info(f"Precision:   {final_results['mean']['precision']:.4f} ± {final_results['std']['precision']:.4f}")
    logger.info(f"Recall:      {final_results['mean']['recall']:.4f} ± {final_results['std']['recall']:.4f} (Sensitivity)")
    logger.info(f"Specificity: {final_results['mean']['specificity']:.4f} ± {final_results['std']['specificity']:.4f}")
    logger.info(f"F1 Score:    {final_results['mean']['f1']:.4f} ± {final_results['std']['f1']:.4f}")
    logger.info(f"F1 Weighted: {final_results['mean']['f1_weighted']:.4f} ± {final_results['std']['f1_weighted']:.4f}")
    logger.info(f"F1 Macro:    {final_results['mean']['f1_macro']:.4f} ± {final_results['std']['f1_macro']:.4f}")
    
    return final_results, all_test_results



def nested_cv_train_eval(
    subjects: List[Dict],
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    scaler_type: str = 'standard',
    aggregation_method: str = 'attention',
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 120,
    early_stopping_patience: int = 10,
    lstm_hidden_dim: int = 128,
    file_embedding_dim: int = 64,
    subject_embedding_dim: int = 64,
    dropout_lstm: float = 0.3,
    dropout_classifier: float = 0.5,
    use_weighted_bce: bool = True,
    device: torch.device = None,
    results_dir: Path = RESULTS_DIR,
    random_seed: int = 42,
    hp_search_grid: Optional[Dict] = None,
    downsample_factor: int = 1,
    use_simple_lstm: bool = True,
    use_feature_reduction: bool = False,
    feature_reduction_method: str = 'pca',
    n_features_reduced: int = 30,
    use_lr_scheduler: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 7,
    lr_scheduler_min_lr: float = 1e-7,
    label_smoothing: float = 0.0
) -> Tuple[Dict, List[Dict]]:
    """
    Perform nested cross-validation with hyperparameter tuning.
    
    Outer CV: splits subjects into train+val vs test
    Inner CV: splits train+val into inner-train and inner-val for hyperparameter tuning
    
    After selecting best hyperparameters:
    1. Fit final scaler on full train+val
    2. Retrain fresh model on full train+val with best hyperparameters
    3. Evaluate on test set
    
    Args:
        subjects: List of SubjectDict
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds
        scaler_type: 'standard', 'minmax', or 'robust'
        aggregation_method: 'attention', 'lstm', 'mean', 'max', 'mean_max'
        batch_size: Batch size
        learning_rate: Default learning rate (used if not in hp_search_grid)
        num_epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        lstm_hidden_dim: Default LSTM hidden dimension (used if not in hp_search_grid)
        file_embedding_dim: File embedding dimension
        subject_embedding_dim: Subject embedding dimension
        dropout_lstm: Default LSTM dropout rate (used if not in hp_search_grid)
        dropout_classifier: Classifier dropout rate
        use_weighted_bce: Use weighted BCE loss
        device: torch.device to use
        results_dir: Directory to save results
        random_seed: Random seed
        hp_search_grid: Dictionary of hyperparameters to tune, e.g.:
            {'learning_rate': [1e-5, 1e-4, 1e-3],
             'dropout_lstm': [0.2, 0.3, 0.5],
             'lstm_hidden_dim': [64, 128, 256]}
    
    Returns:
        Tuple of (final_results_dict, all_test_results_list)
    """
    # Ensure GPU is available - fail if not
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is required but not available. Please ensure GPU is available and PyTorch is installed with CUDA support.")
        device = torch.device('cuda')
    
    # Verify device is GPU
    if device.type != 'cuda':
        raise RuntimeError(f"GPU (CUDA) is required but device is set to {device.type}. Please ensure GPU is available.")
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is required but not available. Please ensure GPU is available and PyTorch is installed with CUDA support.")
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Detect number of features
    n_features = detect_n_features(subjects)
    
    # Prepare hyperparameter grid
    if hp_search_grid is None:
        hp_search_grid = {}
    
    # Create hyperparameter combinations
    hp_keys = list(hp_search_grid.keys())
    hp_values = list(hp_search_grid.values())
    hp_combinations = list(itertools.product(*hp_values))
    
    logger.info(f"Hyperparameter search grid: {hp_search_grid}")
    logger.info(f"Total hyperparameter combinations: {len(hp_combinations)}")
    
    # Prepare data for CV
    subject_ids = np.array([s['subject_id'] for s in subjects])
    labels = np.array([s['label'] for s in subjects])
    
    # Outer CV loop - use StratifiedGroupKFold for class-balanced splits
    outer_sgkf = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(len(subjects))
    
    all_test_results = []
    all_test_metrics = []
    
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_sgkf.split(indices, labels, subject_ids)):
        logger.info(f"\n{'='*80}")
        logger.info(f"OUTER FOLD {outer_fold + 1}/{n_outer_folds}")
        logger.info(f"{'='*80}")
        
        # Split into train+val and test
        train_val_subjects = [subjects[i] for i in train_val_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        test_labels = labels[test_idx]
        train_val_labels = labels[train_val_idx]
        
        logger.info(f"Train+Val: {len(train_val_subjects)} subjects ({np.sum(train_val_labels==0)} healthy, {np.sum(train_val_labels==1)} depressed)")
        logger.info(f"Test: {len(test_subjects)} subjects ({np.sum(test_labels==0)} healthy, {np.sum(test_labels==1)} depressed)")
        
        # Inner CV loop for hyperparameter optimization
        inner_sgkf = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_seed + outer_fold)
        inner_train_val_ids = np.array([s['subject_id'] for s in train_val_subjects])
        inner_train_val_labels = np.array([s['label'] for s in train_val_subjects])
        inner_train_val_indices = np.arange(len(train_val_subjects))
        
        best_hp_config = None
        best_inner_val_auc = 0.0
        best_inner_fold_results = []
        
        # Try each hyperparameter combination
        for hp_idx, hp_combo in enumerate(hp_combinations):
            hp_dict = dict(zip(hp_keys, hp_combo))
            logger.info(f"\n--- Hyperparameter Combination {hp_idx + 1}/{len(hp_combinations)}: {hp_dict} ---")
            
            # Use hyperparameters from grid, fallback to defaults
            current_lr = hp_dict.get('learning_rate', learning_rate)
            current_dropout_lstm = hp_dict.get('dropout_lstm', dropout_lstm)
            current_lstm_hidden = hp_dict.get('lstm_hidden_dim', lstm_hidden_dim)
            
            inner_val_aucs = []
            
            # Inner CV loop with class-balanced splits
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_sgkf.split(inner_train_val_indices, inner_train_val_labels, inner_train_val_ids)):
                logger.info(f"  Inner Fold {inner_fold + 1}/{n_inner_folds} (HP: {hp_dict})")
                
                # Split train_val into train and val
                inner_train_subjects = [train_val_subjects[i] for i in inner_train_idx]
                inner_val_subjects = [train_val_subjects[i] for i in inner_val_idx]
                
                # Create datasets (without scaling first)
                train_dataset = SubjectDataset(inner_train_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
                val_dataset = SubjectDataset(inner_val_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
                
                # Fit scaler on training data
                scaler = fit_scaler_on_dataset(train_dataset, scaler_type=scaler_type)
                
                # Fit feature reducer ONLY on training data (if enabled)
                feature_reducer = None
                if use_feature_reduction:
                    feature_reducer = fit_feature_reducer(train_dataset, 
                                                          method=feature_reduction_method,
                                                          n_components=n_features_reduced)
                
                # Apply scaler (and reducer) to train and val datasets
                train_dataset_scaled = ScaledDataset(train_dataset, scaler, feature_reducer)
                val_dataset_scaled = ScaledDataset(val_dataset, scaler, feature_reducer)
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset_scaled,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0
                )
                val_loader = DataLoader(
                    val_dataset_scaled,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0
                )
                
                # Determine effective number of features (after optional reduction)
                effective_n_features = n_features_reduced if use_feature_reduction else n_features
                
                # Create model with current hyperparameters
                model = HierarchicalLSTMDepression(
                    n_features=effective_n_features,
                    lstm_hidden=current_lstm_hidden,
                    file_embedding=file_embedding_dim,
                    subject_embedding=subject_embedding_dim,
                    dropout=current_dropout_lstm,
                    aggregation_method=aggregation_method,
                    classifier_hidden=CLASSIFIER_HIDDEN_DIM,
                    classifier_hidden2=CLASSIFIER_HIDDEN_DIM2,
                    classifier_dropout=dropout_classifier,
                    use_simple_lstm=use_simple_lstm
                ).to(device)
                
                # Train model
                fold_results_dir = results_dir / f"outer_{outer_fold}" / f"hp_{hp_idx}" / f"inner_{inner_fold}"
                fold_results_dir.mkdir(parents=True, exist_ok=True)
                
                train_result = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_epochs=num_epochs,
                    learning_rate=current_lr,
                    weight_decay=WEIGHT_DECAY,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
                    gradient_clip_norm=GRADIENT_CLIP_NORM,
                    use_mixed_precision=USE_MIXED_PRECISION,
                    use_weighted_bce=use_weighted_bce,
                    save_dir=fold_results_dir,
                    seed=random_seed + outer_fold * 1000 + hp_idx * 100 + inner_fold,
                    log_gradients=LOG_GRADIENTS,
                    gradient_log_interval=GRADIENT_LOG_INTERVAL,
                    diagnose_gradients_epochs=DIAGNOSE_GRADIENTS_EPOCHS,
                    use_lr_scheduler=use_lr_scheduler,
                    lr_scheduler_factor=lr_scheduler_factor,
                    lr_scheduler_patience=lr_scheduler_patience,
                    lr_scheduler_min_lr=lr_scheduler_min_lr,
                    label_smoothing=label_smoothing
                )
                
                inner_val_aucs.append(train_result['best_val_auc'])
                logger.info(f"    Inner fold {inner_fold + 1} val AUC: {train_result['best_val_auc']:.4f}")
            
            # Average validation AUC across inner folds for this hyperparameter combination
            mean_inner_val_auc = np.mean(inner_val_aucs)
            logger.info(f"  Mean inner val AUC for {hp_dict}: {mean_inner_val_auc:.4f} ± {np.std(inner_val_aucs):.4f}")
            
            # Track best hyperparameter configuration
            if mean_inner_val_auc > best_inner_val_auc:
                best_inner_val_auc = mean_inner_val_auc
                best_hp_config = hp_dict.copy()
                best_inner_fold_results = inner_val_aucs.copy()
                logger.info(f"  *** New best hyperparameters: {best_hp_config} (AUC: {best_inner_val_auc:.4f}) ***")
        
        # After inner CV, retrain on full train+val with best hyperparameters
        if best_hp_config is None:
            # No hyperparameter grid provided, use defaults
            best_hp_config = {}
            logger.info("No hyperparameter grid provided, using default hyperparameters")
        
        logger.info(f"\n--- Retraining on full train+val with best hyperparameters: {best_hp_config} ---")
        
        # Extract best hyperparameters
        final_lr = best_hp_config.get('learning_rate', learning_rate)
        final_dropout_lstm = best_hp_config.get('dropout_lstm', dropout_lstm)
        final_lstm_hidden = best_hp_config.get('lstm_hidden_dim', lstm_hidden_dim)
        
        # Prepare full train+val dataset
        train_val_dataset = SubjectDataset(train_val_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        test_dataset = SubjectDataset(test_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        
        # Fit final scaler on full train+val
        logger.info(f"Fitting {scaler_type} scaler on full train+val data...")
        final_scaler = fit_scaler_on_dataset(train_val_dataset, scaler_type=scaler_type)
        
        # Fit feature reducer ONLY on train+val data (if enabled)
        final_feature_reducer = None
        if use_feature_reduction:
            logger.info(f"Fitting {feature_reduction_method} feature reducer to {n_features_reduced} features...")
            final_feature_reducer = fit_feature_reducer(train_val_dataset, 
                                                         method=feature_reduction_method,
                                                         n_components=n_features_reduced)
        
        # Apply scaler (and reducer) to train+val and test
        train_val_dataset_scaled = ScaledDataset(train_val_dataset, final_scaler, final_feature_reducer)
        test_dataset_scaled = ScaledDataset(test_dataset, final_scaler, final_feature_reducer)
        
        # Split train+val into train and val for early stopping (80/20 split)
        # Use StratifiedGroupKFold to maintain class balance
        train_val_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed + outer_fold)
        train_val_subject_ids = np.array([s['subject_id'] for s in train_val_subjects])
        train_val_subject_labels = np.array([s['label'] for s in train_val_subjects])
        train_val_indices = np.arange(len(train_val_subjects))
        
        final_train_idx, final_val_idx = next(train_val_sgkf.split(train_val_indices, train_val_subject_labels, train_val_subject_ids))
        
        # Create indices for train and val in the scaled dataset
        final_train_subjects = [train_val_subjects[i] for i in final_train_idx]
        final_val_subjects = [train_val_subjects[i] for i in final_val_idx]
        
        final_train_dataset = SubjectDataset(final_train_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        final_val_dataset = SubjectDataset(final_val_subjects, normalize=False, handle_nan_inf=True, downsample_factor=downsample_factor)
        
        final_train_dataset_scaled = ScaledDataset(final_train_dataset, final_scaler, final_feature_reducer)
        final_val_dataset_scaled = ScaledDataset(final_val_dataset, final_scaler, final_feature_reducer)
        
        # Create data loaders
        final_train_loader = DataLoader(
            final_train_dataset_scaled,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        final_val_loader = DataLoader(
            final_val_dataset_scaled,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset_scaled,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Determine effective number of features for final model
        final_effective_n_features = n_features_reduced if use_feature_reduction else n_features
        
        # Create fresh model with best hyperparameters
        final_model = HierarchicalLSTMDepression(
            n_features=final_effective_n_features,
            lstm_hidden=final_lstm_hidden,
            file_embedding=file_embedding_dim,
            subject_embedding=subject_embedding_dim,
            dropout=final_dropout_lstm,
            aggregation_method=aggregation_method,
            classifier_hidden=CLASSIFIER_HIDDEN_DIM,
            classifier_hidden2=CLASSIFIER_HIDDEN_DIM2,
            classifier_dropout=dropout_classifier,
            use_simple_lstm=use_simple_lstm
        ).to(device)
        
        # Retrain on full train+val
        retrain_results_dir = results_dir / f"outer_{outer_fold}" / "retrain"
        retrain_results_dir.mkdir(parents=True, exist_ok=True)
        
        retrain_result = train_model(
            model=final_model,
            train_loader=final_train_loader,
            val_loader=final_val_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=final_lr,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            use_mixed_precision=USE_MIXED_PRECISION,
            use_weighted_bce=use_weighted_bce,
            save_dir=retrain_results_dir,
            seed=random_seed + outer_fold,
            log_gradients=LOG_GRADIENTS,
            gradient_log_interval=GRADIENT_LOG_INTERVAL,
            diagnose_gradients_epochs=DIAGNOSE_GRADIENTS_EPOCHS,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            label_smoothing=label_smoothing
        )
        
        logger.info(f"Retrained model validation AUC: {retrain_result['best_val_auc']:.4f}")
        
        # Load best model checkpoint
        if retrain_result['best_model_path'] is not None:
            checkpoint = torch.load(retrain_result['best_model_path'], map_location=device)
            final_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        logger.info(f"\n--- Evaluating on test set ---")
        test_results_dir = results_dir / f"outer_{outer_fold}" / "test"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best hyperparameters
        with open(test_results_dir / 'best_hyperparameters.json', 'w') as f:
            json.dump(best_hp_config, f, indent=2)
        
        test_metrics = evaluate_model(
            model=final_model,
            dataloader=test_loader,
            device=device,
            save_predictions=True,
            output_dir=test_results_dir
        )
        
        all_test_results.append(test_metrics)
        all_test_metrics.append({
            'auc': test_metrics['auc'],
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
            'f1': test_metrics['f1'],
            'f1_weighted': test_metrics['f1_weighted'],
            'f1_macro': test_metrics['f1_macro'],
            'best_hyperparameters': best_hp_config
        })
        
        logger.info(f"Test Results (Outer Fold {outer_fold + 1}):")
        logger.info(f"  Best Hyperparameters: {best_hp_config}")
        logger.info(f"  ROC-AUC:     {test_metrics['auc']:.4f}")
        logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:      {test_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"  F1:          {test_metrics['f1']:.4f}")
        logger.info(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
        logger.info(f"  F1 Macro:    {test_metrics['f1_macro']:.4f}")
    
    # Compute mean and std across all test folds
    if len(all_test_metrics) == 0:
        raise RuntimeError("No test results were collected. Check if CV folds completed successfully.")
    
    # Build metrics array with all metrics
    metric_names = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'f1_weighted', 'f1_macro']
    metrics_array = np.array([[m[name] for name in metric_names] for m in all_test_metrics])
    
    final_results = {
        'mean': {name: float(np.mean(metrics_array[:, i])) for i, name in enumerate(metric_names)},
        'std': {name: float(np.std(metrics_array[:, i])) for i, name in enumerate(metric_names)},
        'all_folds': all_test_metrics
    }
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS (Mean ± Std across all test folds)")
    logger.info(f"{'='*80}")
    logger.info(f"ROC-AUC:     {final_results['mean']['auc']:.4f} ± {final_results['std']['auc']:.4f}")
    logger.info(f"Accuracy:    {final_results['mean']['accuracy']:.4f} ± {final_results['std']['accuracy']:.4f}")
    logger.info(f"Precision:   {final_results['mean']['precision']:.4f} ± {final_results['std']['precision']:.4f}")
    logger.info(f"Recall:      {final_results['mean']['recall']:.4f} ± {final_results['std']['recall']:.4f} (Sensitivity)")
    logger.info(f"Specificity: {final_results['mean']['specificity']:.4f} ± {final_results['std']['specificity']:.4f}")
    logger.info(f"F1 Score:    {final_results['mean']['f1']:.4f} ± {final_results['std']['f1']:.4f}")
    logger.info(f"F1 Weighted: {final_results['mean']['f1_weighted']:.4f} ± {final_results['std']['f1_weighted']:.4f}")
    logger.info(f"F1 Macro:    {final_results['mean']['f1_macro']:.4f} ± {final_results['std']['f1_macro']:.4f}")
    
    return final_results, all_test_results


def plot_confusion_matrix_combined(all_test_results: List[Dict], save_path: Path):
    """Plot combined confusion matrix from all test folds."""
    if len(all_test_results) == 0:
        raise ValueError("Cannot plot confusion matrix: no test results provided")
    
    # Aggregate all predictions
    all_y_true = []
    all_y_pred = []
    
    for result in all_test_results:
        if 'y_true' not in result or 'y_pred' not in result:
            raise ValueError("Test results must contain 'y_true' and 'y_pred' keys")
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
    
    all_y_true = np.array(all_y_true)
    all_y_pred_binary = (np.array(all_y_pred) > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(all_y_true, all_y_pred_binary)
    
    # Create plot
    cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy', 'Depressed'],
                    yticklabels=['Healthy', 'Depressed'])
    else:
        plt.imshow(cm_array, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Healthy', 'Depressed'])
        plt.yticks(tick_marks, ['Healthy', 'Depressed'])
        thresh = cm_array.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm_array[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm_array[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Combined Confusion Matrix (All Test Folds)')
    
    # Add text annotations
    total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
    accuracy = (cm['tn'] + cm['tp']) / total if total > 0 else 0
    
    plt.text(0.5, -0.15, f'Total: {total} | Accuracy: {accuracy:.4f}',
             ha='center', transform=plt.gca().transAxes)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved combined confusion matrix to {save_path}")
    plt.close()


def plot_roc_combined(all_test_results: List[Dict], save_path: Path):
    """Plot combined ROC curve from all test folds."""
    
    
    if len(all_test_results) == 0:
        raise ValueError("Cannot plot ROC curve: no test results provided")
    
    # Aggregate all predictions
    all_y_true = []
    all_y_pred = []
    
    for result in all_test_results:
        if 'y_true' not in result or 'y_pred' not in result:
            raise ValueError("Test results must contain 'y_true' and 'y_pred' keys")
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve (All Test Folds)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved combined ROC curve to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LSTM Depression Detection with Cross-Validation')
    
    # CV type selection
    parser.add_argument('--cv_type', type=str, default='simple',
                       choices=['simple', 'nested'],
                       help='Type of cross-validation: simple (single-level) or nested (with hyperparameter tuning)')
    
    # Data arguments
    parser.add_argument('--condition', type=str, default='All',
                       choices=['CR', 'CRADK', 'ADK', 'SHAM', 'All'],
                       help='Condition to use')
    parser.add_argument('--phase', type=str, default='all',
                       help='Phase to use (e.g., training_pos, induction1, all)')
    parser.add_argument('--processed_audio_dir', type=str, default=PROCESSED_AUDIO_DIR,
                       help='Path to processed audio directory')
    parser.add_argument('--metadata_csv', type=str, default=MERGED_METADATA_CSV,
                       help='Path to metadata CSV')
    
    # Preprocessing arguments
    parser.add_argument('--scaler', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust'],
                       help='Scaler type: standard, minmax, or robust')
    parser.add_argument('--downsample_factor', type=int, default=DOWNSAMPLE_FACTOR,
                       help='Downsampling factor (keep every Nth frame). Set to 1 to disable, 30 for 1/30 downsampling (1 FPS from 30 FPS)')
    
    # Model arguments
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['attention', 'lstm', 'mean', 'max', 'mean_max'],
                       help='Aggregation method (mean recommended for small datasets)')
    parser.add_argument('--lstm_hidden_dim', type=int, default=LSTM_HIDDEN_DIM,
                       help='LSTM hidden dimension (default, can be tuned in nested CV)')
    parser.add_argument('--file_embedding_dim', type=int, default=FILE_EMBEDDING_DIM,
                       help='File embedding dimension')
    parser.add_argument('--subject_embedding_dim', type=int, default=SUBJECT_EMBEDDING_DIM,
                       help='Subject embedding dimension')
    parser.add_argument('--dropout_lstm', type=float, default=DROPOUT_LSTM,
                       help='LSTM dropout rate (default, can be tuned in nested CV)')
    parser.add_argument('--dropout_classifier', type=float, default=DROPOUT_CLASSIFIER,
                       help='Classifier dropout rate')
    parser.add_argument('--use_simple_lstm', action='store_true', default=USE_SIMPLE_LSTM,
                       help='Use simplified single-layer LSTM (recommended for small datasets)')
    parser.add_argument('--no_simple_lstm', dest='use_simple_lstm', action='store_false',
                       help='Use full 2-layer bidirectional LSTM')
    
    # Feature reduction arguments
    parser.add_argument('--use_feature_reduction', action='store_true', default=USE_FEATURE_REDUCTION,
                       help='Enable feature reduction (PCA/SelectKBest)')
    parser.add_argument('--no_feature_reduction', dest='use_feature_reduction', action='store_false',
                       help='Disable feature reduction')
    parser.add_argument('--feature_reduction_method', type=str, default=FEATURE_REDUCTION_METHOD,
                       choices=['pca', 'select_k_best'],
                       help='Feature reduction method')
    parser.add_argument('--n_features_reduced', type=int, default=N_FEATURES_REDUCED,
                       help='Number of features after reduction')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate (default, can be tuned in nested CV)')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Maximum number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--use_weighted_bce', action='store_true', default=USE_WEIGHTED_BCE,
                       help='Use weighted BCE loss')
    parser.add_argument('--no_weighted_bce', dest='use_weighted_bce', action='store_false',
                       help='Disable weighted BCE loss')
    
    # Learning rate scheduler arguments
    parser.add_argument('--use_lr_scheduler', action='store_true', default=USE_LR_SCHEDULER,
                       help='Use learning rate scheduler')
    parser.add_argument('--no_lr_scheduler', dest='use_lr_scheduler', action='store_false',
                       help='Disable learning rate scheduler')
    parser.add_argument('--lr_scheduler_factor', type=float, default=LR_SCHEDULER_FACTOR,
                       help='LR scheduler reduction factor')
    parser.add_argument('--lr_scheduler_patience', type=int, default=LR_SCHEDULER_PATIENCE,
                       help='LR scheduler patience')
    parser.add_argument('--lr_scheduler_min_lr', type=float, default=LR_SCHEDULER_MIN_LR,
                       help='Minimum learning rate')
    
    # Label smoothing
    parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING,
                       help='Label smoothing factor (0 = no smoothing)')
    
    # CV arguments
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds (for simple CV)')
    parser.add_argument('--n_outer_folds', type=int, default=5,
                       help='Number of outer CV folds (for nested CV)')
    parser.add_argument('--n_inner_folds', type=int, default=3,
                       help='Number of inner CV folds (for nested CV)')
    
    # Hyperparameter tuning arguments (for nested CV)
    parser.add_argument('--tune_learning_rate', action='store_true',
                       help='Tune learning rate in nested CV (uses HP_SEARCH_GRID from config)')
    parser.add_argument('--tune_dropout_lstm', action='store_true',
                       help='Tune LSTM dropout in nested CV (uses HP_SEARCH_GRID from config)')
    parser.add_argument('--tune_lstm_hidden_dim', action='store_true',
                       help='Tune LSTM hidden dimension in nested CV (uses HP_SEARCH_GRID from config)')
    parser.add_argument('--tune_all', action='store_true',
                       help='Tune all hyperparameters in nested CV (uses HP_SEARCH_GRID from config)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda'],
                       help='Device to use (GPU/CUDA only - CPU is not supported)')
    parser.add_argument('--results_dir', type=str, default=str(RESULTS_DIR),
                       help='Results directory')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(results_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    cv_type_name = "Simple Cross-Validation" if args.cv_type == 'simple' else "Nested Cross-Validation"
    logger.info("="*80)
    logger.info(f"LSTM Depression Detection - {cv_type_name}")
    logger.info("="*80)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Load data
    logger.info("\nLoading data...")
    subjects = load_subjects_from_processed(
        condition=args.condition,
        phase=args.phase,
        processed_audio_dir=args.processed_audio_dir,
        metadata_csv=args.metadata_csv
    )
    
    if len(subjects) == 0:
        raise ValueError("No subjects loaded. Check condition and phase arguments.")
    
    logger.info(f"Loaded {len(subjects)} subjects")
    
    # Convert device string to torch.device - GPU ONLY, fail if not available
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU (CUDA) is required but not available.\n"
                "Please check:\n"
                "  1. GPU is available: nvidia-smi\n"
                "  2. PyTorch is installed with CUDA support: torch.cuda.is_available()\n"
                "  3. CUDA drivers are properly installed\n"
                "The code will NOT run on CPU. Exiting."
            )
        device = torch.device('cuda')
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    elif args.device == 'cpu':
        raise RuntimeError("CPU device is not allowed. This code requires GPU (CUDA). Please use --device cuda")
    else:
        raise ValueError(f"Unknown device: {args.device}. Only 'cuda' is supported (GPU required).")
    
    # Build hyperparameter search grid for nested CV
    hp_search_grid = None
    if args.cv_type == 'nested':
        hp_search_grid = {}
        if args.tune_all or args.tune_learning_rate:
            hp_search_grid['learning_rate'] = HP_SEARCH_GRID.get('learning_rate', [1e-5, 1e-4, 1e-3])
        if args.tune_all or args.tune_dropout_lstm:
            hp_search_grid['dropout_lstm'] = HP_SEARCH_GRID.get('dropout_lstm', [0.2, 0.3, 0.5])
        if args.tune_all or args.tune_lstm_hidden_dim:
            hp_search_grid['lstm_hidden_dim'] = HP_SEARCH_GRID.get('lstm_hidden_dim', [64, 128, 256])
        
        if len(hp_search_grid) == 0:
            logger.info("No hyperparameter tuning requested for nested CV, using default hyperparameters")
            hp_search_grid = None
        else:
            logger.info(f"Hyperparameter search grid: {hp_search_grid}")
    
    # Run appropriate CV based on cv_type
    if args.cv_type == 'simple':
        # Run simple CV
        final_results, all_test_results = cv_train_eval(
            subjects=subjects,
            n_folds=args.n_folds,
            scaler_type=args.scaler,
            aggregation_method=args.aggregation,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience,
            lstm_hidden_dim=args.lstm_hidden_dim,
            file_embedding_dim=args.file_embedding_dim,
            subject_embedding_dim=args.subject_embedding_dim,
            dropout_lstm=args.dropout_lstm,
            dropout_classifier=args.dropout_classifier,
            use_weighted_bce=args.use_weighted_bce,
            device=device,
            results_dir=results_dir,
            random_seed=args.random_seed,
            downsample_factor=args.downsample_factor,
            use_simple_lstm=args.use_simple_lstm,
            use_feature_reduction=args.use_feature_reduction,
            feature_reduction_method=args.feature_reduction_method,
            n_features_reduced=args.n_features_reduced,
            use_lr_scheduler=args.use_lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_min_lr=args.lr_scheduler_min_lr,
            label_smoothing=args.label_smoothing
        )
    else:  # nested CV
        # Run nested CV
        final_results, all_test_results = nested_cv_train_eval(
            subjects=subjects,
            n_outer_folds=args.n_outer_folds,
            n_inner_folds=args.n_inner_folds,
            scaler_type=args.scaler,
            aggregation_method=args.aggregation,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience,
            lstm_hidden_dim=args.lstm_hidden_dim,
            file_embedding_dim=args.file_embedding_dim,
            subject_embedding_dim=args.subject_embedding_dim,
            dropout_lstm=args.dropout_lstm,
            dropout_classifier=args.dropout_classifier,
            use_weighted_bce=args.use_weighted_bce,
            device=device,
            results_dir=results_dir,
            random_seed=args.random_seed,
            hp_search_grid=hp_search_grid,
            downsample_factor=args.downsample_factor,
            use_simple_lstm=args.use_simple_lstm,
            use_feature_reduction=args.use_feature_reduction,
            feature_reduction_method=args.feature_reduction_method,
            n_features_reduced=args.n_features_reduced,
            use_lr_scheduler=args.use_lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_min_lr=args.lr_scheduler_min_lr,
            label_smoothing=args.label_smoothing
        )
    
    # Save final results
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    # Combined confusion matrix
    plot_confusion_matrix_combined(
        all_test_results,
        save_path=results_dir / 'confusion_matrix_combined.png'
    )
    
    # Combined ROC curve
    plot_roc_combined(
        all_test_results,
        save_path=results_dir / 'roc_curve_combined.png'
    )
    
    logger.info(f"\nAll results saved to: {results_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()

