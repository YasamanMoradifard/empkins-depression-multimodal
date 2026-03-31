"""
Metrics computation utilities.

Computes classification metrics for depression detection.
Includes: Accuracy, Precision, Recall (Sensitivity), Specificity,
          F1, F1 Weighted, F1 Macro, ROC-AUC
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
    
    Returns:
        Dict with metrics:
            - auc (ROC-AUC)
            - accuracy
            - precision
            - recall (sensitivity)
            - specificity
            - f1 (F1 Score)
            - f1_weighted (F1 Weighted by class support)
            - f1_macro (F1 Macro - unweighted mean)
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Check if all predictions are identical (model collapse)
    if len(np.unique(y_pred)) == 1:
        # All predictions are the same - AUC is undefined, return 0.5 (random)
        auc = 0.5
    elif len(np.unique(y_true)) == 1:
        # Only one class in true labels
        auc = 0.0
    else:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle edge cases where AUC can't be computed
            auc = 0.5
    
    # Compute confusion matrix for specificity
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'auc': auc,
        'accuracy': accuracy_score(y_true, y_pred_binary),
        # Precision = TP / (TP + FP)
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        # Recall (Sensitivity) = TP / (TP + FN)
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        # Specificity = TN / (TN + FP)
        'specificity': specificity,
        # F1 Score (default for positive class)
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        # F1 Weighted - weighted by class support
        'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        # F1 Macro - unweighted mean of F1 per class
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    }
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred_binary: Predicted binary labels
    
    Returns:
        Dict with tn, fp, fn, tp
    """
    # Always specify labels=[0, 1] to ensure we get a 2x2 matrix
    # This handles edge cases where only one class is predicted or present
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    
    # cm will always be 2x2 when labels are specified
    if cm.shape == (2, 2):
        # Confusion matrix layout:
        #           Predicted
        #           0    1
        # True  0  TN   FP
        #       1  FN   TP
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback (should not happen with labels specified)
        tn, fp, fn, tp = 0, 0, 0, 0
    
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

