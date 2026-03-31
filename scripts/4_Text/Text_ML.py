#!/usr/bin/env python3
"""
BioSig (EMG/ECG/RSP) ML with nested cross-validation for depression detection (leakage-free).

Pipeline structure:
1. Load and aggregate BioSig data via load_biosig_data (uses BioSigAggregation inside).
2. Nested CV:
   - Outer fold (StratifiedGroupKFold): On each outer split:
     - Preprocessing fit on outer TRAIN only: build_preprocess_pipeline (Constant → NaN → Imputer → Scaler)
       + Mann-Whitney (top K features). Transform outer train and outer test.
     - Inner fold (GroupKFold) on outer train: build_inner_pipeline (RFE + classifier), GridSearchCV for RFE n + HPO.
     - Best model refit on full outer train, evaluated on outer test. RFE selected features saved per fold via extract_selected_features.
3. Results: per-outer-fold test metrics, dashboards, selected features per fold, HPO params.

No correlation filter. Only build_preprocess_pipeline, build_inner_pipeline, extract_selected_features, load_biosig_data, BioSigAggregation.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.fft import rfft, rfftfreq
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy.stats import mannwhitneyu, skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42
POS_LABEL = 1   # Depressed
NEG_LABEL = 0   # Healthy

META_COLS = ["ID", "label", "condition", "phase", "Aufgabe"]

# =============================================================================
# Phase name standardization mapping
# =============================================================================
PHASE_NAME_MAPPING = {
    # Audio phases
    "training_pos": "training_pos",
    "training_neg": "training_neg",
    "induction1": "induction1",
    "induction2": "induction2",
    # Video phases (same as Audio)
    "latency": "latency",
    # BioSig phases
    "training": "training_pos",
    "coping": "training_neg",
    "emotion_induction_1": "induction1",
    "emotion_induction_2": "induction2",
}

STANDARD_PHASES = ["training_pos", "training_neg", "induction1", "induction2", "latency"]


def standardize_phase_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize phase names across all modalities."""
    if "phase" in df.columns:
        df = df.copy()
        df["phase"] = df["phase"].map(lambda x: PHASE_NAME_MAPPING.get(x, x))
    return df


# =============================================================================
# Data Loading Functions for Each Modality
# =============================================================================

def normalize_id(pid):
    """Convert participant IDs to a standardized string format."""
    if pd.isna(pid) or (isinstance(pid, float) and math.isnan(pid)):
        return np.nan
    try:
        pid_int = int(float(pid))
        if pid_int < 100:
            return str(pid_int).zfill(3)
        else:
            return str(pid_int)
    except ValueError:
        return np.nan


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names for XGBoost compatibility.
    Handles MultiIndex columns (e.g. from groupby.agg) by flattening to strings.
    """
    def sanitize_name(name: str) -> str:
        name = str(name)
        name = name.replace('[', '_').replace(']', '_').replace('<', '_')
        name = name.replace('>', '_').replace('|', '_').replace('&', '_')
        return name

    def flatten_col(col) -> str:
        """Convert column (possibly tuple from MultiIndex) to a flat string."""
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p]
            return '_'.join(parts) if parts else sanitize_name(str(col))
        return str(col)

    df = df.copy()
    df.columns = [sanitize_name(flatten_col(col)) for col in df.columns]
    return df


def map_labels_to_binary(y: pd.Series) -> pd.Series:
    """Standardize labels to {0,1}, with Depressed=1, Healthy=0."""
    mapping = {
        'Depressed': 1, 'depressed': 1, 'DEPRESSED': 1, 1: 1, True: 1,
        'Healthy': 0, 'healthy': 0, 'HEALTHY': 0, 0: 0, False: 0
    }
    return y.map(lambda v: mapping.get(v, v)).astype(int)


def load_text_data(text_csv_path: Path, condition: str = "all") -> pd.DataFrame:
    """
    Load Text modality features from a pre-computed CSV.
    
    Expected format:
        - Column 'patient_id' (participant ID, not zero-padded)
        - Column 'Condition' (treatment condition: CR, ADK, CRADK, SHAM, ...)
        - Column 'Diagnose' (diagnosis label, e.g. 'Depressed' / 'Healthy')
        - 14 feature columns (all numeric)
    
    Processing:
        - Zero-pad 'patient_id' to 3 digits and rename to 'ID'
        - Optionally filter rows by 'Condition' (if condition != 'all')
        - Map 'Diagnose' to binary 'label' using map_labels_to_binary
        - No aggregation by phase (one row per ID)
    """
    if not text_csv_path.exists():
        raise FileNotFoundError(f"Text data CSV not found: {text_csv_path}")
    
    df = pd.read_csv(text_csv_path)
    # Basic column checks
    required_cols = ["patient_id", "condition", "Diagnose"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Text CSV must contain columns {required_cols}, missing: {missing}")
 
    # Zero-pad patient_id and rename to ID
    df = df.copy()
    df["ID"] = df["patient_id"].apply(normalize_id).astype(str)
    df = df.drop(columns=["patient_id"])
    
    # Normalize / filter by condition
    df["condition"] = df["condition"].astype(str).str.strip()
    if condition.lower() != "all":
        df = df[df["condition"] == condition].copy()
        if df.empty:
            raise ValueError(f"No text rows found for condition '{condition}' in {text_csv_path}")
    
    # Map Diagnose -> binary label
    df["label"] = map_labels_to_binary(df["Diagnose"])
    
    # Treat all remaining non-ID / non-label / non-condition columns as numeric features
    feature_cols = [
        c for c in df.columns
        if c not in ["ID", "label", "condition", "Diagnose"]
    ]
    if len(feature_cols) != 14:
        print(f"[WARN] Text data: expected 14 feature columns, found {len(feature_cols)}.")
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=["Diagnose"])

    # Add modality prefix to feature columns
    rename_map = {c: f"Text_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    print(f"[INFO] Text data: {df.shape[0]} rows, {len(feature_cols)} feature columns")
    print(f"[INFO] Unique participants (text): {df['ID'].nunique()}")
    if condition.lower() != "all":
        print(f"[INFO] Condition filter applied to text modality: {condition}")
    print("-----------------------------------------\n")
    return df


# =============================================================================
# Classifier Models
# =============================================================================

def get_classifier_models() -> Dict[str, object]:
    """
    Return classifiers 
    """
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=100000,
            class_weight="balanced",
            solver="saga",
            random_state=RANDOM_STATE,
        ),
        "SVC_RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=100000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=RANDOM_STATE,
        )
    
    return models


def get_param_grids_clf_only() -> Dict[str, Dict]:
    """Classifier-only param grids (no RFE n). RFE n is chosen by a for loop over 1..n_features."""
    grids = {
        "LogisticRegression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "SVC_RBF": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto", 0.01, 0.1]},
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None]},
        "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.1, 0.5, 1.0]},
        "DecisionTree": {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 2, 4]},
        "KNN": {"clf__n_neighbors": [3, 5, 7, 9]},
    }
    if HAS_XGB:
        grids["XGBoost"] = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1],
        }
    return grids


def build_pipeline(clf, rfe_n: int) -> Pipeline:
    """Early fusion inner: RFE(n) on concatenated features -> classifier. MW per modality done separately.
    RFE always uses DecisionTreeClassifier for consistent, fast feature selection across all final classifiers."""
    rfe_estimator = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    return Pipeline([
        ("clf", clf),
    ])


def extract_selected_features(pipe: Pipeline, feature_names: List[str]) -> List[str]:
    """Extract selected feature names from early fusion pipeline (RFE only)."""
    if "rfe_selector" not in pipe.named_steps:
        return list(feature_names)
    rfe = pipe.named_steps["rfe_selector"]
    if hasattr(rfe, "support_"):
        return [f for f, s in zip(feature_names, rfe.support_) if s]
    return list(feature_names)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive metrics including accuracy, precision, recall, specificity, F1, and confusion matrix."""

    y_pred = model.predict(X_test)

    # Calculate all metrics
    metrics: Dict[str, float] = {
        # Accuracy
        "accuracy": float(accuracy_score(y_test, y_pred)),
        # Precision
        "precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
        # Recall (Sensitivity)
        "recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        "sensitivity": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        # F1 metrics
        "f1": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
    }

    # Calculate ROC-AUC if model supports probability prediction
    y_pred_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        else:
            metrics["roc_auc"] = float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["tn"] = int(tn)
            metrics["fp"] = int(fp)
            metrics["fn"] = int(fn)
            metrics["tp"] = int(tp)
            metrics["cm"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
            
            # Calculate specificity: TN / (TN + FP)
            if (tn + fp) > 0:
                metrics["specificity"] = float(tn / (tn + fp))
            else:
                metrics["specificity"] = float("nan")
        else:
            metrics["tn"] = 0
            metrics["fp"] = 0
            metrics["fn"] = 0
            metrics["tp"] = 0
            metrics["specificity"] = float("nan")
            metrics["cm"] = [[0, 0], [0, 0]]
    except Exception:
        metrics["tn"] = 0
        metrics["fp"] = 0
        metrics["fn"] = 0
        metrics["tp"] = 0
        metrics["specificity"] = float("nan")
        metrics["cm"] = [[0, 0], [0, 0]]

    # Store predictions and true labels
    metrics["y_true"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_pred_proba"] = y_pred_proba.tolist() if y_pred_proba is not None else None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    normalization: str,
    save_path: Path,
) -> None:
    """Plot and save confusion matrix for a model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Healthy", "Depressed"],
            yticklabels=["Healthy", "Depressed"],
            cbar_kws={"label": "Count"},
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(label="Count")
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Healthy", "Depressed"])
        plt.yticks(tick_marks, ["Healthy", "Depressed"])
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix\n{model_name} ({normalization})")
    
    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    plt.text(
        0.5,
        -0.15,
        f"Total: {total} | Accuracy: {accuracy:.4f}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=10,
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")


def plot_roc_curves(
    roc_data: Dict[str, Tuple[List[float], List[float]]],
    normalization: str,
    save_path: Path,
) -> None:
    """
    Plot ROC curves for all models on the same figure.
    
    Args:
        roc_data: Dictionary mapping model_name -> (y_true_list, y_pred_proba_list)
        normalization: Name of the normalization method used
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Define colors for different models
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    model_names = list(roc_data.keys())
    
    for idx, model_name in enumerate(model_names):
        y_true_list, y_proba_list = roc_data[model_name]
        
        if not y_true_list or not y_proba_list:
            continue
        
        # Convert to numpy arrays
        y_true = np.array(y_true_list)
        y_proba = np.array(y_proba_list)
        
        # Skip if no valid probabilities
        if len(y_true) == 0 or len(y_proba) == 0:
            continue
        
        # Compute ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            color = colors[idx % len(colors)]
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        except Exception as e:
            print(f"[WARN] Could not compute ROC curve for {model_name}: {e}")
            continue
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - All Models\n(Normalization: {normalization})', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC curves to {save_path}")


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    norm_method: str,
    out_dir: Path,
    base_name: str,
    top_n: int = 20,
) -> None:
    """Plot feature importance for models that support it."""
    
    # Extract the classifier from pipeline
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['clf']
    else:
        clf = model
    
    # Get feature importance and determine the type
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        # Determine importance type based on model
        if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)):
            importance_type = "Gini Importance"
        elif isinstance(clf, AdaBoostClassifier):
            importance_type = "Weighted Error Reduction"
        elif HAS_XGB and isinstance(clf, XGBClassifier):
            importance_type = "Gain-based Importance"
        else:
            importance_type = "Feature Importance Score"
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
        # For linear models, importance is based on coefficients
        if isinstance(clf, LogisticRegression):
            importance_type = "|Coefficient| (Log-Odds)"
        elif isinstance(clf, SVC):
            importance_type = "|Coefficient| (SVM Weight)"
        else:
            importance_type = "|Coefficient|"
    else:
        # Model doesn't support feature importance
        return
    
    # Get feature names after all transformations (support rfe_selector or feature_selector)
    try:
        selector = None
        if hasattr(model, "named_steps"):
            if "rfe_selector" in model.named_steps:
                selector = model.named_steps["rfe_selector"]
            elif "feature_selector" in model.named_steps:
                selector = model.named_steps["feature_selector"]
        if selector is not None and hasattr(selector, "selected_features_") and selector.selected_features_ is not None:
            selected_indices = selector.selected_features_
            selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]
        else:
            selected_features = feature_names
        
        # Make sure we have matching lengths
        if len(importances) != len(selected_features):
            # Try to match by using the smaller length
            min_len = min(len(importances), len(selected_features))
            importances = importances[:min_len]
            selected_features = selected_features[:min_len]
        
        if len(importances) == 0:
            return
            
        # Sort by importance and get top N
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'Top {top_n} Feature Importances\n{model_name} ({norm_method})')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        save_path = out_dir / f"{base_name}_{norm_method}_{model_name}_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved feature importance to {save_path}")
        
    except Exception as e:
        print(f"[WARN] Could not plot feature importance for {model_name}: {e}")


def plot_model_comparison_heatmap(
    summary_df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
) -> None:
    """Create heatmap comparing all models across metrics."""
    
    try:
        # Metrics to include in heatmap
        metrics = ['accuracy_mean', 'f1_weighted_mean', 'precision_mean', 
                   'recall_mean', 'specificity_mean', 'roc_auc_mean']
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in summary_df.columns]
        
        if not available_metrics:
            print("[WARN] No metrics available for heatmap")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric in summary_df.columns:
                # For single normalization, create a simple bar chart instead of heatmap
                model_values = summary_df.set_index('model')[metric].sort_values(ascending=False)
                
                # Create bar chart
                bars = ax.barh(range(len(model_values)), model_values.values, 
                              color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_values))))
                ax.set_yticks(range(len(model_values)))
                ax.set_yticklabels(model_values.index)
                ax.set_xlabel(metric.replace('_mean', '').replace('_', ' ').title())
                ax.set_title(metric.replace('_mean', '').replace('_', ' ').title())
                ax.set_xlim(0, 1)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (idx, val) in enumerate(model_values.items()):
                    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
            else:
                ax.set_visible(False)
        
        plt.suptitle('Model Comparison Across Metrics', fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = out_dir / f"{base_name}_model_comparison_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved model comparison heatmap to {save_path}")
        
    except Exception as e:
        print(f"[WARN] Could not create model comparison heatmap: {e}")


def create_summary_dashboard(
    results_df: pd.DataFrame,
    cm_data: Dict[str, Tuple[List[int], List[int]]],
    norm_method: str,
    out_dir: Path,
    base_name: str,
    class_balance: Dict[str, Dict[str, float]] = None,
) -> None:
    """Create a comprehensive dashboard with multiple visualizations.
    
    Layout:
        Row 1: Performance summary table (with class balance, sorted by F1-weighted)
        Row 2: F1-weighted model comparison (bar) + F1-weighted across CV folds (line)
        Row 3: Confusion matrices for top 2 models (by F1-weighted)
        Row 4: ROC curves for top 2 models + Specificity vs Sensitivity
        Row 5: Precision-Recall-F1 trade-off comparison by model
    """
    
    try:
        # Use all results (only one normalization method now)
        norm_results = results_df.copy()
        
        if norm_results.empty:
            return
        
        # Calculate model rankings by F1-weighted (descending)
        model_f1_scores = norm_results.groupby('model')['f1_weighted'].mean().sort_values(ascending=False)
        models_sorted = model_f1_scores.index.tolist()
        top_2_models = models_sorted[:2] if len(models_sorted) >= 2 else models_sorted
        
        # Create figure with 5 rows and increased spacing
        fig = plt.figure(figsize=(22, 28))
        gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.35, 
                              height_ratios=[1.2, 1, 1.2, 1.2, 1])
        
        # =====================================================================
        # ROW 1: Performance Summary Table (spans both columns)
        # =====================================================================
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Build table data sorted by F1-weighted (best first)
        stats_data = []
        for model in models_sorted:
            model_data = norm_results[norm_results['model'] == model]
            roc_auc_val = model_data['roc_auc'].mean() if 'roc_auc' in model_data.columns else float('nan')
            roc_auc_std = model_data['roc_auc'].std() if 'roc_auc' in model_data.columns else float('nan')
            spec_val = model_data['specificity'].mean() if 'specificity' in model_data.columns else float('nan')
            spec_std = model_data['specificity'].std() if 'specificity' in model_data.columns else float('nan')
            
            stats_data.append([
                model,
                f"{model_data['f1_weighted'].mean():.3f} ± {model_data['f1_weighted'].std():.3f}",
                f"{model_data['accuracy'].mean():.3f} ± {model_data['accuracy'].std():.3f}",
                f"{model_data['precision'].mean():.3f} ± {model_data['precision'].std():.3f}",
                f"{model_data['recall'].mean():.3f} ± {model_data['recall'].std():.3f}",
                f"{spec_val:.3f} ± {spec_std:.3f}" if not np.isnan(spec_val) else "N/A",
                f"{roc_auc_val:.3f} ± {roc_auc_std:.3f}" if not np.isnan(roc_auc_val) else "N/A"
            ])
        
        # Determine class balance string
        if class_balance is not None:
            class_0_pct = class_balance.get('0', {}).get('percent', 0) * 100
            class_1_pct = class_balance.get('1', {}).get('percent', 0) * 100
            balance_str = f"Class Balance: Healthy={class_0_pct:.1f}% | Depressed={class_1_pct:.1f}%"
        else:
            balance_str = "Class Balance: N/A"
        
        col_labels = ['Model', 'F1-Weighted', 'Accuracy', 'Precision', 'Recall (Sens.)', 'Specificity', 'ROC-AUC']
        
        table = ax_table.table(cellText=stats_data,
                               colLabels=col_labels,
                               cellLoc='center',
                               loc='center',
                               colColours=['#4a90d9']*7)
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 2.2)
        
        # Style header row
        for j in range(len(col_labels)):
            table[(0, j)].set_text_props(fontweight='bold', color='white')
            table[(0, j)].set_facecolor('#2c5282')
        
        # Highlight best F1-weighted (first row after header)
        if len(stats_data) > 0:
            for j in range(len(col_labels)):
                table[(1, j)].set_facecolor('#c6f6d5')  # Light green for best model
        
        # Highlight best values in each metric column (columns 1-6)
        for col_idx in range(1, 7):
            col_values = []
            for row in stats_data:
                try:
                    val = float(row[col_idx].split(' ±')[0]) if '±' in row[col_idx] and row[col_idx] != 'N/A' else -1
                except:
                    val = -1
                col_values.append(val)
            
            if max(col_values) > 0:
                best_row = col_values.index(max(col_values))
                if best_row != 0:  # Don't override if already best overall
                    table[(best_row + 1, col_idx)].set_facecolor('#bee3f8')  # Light blue
        
        ax_table.set_title(f'Performance Summary (Mean ± Std) — Sorted by F1-Weighted\n{balance_str}', 
                          pad=25, fontsize=14, fontweight='bold')
        
        # =====================================================================
        # ROW 2: F1-Weighted Model Comparison (bar) + F1 across CV Folds (line)
        # =====================================================================
        
        # Left: F1-weighted bar chart (sorted)
        ax_f1_bar = fig.add_subplot(gs[1, 0])
        f1_means = model_f1_scores.values
        colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models_sorted)))[::-1]
        bars = ax_f1_bar.barh(range(len(models_sorted)), f1_means, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax_f1_bar.set_yticks(range(len(models_sorted)))
        ax_f1_bar.set_yticklabels(models_sorted, fontsize=10)
        ax_f1_bar.set_xlabel('Mean F1-Weighted Score', fontsize=11)
        ax_f1_bar.set_title('Model Comparison by F1-Weighted', fontsize=12, fontweight='bold', pad=15)
        ax_f1_bar.grid(axis='x', alpha=0.3)
        ax_f1_bar.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, f1_means):
            ax_f1_bar.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                          f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Right: F1-weighted across CV folds
        ax_f1_folds = fig.add_subplot(gs[1, 1])
        colors_line = plt.cm.tab10(np.linspace(0, 1, len(models_sorted)))
        for i, model in enumerate(models_sorted):
            model_data = norm_results[norm_results['model'] == model].sort_values('fold')
            ax_f1_folds.plot(model_data['fold'], model_data['f1_weighted'], 
                            marker='o', label=model, color=colors_line[i], linewidth=2, markersize=8)
        ax_f1_folds.set_xlabel('CV Fold', fontsize=11)
        ax_f1_folds.set_ylabel('F1-Weighted Score', fontsize=11)
        ax_f1_folds.set_title('F1-Weighted Across CV Folds', fontsize=12, fontweight='bold', pad=15)
        ax_f1_folds.legend(loc='lower left', fontsize=8, framealpha=0.9)
        ax_f1_folds.grid(alpha=0.3)
        ax_f1_folds.set_ylim(0, 1.05)
        
        # =====================================================================
        # ROW 3: Confusion Matrices for Top 2 Models
        # =====================================================================
        
        for idx, model_name in enumerate(top_2_models):
            ax_cm = fig.add_subplot(gs[2, idx])
            
            if model_name in cm_data:
                y_true, y_pred = cm_data[model_name]
                cm = confusion_matrix(np.array(y_true), np.array(y_pred), labels=[0, 1])
                
                if HAS_SEABORN:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                               xticklabels=['Healthy', 'Depressed'],
                               yticklabels=['Healthy', 'Depressed'],
                               annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                               cbar_kws={'shrink': 0.8})
                else:
                    im = ax_cm.imshow(cm, cmap='Blues')
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(['Healthy', 'Depressed'], fontsize=10)
                    ax_cm.set_yticklabels(['Healthy', 'Depressed'], fontsize=10)
                    plt.colorbar(im, ax=ax_cm, shrink=0.8)
                    for i in range(2):
                        for j in range(2):
                            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', 
                                      fontsize=16, fontweight='bold',
                                      color='white' if cm[i, j] > cm.max()/2 else 'black')
                
                # Calculate metrics for subtitle
                tn, fp, fn, tp = cm.ravel()
                acc = (tp + tn) / (tp + tn + fp + fn)
                rank_label = "Best" if idx == 0 else "2nd Best"
                ax_cm.set_title(f'#{idx+1} {model_name} ({rank_label})\nAccuracy: {acc:.3f}', 
                               fontsize=11, fontweight='bold', pad=15)
                ax_cm.set_ylabel('True Label', fontsize=10)
                ax_cm.set_xlabel('Predicted Label', fontsize=10)
            else:
                ax_cm.text(0.5, 0.5, f'No CM data for {model_name}', ha='center', va='center', fontsize=12)
                ax_cm.set_title(f'#{idx+1} {model_name}', fontsize=11, fontweight='bold', pad=15)
        
        # =====================================================================
        # ROW 4: ROC Curves for Top 2 Models + Specificity vs Sensitivity
        # =====================================================================
        
        # Left: ROC curves for top 2 models
        ax_roc = fig.add_subplot(gs[3, 0])
        roc_colors = ['#e74c3c', '#3498db']  # Red for best, blue for 2nd
        
        for idx, model_name in enumerate(top_2_models):
            # We need to get ROC data - check if it's available in results
            model_data = norm_results[norm_results['model'] == model_name]
            
            # Collect y_true and y_pred_proba from cm_data (we'll need to pass roc_data separately)
            if model_name in cm_data:
                y_true_list, y_pred_list = cm_data[model_name]
                # Try to compute ROC if we have probability data
                # For now, we'll just show the confusion matrix-based points
                
                # Plot as a point (TPR vs FPR from confusion matrix)
                y_true_arr = np.array(y_true_list)
                y_pred_arr = np.array(y_pred_list)
                cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                rank_label = "Best" if idx == 0 else "2nd"
                ax_roc.scatter(fpr, tpr, s=200, c=roc_colors[idx], 
                              label=f'{model_name} ({rank_label}): TPR={tpr:.2f}, FPR={fpr:.2f}',
                              edgecolors='black', linewidth=1.5, zorder=5)
        
        # Plot diagonal
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC=0.5)')
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax_roc.set_title('ROC Space: Top 2 Models', fontsize=12, fontweight='bold', pad=15)
        ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax_roc.grid(alpha=0.3)
        ax_roc.set_aspect('equal')
        
        # Right: Specificity vs Sensitivity for all models
        ax_spec_sens = fig.add_subplot(gs[3, 1])
        if 'specificity' in norm_results.columns and 'recall' in norm_results.columns:
            spec_means = [norm_results[norm_results['model']==m]['specificity'].mean() for m in models_sorted]
            sens_means = [norm_results[norm_results['model']==m]['recall'].mean() for m in models_sorted]
            
            colors_scatter = plt.cm.tab10(np.linspace(0, 1, len(models_sorted)))
            for i, model in enumerate(models_sorted):
                marker = '*' if model in top_2_models else 'o'
                size = 250 if model in top_2_models else 100
                ax_spec_sens.scatter(spec_means[i], sens_means[i], s=size, label=model, 
                                    color=colors_scatter[i], edgecolors='black', linewidth=1,
                                    marker=marker)
            
            ax_spec_sens.set_xlabel('Specificity (True Negative Rate)', fontsize=11)
            ax_spec_sens.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
            ax_spec_sens.set_title('Specificity vs Sensitivity\n(★ = Top 2 Models)', fontsize=12, fontweight='bold', pad=15)
            ax_spec_sens.set_xlim(-0.02, 1.05)
            ax_spec_sens.set_ylim(-0.02, 1.05)
            ax_spec_sens.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax_spec_sens.grid(alpha=0.3)
            ax_spec_sens.legend(fontsize=8, loc='lower right', framealpha=0.9)
            ax_spec_sens.set_aspect('equal')
        else:
            ax_spec_sens.text(0.5, 0.5, 'Specificity data not available', ha='center', va='center', fontsize=12)
            ax_spec_sens.set_title('Specificity vs Sensitivity', fontsize=12, fontweight='bold', pad=15)
        
        # =====================================================================
        # ROW 5: Precision-Recall-F1 Trade-off (spans both columns)
        # =====================================================================
        ax_prf = fig.add_subplot(gs[4, :])
        x = np.arange(len(models_sorted))
        width = 0.25
        
        precision_means = [norm_results[norm_results['model']==m]['precision'].mean() for m in models_sorted]
        recall_means = [norm_results[norm_results['model']==m]['recall'].mean() for m in models_sorted]
        f1_means_list = [norm_results[norm_results['model']==m]['f1_weighted'].mean() for m in models_sorted]
        
        bars1 = ax_prf.bar(x - width, precision_means, width, label='Precision', alpha=0.85, color='#27ae60', edgecolor='black', linewidth=0.5)
        bars2 = ax_prf.bar(x, recall_means, width, label='Recall (Sensitivity)', alpha=0.85, color='#3498db', edgecolor='black', linewidth=0.5)
        bars3 = ax_prf.bar(x + width, f1_means_list, width, label='F1-Weighted', alpha=0.85, color='#e74c3c', edgecolor='black', linewidth=0.5)
        
        ax_prf.set_ylabel('Score', fontsize=12)
        ax_prf.set_title('Precision–Recall–F1 Trade-off Comparison by Model', fontsize=13, fontweight='bold', pad=20)
        ax_prf.set_xticks(x)
        ax_prf.set_xticklabels(models_sorted, rotation=30, ha='right', fontsize=10)
        ax_prf.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax_prf.grid(axis='y', alpha=0.3)
        ax_prf.set_ylim(0, 1.15)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax_prf.annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width()/2, height),
                                   xytext=(0, 4), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Highlight top 2 models with background
        for i, model in enumerate(models_sorted):
            if model in top_2_models:
                ax_prf.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='gold')
        
        # =====================================================================
        # Final layout and save
        # =====================================================================
        plt.suptitle(f'ML Classification Results Dashboard\nNormalization: {norm_method}', 
                    fontsize=18, y=0.995, fontweight='bold')
        
        save_path = out_dir / f"{base_name}_{norm_method}_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[INFO] Saved dashboard to {save_path}")
        
    except Exception as e:
        import traceback
        print(f"[WARN] Could not create dashboard for {norm_method}: {e}")
        traceback.print_exc()


def create_test_dashboard(
    test_results: Dict[str, Dict[str, float]],
    test_cm_data: Dict[str, Tuple[List[int], List[int]]],
    cv_results_df: pd.DataFrame,
    norm_method: str,
    out_dir: Path,
    base_name: str,
    class_balance: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Create dashboard using TEST SET results (same as BioSig_ML)."""
    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "N/A"

    try:
        if not test_results:
            print("[WARN] No test results provided for dashboard")
            return
        test_rows = []
        for model_name, metrics in test_results.items():
            row = {"model": model_name}
            for k, v in metrics.items():
                if k not in ["cm", "y_true", "y_pred", "y_pred_proba"]:
                    row[k] = v
            test_rows.append(row)
        test_df = pd.DataFrame(test_rows)
        if test_df.empty:
            return
        test_df = test_df.sort_values("f1_weighted", ascending=False).reset_index(drop=True)
        models_sorted = test_df["model"].tolist()
        top_2_models = models_sorted[:2] if len(models_sorted) >= 2 else models_sorted

        fig = plt.figure(figsize=(22, 28))
        gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.35, height_ratios=[1.2, 1, 1.2, 1.2, 1])

        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis("tight")
        ax_table.axis("off")
        stats_data = []
        for _, row in test_df.iterrows():
            stats_data.append([
                row["model"],
                _fmt(row.get("f1_weighted")),
                _fmt(row.get("accuracy")),
                _fmt(row.get("precision")),
                _fmt(row.get("recall")),
                _fmt(row.get("specificity")),
                _fmt(row.get("roc_auc")),
            ])
        if class_balance:
            balance_str = f"Healthy={class_balance.get('0', {}).get('percent', 0)*100:.1f}% | Depressed={class_balance.get('1', {}).get('percent', 0)*100:.1f}%"
        else:
            balance_str = "N/A"
        test_size = len(test_cm_data[list(test_cm_data.keys())[0]][0]) if test_cm_data else "N/A"
        col_labels = ["Model", "F1-Weighted", "Accuracy", "Precision", "Recall (Sens.)", "Specificity", "ROC-AUC"]
        table = ax_table.table(cellText=stats_data, colLabels=col_labels, cellLoc="center", loc="center", colColours=["#2d5016"] * 7)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 2.2)
        for j in range(len(col_labels)):
            table[(0, j)].set_text_props(fontweight="bold", color="white")
            table[(0, j)].set_facecolor("#1a5f1a")
        if stats_data:
            for j in range(len(col_labels)):
                table[(1, j)].set_facecolor("#90EE90")
        ax_table.set_title(f"TEST SET Performance Summary — Sorted by F1-Weighted\nClass Balance: {balance_str} | Test Size: {test_size}", pad=25, fontsize=14, fontweight="bold", color="#1a5f1a")

        ax_f1 = fig.add_subplot(gs[1, 0])
        f1_values = test_df["f1_weighted"].values
        colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models_sorted)))[::-1]
        bars = ax_f1.barh(range(len(models_sorted)), f1_values, color=colors_bar, edgecolor="black", linewidth=0.5)
        ax_f1.set_yticks(range(len(models_sorted)))
        ax_f1.set_yticklabels(models_sorted, fontsize=10)
        ax_f1.set_xlabel("Test F1-Weighted")
        ax_f1.set_title("Test Set: Model Comparison by F1-Weighted", fontsize=12, fontweight="bold", pad=15)
        ax_f1.set_xlim(0, 1)
        ax_f1.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, f1_values):
            v = val if not (isinstance(val, float) and np.isnan(val)) else 0
            ax_f1.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        ax_metrics = fig.add_subplot(gs[1, 1])
        x = np.arange(len(models_sorted))
        width = 0.15
        for i, metric in enumerate(["accuracy", "precision", "recall", "specificity", "roc_auc"]):
            if metric in test_df.columns:
                vals = [test_df[test_df["model"] == m][metric].values[0] if m in test_df["model"].values else 0 for m in models_sorted]
                offset = (i - 2.5) * width
                ax_metrics.bar(x + offset, vals, width, label=metric.replace("_", " ").title(), alpha=0.8)
        ax_metrics.set_ylabel("Score")
        ax_metrics.set_title("Test Set: All Metrics", fontsize=12, fontweight="bold", pad=15)
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(models_sorted, rotation=30, ha="right", fontsize=9)
        ax_metrics.legend(loc="lower right", fontsize=8)
        ax_metrics.set_ylim(0, 1.15)
        ax_metrics.grid(axis="y", alpha=0.3)

        for idx, model_name in enumerate(top_2_models):
            ax_cm = fig.add_subplot(gs[2, idx])
            if model_name in test_cm_data:
                y_true, y_pred = test_cm_data[model_name]
                cm = confusion_matrix(np.array(y_true), np.array(y_pred), labels=[NEG_LABEL, POS_LABEL])
                if HAS_SEABORN:
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax_cm,
                                xticklabels=["Healthy", "Depressed"], yticklabels=["Healthy", "Depressed"],
                                annot_kws={"fontsize": 14}, cbar_kws={"shrink": 0.8})
                else:
                    im = ax_cm.imshow(cm, cmap="Greens")
                    ax_cm.set_xticks([0, 1])
                    ax_cm.set_yticks([0, 1])
                    ax_cm.set_xticklabels(["Healthy", "Depressed"])
                    ax_cm.set_yticklabels(["Healthy", "Depressed"])
                    for i in range(2):
                        for j in range(2):
                            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                                      color="white" if cm[i, j] > cm.max() / 2 else "black")
                tn, fp, fn, tp = cm.ravel()
                acc = (tp + tn) / (tp + tn + fp + fn)
                rank = "Best" if idx == 0 else "2nd Best"
                ax_cm.set_title(f"TEST: #{idx+1} {model_name} ({rank})\nAccuracy: {acc:.3f}", fontsize=11, fontweight="bold", pad=15, color="#1a5f1a")
            else:
                ax_cm.text(0.5, 0.5, f"No test CM for {model_name}", ha="center", va="center")
                ax_cm.set_title(f"#{idx+1} {model_name}", fontsize=11, fontweight="bold", pad=15)

        ax_roc = fig.add_subplot(gs[3, 0])
        for idx, model_name in enumerate(top_2_models):
            if model_name in test_cm_data:
                y_true, y_pred = test_cm_data[model_name]
                cm = confusion_matrix(np.array(y_true), np.array(y_pred), labels=[NEG_LABEL, POS_LABEL])
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                rank = "Best" if idx == 0 else "2nd"
                ax_roc.scatter(fpr, tpr, s=250, label=f"{model_name} ({rank}): TPR={tpr:.2f}, FPR={fpr:.2f}", marker="*")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.set_xlabel("FPR (1 - Specificity)")
        ax_roc.set_ylabel("TPR (Sensitivity)")
        ax_roc.set_title("TEST: ROC Space — Top 2 Models", fontsize=12, fontweight="bold", pad=15, color="#1a5f1a")
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(alpha=0.3)
        ax_roc.set_aspect("equal")

        ax_spec = fig.add_subplot(gs[3, 1])
        if "specificity" in test_df.columns and "recall" in test_df.columns:
            for i, model in enumerate(models_sorted):
                model_df = test_df[test_df["model"] == model]
                if len(model_df) > 0:
                    spec = model_df["specificity"].values[0]
                    sens = model_df["recall"].values[0]
                    marker = "*" if model in top_2_models else "o"
                    size = 300 if model in top_2_models else 120
                    ax_spec.scatter(spec, sens, s=size, label=model, marker=marker)
            ax_spec.set_xlabel("Specificity")
            ax_spec.set_ylabel("Sensitivity")
            ax_spec.set_title("TEST: Specificity vs Sensitivity (★ = Top 2)", fontsize=12, fontweight="bold", pad=15, color="#1a5f1a")
            ax_spec.set_xlim(-0.02, 1.05)
            ax_spec.set_ylim(-0.02, 1.05)
            ax_spec.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax_spec.legend(fontsize=8, loc="lower right")
        ax_spec.grid(alpha=0.3)

        ax_prf = fig.add_subplot(gs[4, :])
        x = np.arange(len(models_sorted))
        w = 0.25
        prec = [test_df[test_df["model"] == m]["precision"].values[0] if len(test_df[test_df["model"] == m]) > 0 else 0.0 for m in models_sorted]
        rec = [test_df[test_df["model"] == m]["recall"].values[0] if len(test_df[test_df["model"] == m]) > 0 else 0.0 for m in models_sorted]
        f1 = [test_df[test_df["model"] == m]["f1_weighted"].values[0] if len(test_df[test_df["model"] == m]) > 0 else 0.0 for m in models_sorted]
        ax_prf.bar(x - w, prec, w, label="Precision", alpha=0.85, color="#27ae60")
        ax_prf.bar(x, rec, w, label="Recall", alpha=0.85, color="#3498db")
        ax_prf.bar(x + w, f1, w, label="F1-Weighted", alpha=0.85, color="#e74c3c")
        ax_prf.set_ylabel("Score")
        ax_prf.set_title("TEST: Precision–Recall–F1 by Model", fontsize=13, fontweight="bold", pad=20, color="#1a5f1a")
        ax_prf.set_xticks(x)
        ax_prf.set_xticklabels(models_sorted, rotation=30, ha="right")
        ax_prf.legend(loc="upper right")
        ax_prf.set_ylim(0, 1.15)
        ax_prf.grid(axis="y", alpha=0.3)

        plt.suptitle(f"ML Classification Results Dashboard (TEST SET)\nNormalization: {norm_method}", fontsize=18, y=0.995, fontweight="bold", color="#1a5f1a")
        save_path = out_dir / f"{base_name}_{norm_method}_test_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        print(f"[INFO] Saved TEST dashboard to {save_path}")
    except Exception as e:
        import traceback
        print(f"[WARN] Could not create TEST dashboard: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI and main orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text ML: 5-fold CV binary classification for depression detection (no preprocessing)")

    parser.add_argument(
        "--text_data_path",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/4_Text_data/Data/text_features_3370.csv", # 3370 had the best results
        help="Path to text modality CSV (patient_id + numeric features).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "CR", "CRADK", "ADK", "SHAM"],
        help="Condition to filter (or 'all').",
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="Number of CV folds (stratified by group ID, leakage-free).",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    cfg = parse_args()

    print("=" * 80)
    print("[INFO] Text ML — 5-fold CV binary classification (no normalization, no feature selection; leakage-free)")
    print(json.dumps(vars(cfg), indent=2))
    print("=" * 80)

    root_results_dir = Path(__file__).parent / "result_nested_CV"
    out_dir = root_results_dir / f"text_ml_{cfg.condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"text_ml_{cfg.condition}_cv{cfg.k_fold}"

    # Load pre-prepared text data (no preprocessing applied here)
    df = load_text_data(
        text_csv_path=Path(cfg.text_data_path),
        condition=cfg.condition,
    )
    print(f"[INFO] Loaded data shape: {df.shape}")

    df = sanitize_feature_names(df)
    print(f"[INFO] Unique participants: {df['ID'].nunique()}")

    label_counts = df["label"].value_counts().to_dict()
    print(f"[INFO] Label counts: {label_counts}")
    total_samples = float(len(df))
    class_balance = {
        str(cls): {"count": int(cnt), "percent": float(cnt / total_samples) if total_samples > 0 else float("nan")}
        for cls, cnt in sorted(label_counts.items())
    }

    y = df["label"].to_numpy(dtype=int)
    groups = df["ID"].astype(str).to_numpy()
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    feature_df = df.drop(columns=meta_cols_present, errors="ignore")
    numeric_cols_final = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feature_df[numeric_cols_final].copy()
    print(f"[INFO] Feature matrix X: {X.shape}")
    if X.shape[1] == 0:
        raise ValueError("No numeric features found. Check your data.")

    args_json_path = out_dir / f"{base_name}_args.json"
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": vars(cfg), "class_balance": class_balance, "n_samples": int(total_samples), "n_features": int(X.shape[1])},
            f, indent=2,
        )
    print(f"[INFO] Saved args to {args_json_path}")

    norm_method = "none"
    models = get_classifier_models()
    k_fold = cfg.k_fold
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < k_fold:
        raise ValueError(f"Not enough unique groups ({n_unique_groups}) for {k_fold} folds.")

    # 5-fold stratified group CV: split by participant ID so no leakage (same ID never in train and test)
    cv = StratifiedGroupKFold(n_splits=k_fold, shuffle=True, random_state=RANDOM_STATE)
    all_results_rows: List[Dict[str, float]] = []
    test_results_per_fold: Dict[str, List[Dict[str, float]]] = {name: [] for name in models.keys()}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        print("\n" + "=" * 80)
        print(f"[INFO] FOLD {fold}/{k_fold} | train={len(train_idx)}, test={len(test_idx)}")
        print("=" * 80)

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, base_clf in models.items():
            clf = clone(base_clf)
            clf.fit(X_train, y_train)
            test_metrics = evaluate_model(clf, X_test, y_test)
            test_results_per_fold[model_name].append(test_metrics)
            row = {"condition": cfg.condition, "fold": fold, "model": model_name}
            for k, v in test_metrics.items():
                if k not in ("cm", "y_true", "y_pred", "y_pred_proba"):
                    row[k] = v
            all_results_rows.append(row)
            print(f"  {model_name} | Test F1_wgt={test_metrics.get('f1_weighted', 0):.4f} ROC-AUC={test_metrics.get('roc_auc', 0):.4f}")

    # ----------  Aggregate and report  ----------
    print("\n" + "=" * 80)
    print(f"[RESULTS] {k_fold}-fold CV — test set metrics per fold (mean ± std)")
    print("=" * 80)
    for model_name in models.keys():
        list_m = test_results_per_fold.get(model_name, [])
        if not list_m:
            continue
        mean_f1 = float(np.mean([m.get("f1_weighted", 0) for m in list_m]))
        std_f1 = float(np.std([m.get("f1_weighted", 0) for m in list_m]))
        aucs = [m.get("roc_auc", 0) for m in list_m if isinstance(m.get("roc_auc"), (int, float)) and not np.isnan(m.get("roc_auc", np.nan))]
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        print(f"{model_name:16s} | Test F1_wgt={mean_f1:.3f} ± {std_f1:.3f}  ROC-AUC={mean_auc:.3f}")

    if all_results_rows:
        results_df = pd.DataFrame(all_results_rows)
        test_results_path = out_dir / f"{base_name}_test_results.csv"
        results_df.to_csv(test_results_path, index=False)
        print(f"\n[INFO] Saved per-fold results to {test_results_path}")

        metric_cols = [c for c in results_df.columns if c not in ["condition", "fold", "model"]]
        summary_df = results_df.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
        if isinstance(summary_df.columns, pd.MultiIndex):
            summary_df.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in summary_df.columns.values]
        summary_path = out_dir / f"{base_name}_summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved summary results to {summary_path}")

        test_results: Dict[str, Dict[str, float]] = {}
        test_cm_data: Dict[str, Tuple[List[int], List[int]]] = {}
        test_roc_data: Dict[str, Tuple[List[float], List[float]]] = {}
        for model_name in models.keys():
            list_m = test_results_per_fold.get(model_name, [])
            if not list_m:
                continue
            all_y_true = []
            all_y_pred = []
            all_y_proba = []
            for m in list_m:
                all_y_true.extend(m.get("y_true", []))
                all_y_pred.extend(m.get("y_pred", []))
                if m.get("y_pred_proba"):
                    all_y_proba.extend(m["y_pred_proba"])
            test_cm_data[model_name] = (all_y_true, all_y_pred)
            if all_y_proba and len(all_y_proba) == len(all_y_true):
                test_roc_data[model_name] = (all_y_true, all_y_proba)
            yt, yp = np.array(all_y_true), np.array(all_y_pred)
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = (int(cm.ravel()[i]) for i in range(4)) if cm.size == 4 else (0, 0, 0, 0)
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
            roc_auc_val = float(roc_auc_score(yt, all_y_proba)) if all_y_proba and len(all_y_proba) == len(yt) else float("nan")
            test_results[model_name] = {
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, average="binary", zero_division=0)),
                "recall": float(recall_score(yt, yp, average="binary", zero_division=0)),
                "f1": float(f1_score(yt, yp, average="binary", zero_division=0)),
                "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
                "specificity": specificity,
                "roc_auc": roc_auc_val,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }

        print("\n[INFO] Generating confusion matrices (aggregated test set)...")
        for model_name, (y_true_list, y_pred_list) in test_cm_data.items():
            if y_true_list and y_pred_list:
                plot_confusion_matrix(
                    np.array(y_true_list), np.array(y_pred_list),
                    model_name, norm_method,
                    out_dir / f"{base_name}_{model_name}_confusion_matrix.png",
                )

        print("\n[INFO] Generating ROC curves (aggregated test set)...")
        if test_roc_data:
            plot_roc_curves(test_roc_data, norm_method, out_dir / f"{base_name}_roc_curves.png")

        print("\n[INFO] Generating model comparison heatmap...")
        plot_model_comparison_heatmap(summary_df, out_dir, base_name)

        print("\n[INFO] Generating TEST SET dashboard...")
        create_test_dashboard(
            test_results, test_cm_data, results_df, norm_method,
            out_dir, base_name, class_balance,
        )

    print("\n" + "=" * 80)
    print("[INFO] 5-fold CV experiment completed successfully (leakage-free).")
    print(f"[INFO] Results saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()