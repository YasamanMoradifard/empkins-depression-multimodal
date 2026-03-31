"""
Visualization utilities for model outputs.

Creates plots for ROC curves, attention weights, and embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from sklearn.metrics import roc_curve, auc
import json


def plot_roc_curve(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   save_path: Optional[Path] = None,
                   title: str = 'ROC Curve'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_weights(attention_weights: List[Dict],
                          save_path: Optional[Path] = None,
                          max_subjects: int = 5):
    """
    Plot attention weights for example subjects.
    
    Args:
        attention_weights: List of dicts with 'subject_id', 'attention_weights', 'n_files'
        save_path: Path to save figure
        max_subjects: Maximum number of subjects to plot
    """
    n_subjects = min(len(attention_weights), max_subjects)
    
    fig, axes = plt.subplots(n_subjects, 1, figsize=(10, 2 * n_subjects))
    if n_subjects == 1:
        axes = [axes]
    
    for i, attn_data in enumerate(attention_weights[:n_subjects]):
        subject_id = attn_data['subject_id']
        weights = np.array(attn_data['attention_weights'])
        n_files = attn_data['n_files']
        
        axes[i].bar(range(n_files), weights, alpha=0.7)
        axes[i].set_xlabel('File Index')
        axes[i].set_ylabel('Attention Weight')
        axes[i].set_title(f'Subject {subject_id} - Attention Weights')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict,
                         save_path: Optional[Path] = None):
    """Plot training history (loss and metrics over epochs)."""
    epochs = range(1, len(history['train']['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train']['loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val']['loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 1].plot(epochs, history['train']['auc'], label='Train', marker='o')
    axes[0, 1].plot(epochs, history['val']['auc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('Training and Validation AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train']['accuracy'], label='Train', marker='o')
    axes[1, 0].plot(epochs, history['val']['accuracy'], label='Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training and Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1
    axes[1, 1].plot(epochs, history['train']['f1'], label='Train', marker='o')
    axes[1, 1].plot(epochs, history['val']['f1'], label='Val', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('Training and Validation F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_metrics_separate(history: Dict,
                                   save_dir: Optional[Path] = None,
                                   fold_name: str = ""):
    """
    Plot training history as separate plots: Loss, AUC, and F1.
    
    Args:
        history: Dict with 'train' and 'val' keys, each containing 'loss', 'auc', 'f1' lists
        save_dir: Directory to save plots (will create loss.png, auc.png, f1.png)
        fold_name: Optional fold identifier for plot titles
    """
    epochs = range(1, len(history['train']['loss']) + 1)
    title_suffix = f" - {fold_name}" if fold_name else ""
    
    # Plot 1: Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train']['loss'], label='Train', marker='o', linewidth=2, markersize=4)
    plt.plot(epochs, history['val']['loss'], label='Validation', marker='s', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss{title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_path = save_dir / 'loss.png'
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {loss_path}")
    else:
        plt.show()
    plt.close()
    
    # Plot 2: AUC
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train']['auc'], label='Train', marker='o', linewidth=2, markersize=4)
    plt.plot(epochs, history['val']['auc'], label='Validation', marker='s', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.title(f'Training and Validation AUC{title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        auc_path = save_dir / 'auc.png'
        plt.savefig(auc_path, dpi=150, bbox_inches='tight')
        print(f"Saved AUC plot to {auc_path}")
    else:
        plt.show()
    plt.close()
    
    # Plot 3: F1
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train']['f1'], label='Train', marker='o', linewidth=2, markersize=4)
    plt.plot(epochs, history['val']['f1'], label='Validation', marker='s', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'Training and Validation F1{title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        f1_path = save_dir / 'f1.png'
        plt.savefig(f1_path, dpi=150, bbox_inches='tight')
        print(f"Saved F1 plot to {f1_path}")
    else:
        plt.show()
    plt.close()
    
    # Plot 4: Overfitting Analysis (NEW)
    plot_overfitting_analysis(history, save_dir, fold_name)


def plot_overfitting_analysis(history: Dict,
                              save_dir: Optional[Path] = None,
                              fold_name: str = ""):
    """
    Generate comprehensive overfitting analysis plot.
    
    Shows:
    - Train vs Val AUC gap over time
    - Train vs Val Loss gap over time  
    - Learning rate changes (if available)
    - Overfitting indicators and warnings
    
    Args:
        history: Training history dict
        save_dir: Directory to save plot
        fold_name: Optional fold identifier
    """
    epochs = list(range(1, len(history['train']['loss']) + 1))
    title_suffix = f" - {fold_name}" if fold_name else ""
    
    train_auc = np.array(history['train']['auc'])
    val_auc = np.array(history['val']['auc'])
    train_loss = np.array(history['train']['loss'])
    val_loss = np.array(history['val']['loss'])
    
    # Compute gaps
    auc_gap = train_auc - val_auc  # Positive = overfitting
    loss_gap = val_loss - train_loss  # Positive = overfitting
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Overfitting Analysis{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: AUC Comparison with gap shading
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_auc, 'b-', label='Train AUC', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_auc, 'r-', label='Val AUC', linewidth=2, marker='s', markersize=3)
    ax1.fill_between(epochs, val_auc, train_auc, alpha=0.3, 
                     color='red' if np.mean(auc_gap) > 0 else 'green',
                     label=f'Gap (avg: {np.mean(auc_gap):.3f})')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('Train vs Validation AUC')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Loss Comparison with gap shading
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax2.fill_between(epochs, train_loss, val_loss, alpha=0.3,
                     color='red' if np.mean(loss_gap) > 0 else 'green',
                     label=f'Gap (avg: {np.mean(loss_gap):.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Train vs Validation Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gap Evolution
    ax3 = axes[1, 0]
    ax3.plot(epochs, auc_gap, 'purple', label='AUC Gap (Train - Val)', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='No Gap (Ideal)')
    ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning (0.1)')
    ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Severe (0.2)')
    ax3.fill_between(epochs, 0, auc_gap, where=(auc_gap > 0.1), alpha=0.3, color='orange')
    ax3.fill_between(epochs, 0, auc_gap, where=(auc_gap > 0.2), alpha=0.3, color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC Gap')
    ax3.set_title('Overfitting Gap Over Time')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute overfitting metrics
    final_train_auc = train_auc[-1]
    final_val_auc = val_auc[-1]
    best_val_auc = np.max(val_auc)
    best_val_epoch = np.argmax(val_auc) + 1
    final_gap = auc_gap[-1]
    max_gap = np.max(auc_gap)
    avg_gap = np.mean(auc_gap[-10:]) if len(auc_gap) >= 10 else np.mean(auc_gap)  # Last 10 epochs
    
    # Determine overfitting status
    if final_gap > 0.2 or avg_gap > 0.15:
        status = "🔴 SEVERE OVERFITTING"
        status_color = 'red'
    elif final_gap > 0.1 or avg_gap > 0.1:
        status = "🟡 MODERATE OVERFITTING"
        status_color = 'orange'
    elif final_gap > 0.05:
        status = "🟢 MILD OVERFITTING"
        status_color = 'yellowgreen'
    else:
        status = "✅ GOOD GENERALIZATION"
        status_color = 'green'
    
    # Create summary text
    summary_text = f"""
    ╔══════════════════════════════════════════════════╗
    ║           OVERFITTING ANALYSIS SUMMARY           ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  Status: {status:<30}       ║
    ║                                                  ║
    ╠══════════════════════════════════════════════════╣
    ║  TRAINING METRICS                                ║
    ║  ─────────────────                               ║
    ║  Final Train AUC:     {final_train_auc:.4f}                       ║
    ║  Final Val AUC:       {final_val_auc:.4f}                       ║
    ║  Best Val AUC:        {best_val_auc:.4f} (epoch {best_val_epoch:3d})            ║
    ║                                                  ║
    ╠══════════════════════════════════════════════════╣
    ║  OVERFITTING INDICATORS                          ║
    ║  ──────────────────────                          ║
    ║  Final Gap:           {final_gap:.4f}                       ║
    ║  Max Gap:             {max_gap:.4f}                       ║
    ║  Avg Gap (last 10):   {avg_gap:.4f}                       ║
    ║                                                  ║
    ╠══════════════════════════════════════════════════╣
    ║  INTERPRETATION                                  ║
    ║  ──────────────                                  ║
    ║  Gap < 0.05:  Excellent generalization          ║
    ║  Gap 0.05-0.1: Acceptable                        ║
    ║  Gap 0.1-0.2:  Overfitting - increase dropout   ║
    ║  Gap > 0.2:    Severe - simplify model          ║
    ╚══════════════════════════════════════════════════╝
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        overfitting_path = save_dir / 'overfitting_analysis.png'
        plt.savefig(overfitting_path, dpi=150, bbox_inches='tight')
        print(f"Saved overfitting analysis to {overfitting_path}")
        
        # Also save as JSON for programmatic access
        analysis_data = {
            'status': status,
            'final_train_auc': float(final_train_auc),
            'final_val_auc': float(final_val_auc),
            'best_val_auc': float(best_val_auc),
            'best_val_epoch': int(best_val_epoch),
            'final_gap': float(final_gap),
            'max_gap': float(max_gap),
            'avg_gap_last_10': float(avg_gap),
            'auc_gap_history': auc_gap.tolist(),
            'is_overfitting': final_gap > 0.1 or avg_gap > 0.1
        }
        json_path = save_dir / 'overfitting_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"Saved overfitting analysis JSON to {json_path}")
    else:
        plt.show()
    
    plt.close()
    
    return analysis_data if save_dir else None

