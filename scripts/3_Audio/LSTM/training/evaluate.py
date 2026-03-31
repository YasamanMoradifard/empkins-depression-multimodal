"""
Evaluation script for trained models.

Computes metrics and saves predictions with attention weights.
Generates confusion matrix and ROC curve visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from models.hierarchical_lstm import HierarchicalLSTMDepression
from utils.metrics import compute_metrics, compute_confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: Dict[str, int], save_path: Path, title: str = "Confusion Matrix"):
    """
    Plot and save confusion matrix as an image.
    
    Args:
        cm: Dict with 'tn', 'fp', 'fn', 'tp'
        save_path: Path to save the image
        title: Title for the plot
    """
    cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Healthy', 'Predicted Depressed'],
                    yticklabels=['Actual Healthy', 'Actual Depressed'],
                    annot_kws={'size': 14})
    else:
        plt.imshow(cm_array, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Predicted Healthy', 'Predicted Depressed'])
        plt.yticks(tick_marks, ['Actual Healthy', 'Actual Depressed'])
        thresh = cm_array.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm_array[i, j], 'd'),
                        horizontalalignment="center",
                        fontsize=14,
                        color="white" if cm_array[i, j] > thresh else "black")
    
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add summary statistics
    total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
    accuracy = (cm['tn'] + cm['tp']) / total if total > 0 else 0
    sensitivity = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
    specificity = cm['tn'] / (cm['tn'] + cm['fp']) if (cm['tn'] + cm['fp']) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}'
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, 
                   title: str = "ROC Curve"):
    """
    Plot and save ROC curve as an image.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        save_path: Path to save the image
        title: Title for the plot
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Mark optimal threshold point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Optimal threshold = {optimal_threshold:.3f}', zorder=5)
    plt.annotate(f'Threshold={optimal_threshold:.2f}', 
                 xy=(fpr[optimal_idx], tpr[optimal_idx]),
                 xytext=(fpr[optimal_idx]+0.1, tpr[optimal_idx]-0.1),
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def load_model(model_path: Path, device: torch.device, **model_kwargs) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same architecture
    model = HierarchicalLSTMDepression(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best validation AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")
    
    return model


def evaluate_model(model: torch.nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   save_predictions: bool = True,
                   output_dir: Optional[Path] = None) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dict with metrics and optionally saved predictions
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_subject_ids = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Validate batch structure
            required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels', 'subject_ids', 'subject_file_counts']
            if not all(key in batch for key in required_keys):
                missing = [key for key in required_keys if key not in batch]
                raise KeyError(f"Batch missing required keys: {missing}")
            
            batch['file_sequences'] = batch['file_sequences'].to(device)
            batch['frame_lengths'] = batch['frame_lengths'].to(device)
            batch['file_mask'] = batch['file_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(batch)
            if 'prediction' not in outputs:
                raise KeyError("Model output missing 'prediction' key")
            # Apply sigmoid to convert logits to probabilities
            predictions = torch.sigmoid(outputs['prediction'].squeeze(-1)).cpu().numpy()
            
            all_predictions.append(predictions)
            all_labels.append(batch['labels'].cpu().numpy())  # Move to CPU before converting to numpy
            all_subject_ids.extend(batch['subject_ids'])
            
            # Save attention weights if available
            if outputs['attention_weights'] is not None:
                attn = outputs['attention_weights'].cpu().numpy()
                # Store attention weights per subject (handle variable file counts)
                for i, subject_id in enumerate(batch['subject_ids']):
                    n_files = batch['subject_file_counts'][i]
                    all_attention_weights.append({
                        'subject_id': subject_id,
                        'attention_weights': attn[i, :n_files].tolist(),
                        'n_files': n_files
                    })
    
    # Check for empty dataloader
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - cannot compute metrics")
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions)
    cm = compute_confusion_matrix(all_labels, all_predictions > 0.5)
    
    # Store labels and predictions for visualization
    metrics['y_true'] = all_labels.tolist()
    metrics['y_pred'] = all_predictions.tolist()
    
    # Log all metrics
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"  ROC-AUC:      {metrics['auc']:.4f}")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:    {metrics['precision']:.4f}")
    logger.info(f"  Recall:       {metrics['recall']:.4f} (Sensitivity)")
    logger.info(f"  Specificity:  {metrics['specificity']:.4f}")
    logger.info(f"  F1 Score:     {metrics['f1']:.4f}")
    logger.info(f"  F1 Weighted:  {metrics['f1_weighted']:.4f}")
    logger.info(f"  F1 Macro:     {metrics['f1_macro']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"                 Healthy  Depressed")
    logger.info(f"  Actual Healthy    {cm['tn']:4d}      {cm['fp']:4d}")
    logger.info(f"  Actual Depressed  {cm['fn']:4d}      {cm['tp']:4d}")
    logger.info(f"{'='*50}")
    
    # Save predictions and visualizations
    if save_predictions and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'subject_id': all_subject_ids,
            'true_label': all_labels,
            'predicted_prob': all_predictions,
            'predicted_label': (all_predictions > 0.5).astype(int)
        })
        
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Save attention weights if available
        if all_attention_weights:
            attn_path = output_dir / 'attention_weights.json'
            with open(attn_path, 'w') as f:
                json.dump(all_attention_weights, f, indent=2)
            logger.info(f"Saved attention weights to {attn_path}")
        
        # Save confusion matrix image
        plot_confusion_matrix(
            cm=cm,
            save_path=output_dir / 'confusion_matrix.png',
            title='Confusion Matrix'
        )
        
        # Save ROC curve image
        plot_roc_curve(
            y_true=all_labels,
            y_pred=all_predictions,
            save_path=output_dir / 'roc_curve.png',
            title='ROC Curve'
        )
        
        # Save metrics as JSON
        metrics_path = output_dir / 'test_metrics.json'
        metrics_to_save = {k: v for k, v in metrics.items() if k not in ['y_true', 'y_pred']}
        metrics_to_save['confusion_matrix'] = cm
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module loaded. Import and use evaluate_model() function.")

