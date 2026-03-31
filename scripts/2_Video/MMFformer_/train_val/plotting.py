#-*- coding: utf-8 -*-
"""
Plotting utilities for training visualization
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path


def plot_train_val_curves(train_history, val_history, save_path, metric='acc'):
    """
    Plot training vs validation curves for a given metric.
    
    Args:
        train_history: List of training metric values per epoch
        val_history: List of validation metric values per epoch
        save_path: Path to save the plot
        metric: 'acc', 'loss', or 'f1'
    """
    epochs = range(1, len(train_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history, 'b-', label=f'Train {metric.upper()}', linewidth=2)
    plt.plot(epochs, val_history, 'r-', label=f'Validation {metric.upper()}', linewidth=2)
    plt.title(f'Train vs Validation {metric.upper()} Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path, split='validation'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        save_path: Path to save the confusion matrix image
        split: 'validation' or 'test' for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Depressed'],
                yticklabels=['Healthy', 'Depressed'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {split.capitalize()}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_data_statistics(dataloader, dataset_name='d02', condition='all', phase='all', modalities='av'):
    """
    Get statistics about the dataset.
    
    Args:
        dataloader: PyTorch DataLoader
        dataset_name: Name of the dataset
        condition: Condition filter applied
        phase: Phase filter applied
        modalities: Modalities used ('av', 'video', 'audio')
    
    Returns:
        dict: Statistics including counts, class balance, etc.
    """
    dataset = dataloader.dataset
    
    # Count samples and labels
    total_samples = len(dataset)
    
    # Handle Subset datasets - need to iterate through dataloader to get actual labels
    # or access underlying dataset if it's a Subset
    from torch.utils.data import Subset
    
    # Try to get labels from the dataset
    labels = []
    if isinstance(dataset, Subset):
        # For Subset, access the underlying dataset
        underlying_dataset = dataset.dataset
        if hasattr(underlying_dataset, 'labels'):
            # Get labels for the indices in this subset
            # Handle both old and new PyTorch Subset API
            if hasattr(dataset, 'indices'):
                subset_indices = dataset.indices
                labels = [underlying_dataset.labels[idx] for idx in subset_indices]
            else:
                # Fallback: iterate through dataloader if indices attribute doesn't exist
                for _, y, _ in dataloader:
                    labels.extend(y.cpu().numpy().flatten().tolist())
        else:
            # If no labels attribute, iterate through dataloader
            for _, y, _ in dataloader:
                labels.extend(y.cpu().numpy().flatten().tolist())
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # If no labels attribute, iterate through dataloader
        for _, y, _ in dataloader:
            labels.extend(y.cpu().numpy().flatten().tolist())
    
    # Count class distribution
    if labels:
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        healthy_count = label_counts.get(0, 0)
        depressed_count = label_counts.get(1, 0)
        healthy_pct = (healthy_count / total_samples * 100) if total_samples > 0 else 0
        depressed_pct = (depressed_count / total_samples * 100) if total_samples > 0 else 0
    else:
        healthy_count = depressed_count = healthy_pct = depressed_pct = 0
    
    stats = {
        'dataset_name': dataset_name,
        'condition': condition,
        'phase': phase,
        'modalities': modalities,
        'total_samples': total_samples,
        'healthy_count': healthy_count,
        'depressed_count': depressed_count,
        'healthy_percentage': healthy_pct,
        'depressed_percentage': depressed_pct,
        'class_balance_ratio': healthy_count / depressed_count if depressed_count > 0 else float('inf'),
    }
    
    return stats


def format_statistics_string(stats):
    """
    Format statistics dictionary as a readable string.
    
    Args:
        stats: Dictionary from get_data_statistics
    
    Returns:
        str: Formatted string
    """
    lines = [
        "=" * 60,
        "DATASET STATISTICS",
        "=" * 60,
        f"Dataset: {stats['dataset_name']}",
        f"Condition: {stats['condition']}",
        f"Phase: {stats['phase']}",
        f"Modalities: {stats['modalities']}",
        "",
        "Sample Counts:",
        f"  Total Samples: {stats['total_samples']}",
        f"  Healthy (Class 0): {stats['healthy_count']} ({stats['healthy_percentage']:.2f}%)",
        f"  Depressed (Class 1): {stats['depressed_count']} ({stats['depressed_percentage']:.2f}%)",
        f"  Class Balance Ratio (H/D): {stats['class_balance_ratio']:.2f}",
        "=" * 60,
    ]
    return "\n".join(lines)

