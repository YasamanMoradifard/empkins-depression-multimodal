"""
Training loop for hierarchical LSTM depression detection model.

Implements training with early stopping, metric logging, and model checkpointing.
See LSTM_ARCHITECTURE_GUIDE.md section 6 for training strategy details.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models.hierarchical_lstm import HierarchicalLSTMDepression
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation loss."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'loss'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'loss' (lower is better) or 'auc' (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        if mode == 'loss':
            self.best_metric = float('inf')
            self.is_better = lambda current, best: current < best - min_delta
        else:  # mode == 'auc'
            self.best_metric = 0.0
            self.is_better = lambda current, best: current > best + min_delta
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, val_metric: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_metric: Validation loss or AUC depending on mode
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(val_metric, self.best_metric):
            self.best_metric = val_metric
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def compute_pos_weight(labels: torch.Tensor) -> float:
    """Compute positive weight for weighted BCE loss."""
    n_positive = labels.sum().item()
    n_negative = len(labels) - n_positive
    if n_positive == 0:
        return 1.0
    return n_negative / n_positive


def apply_label_smoothing(labels: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """
    Apply label smoothing to binary labels.
    
    Transforms labels from {0, 1} to {smoothing, 1-smoothing}.
    This helps prevent overconfident predictions and improves generalization.
    
    Args:
        labels: Binary labels (0 or 1)
        smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = smooth to 0.1/0.9)
    
    Returns:
        Smoothed labels
    """
    if smoothing <= 0:
        return labels
    return labels * (1 - smoothing) + (1 - labels) * smoothing


def compute_gradient_statistics(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics to detect vanishing/exploding gradients.
    
    Returns:
        Dict with gradient statistics:
        - total_norm: Total gradient norm (before clipping)
        - max_grad: Maximum absolute gradient value
        - min_grad: Minimum absolute gradient value (excluding zeros)
        - mean_grad: Mean absolute gradient value
        - zero_grad_ratio: Ratio of parameters with zero gradients
        - layer_stats: Per-layer gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')
    grad_sum = 0.0
    param_count = 0
    zero_grad_count = 0
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad = param.grad.data
            param_norm = param_grad.norm(2).item()
            total_norm += param_norm ** 2
            
            # Per-layer statistics
            layer_max = param_grad.abs().max().item()
            layer_min = param_grad[param_grad != 0].abs().min().item() if (param_grad != 0).any() else 0.0
            layer_mean = param_grad.abs().mean().item()
            layer_zero_ratio = (param_grad == 0).float().mean().item()
            
            layer_stats[name] = {
                'norm': param_norm,
                'max': layer_max,
                'min': layer_min if layer_min != float('inf') else 0.0,
                'mean': layer_mean,
                'zero_ratio': layer_zero_ratio,
                'num_params': param.numel()
            }
            
            # Global statistics
            max_grad = max(max_grad, layer_max)
            if layer_min > 0:
                min_grad = min(min_grad, layer_min)
            grad_sum += param_grad.abs().sum().item()
            param_count += param.numel()
            zero_grad_count += (param_grad == 0).sum().item()
        else:
            layer_stats[name] = {
                'norm': 0.0,
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'zero_ratio': 1.0,
                'num_params': param.numel()
            }
            zero_grad_count += param.numel()
            param_count += param.numel()
    
    total_norm = total_norm ** (1. / 2)
    mean_grad = grad_sum / param_count if param_count > 0 else 0.0
    zero_grad_ratio = zero_grad_count / param_count if param_count > 0 else 1.0
    
    return {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'min_grad': min_grad if min_grad != float('inf') else 0.0,
        'mean_grad': mean_grad,
        'zero_grad_ratio': zero_grad_ratio,
        'layer_stats': layer_stats
    }


def diagnose_gradients(model: nn.Module, threshold: float = 1e-6):
    """
    Print detailed gradient diagnosis.
    
    Args:
        model: PyTorch model
        threshold: Threshold below which gradients are considered "vanishing"
    """
    logger.info("\n" + "="*80)
    logger.info("GRADIENT DIAGNOSIS")
    logger.info("="*80)
    
    grad_stats = compute_gradient_statistics(model)
    
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total gradient norm: {grad_stats['total_norm']:.6e}")
    logger.info(f"  Max gradient: {grad_stats['max_grad']:.6e}")
    logger.info(f"  Min gradient: {grad_stats['min_grad']:.6e}")
    logger.info(f"  Mean gradient: {grad_stats['mean_grad']:.6e}")
    logger.info(f"  Zero gradient ratio: {grad_stats['zero_grad_ratio']:.3f} ({grad_stats['zero_grad_ratio']*100:.1f}%)")
    
    if grad_stats['total_norm'] < threshold:
        logger.warning(f"\n⚠️  WARNING: VANISHING GRADIENTS DETECTED!")
        logger.warning(f"   Total norm ({grad_stats['total_norm']:.2e}) is below threshold ({threshold:.2e})")
    
    logger.info(f"\nPer-Layer Statistics (top 10 layers by gradient norm):")
    logger.info(f"{'Layer':<50} {'Norm':<12} {'Max':<12} {'Min':<12} {'Zero%':<8}")
    logger.info("-"*80)
    
    sorted_layers = sorted(grad_stats['layer_stats'].items(), 
                          key=lambda x: x[1]['norm'], reverse=True)
    
    for name, stats in sorted_layers[:10]:  # Show top 10 layers
        logger.info(f"{name[:48]:<50} {stats['norm']:<12.6e} {stats['max']:<12.6e} "
              f"{stats['min']:<12.6e} {stats['zero_ratio']*100:<7.1f}%")
    
    # Check LSTM layers specifically
    logger.info(f"\nLSTM Layer Analysis:")
    lstm_layers = {k: v for k, v in grad_stats['layer_stats'].items() if 'lstm' in k.lower()}
    if lstm_layers:
        for name, stats in lstm_layers.items():
            status = "⚠️ VANISHING" if stats['norm'] < threshold else "✓ OK"
            logger.info(f"  {name}: norm={stats['norm']:.6e} {status}")
    else:
        logger.info("  No LSTM layers found in model")
    
    logger.info("="*80 + "\n")


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                gradient_clip_norm: float = 1.0,
                use_mixed_precision: bool = False,
                log_gradients: bool = True,
                log_interval: int = 5,
                label_smoothing: float = 0.0) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        log_gradients: Whether to log gradient statistics
        log_interval: Log gradients every N batches (0 = only first batch)
        label_smoothing: Amount of label smoothing to apply (0.0 = none)
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Track gradient statistics
    gradient_stats_list = []
    
    for batch_idx, batch in enumerate(dataloader):
        # Validate batch structure
        required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels']
        if not all(key in batch for key in required_keys):
            missing = [key for key in required_keys if key not in batch]
            raise KeyError(f"Batch missing required keys: {missing}")
        
        # Move batch to device
        batch['file_sequences'] = batch['file_sequences'].to(device)
        batch['frame_lengths'] = batch['frame_lengths'].to(device)
        batch['file_mask'] = batch['file_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Apply label smoothing for loss computation
        smoothed_labels = apply_label_smoothing(labels, label_smoothing)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                if 'prediction' not in outputs:
                    raise KeyError("Model output missing 'prediction' key")
                predictions = outputs['prediction'].squeeze(-1)  # Only squeeze last dimension
                loss = criterion(predictions, smoothed_labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient monitoring (before clipping)
            should_log = log_gradients and (batch_idx == 0 or (log_interval > 0 and batch_idx % log_interval == 0))
            if should_log:
                grad_stats = compute_gradient_statistics(model)
                gradient_stats_list.append(grad_stats)
                
                # Log warnings for vanishing/exploding gradients
                if grad_stats['total_norm'] < 1e-6:
                    logger.warning(f"Batch {batch_idx}: ⚠️ VANISHING GRADIENTS detected! Total norm: {grad_stats['total_norm']:.2e}")
                elif grad_stats['total_norm'] > 100:
                    logger.warning(f"Batch {batch_idx}: ⚠️ EXPLODING GRADIENTS detected! Total norm: {grad_stats['total_norm']:.2f}")
                
                if grad_stats['zero_grad_ratio'] > 0.5:
                    logger.warning(f"Batch {batch_idx}: ⚠️ {grad_stats['zero_grad_ratio']*100:.1f}% of parameters have zero gradients!")
                
                # Log detailed stats for first batch or periodically
                if batch_idx == 0:
                    logger.info(f"Gradient stats (batch {batch_idx}): "
                              f"norm={grad_stats['total_norm']:.6f}, "
                              f"max={grad_stats['max_grad']:.6f}, "
                              f"min={grad_stats['min_grad']:.6f}, "
                              f"mean={grad_stats['mean_grad']:.6f}, "
                              f"zero_ratio={grad_stats['zero_grad_ratio']:.3f}")
                elif log_interval > 0:
                    logger.debug(f"Gradient stats (batch {batch_idx}): "
                               f"norm={grad_stats['total_norm']:.6f}, "
                               f"zero_ratio={grad_stats['zero_grad_ratio']:.3f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            if 'prediction' not in outputs:
                raise KeyError("Model output missing 'prediction' key")
            predictions = outputs['prediction'].squeeze(-1)  # Only squeeze last dimension
            loss = criterion(predictions, smoothed_labels)
            
            loss.backward()
            
            # Gradient monitoring (before clipping)
            should_log = log_gradients and (batch_idx == 0 or (log_interval > 0 and batch_idx % log_interval == 0))
            if should_log:
                grad_stats = compute_gradient_statistics(model)
                gradient_stats_list.append(grad_stats)
                
                # Log warnings for vanishing/exploding gradients
                if grad_stats['total_norm'] < 1e-6:
                    logger.warning(f"Batch {batch_idx}: ⚠️ VANISHING GRADIENTS detected! Total norm: {grad_stats['total_norm']:.2e}")
                elif grad_stats['total_norm'] > 100:
                    logger.warning(f"Batch {batch_idx}: ⚠️ EXPLODING GRADIENTS detected! Total norm: {grad_stats['total_norm']:.2f}")
                
                if grad_stats['zero_grad_ratio'] > 0.5:
                    logger.warning(f"Batch {batch_idx}: ⚠️ {grad_stats['zero_grad_ratio']*100:.1f}% of parameters have zero gradients!")
                
                # Log detailed stats for first batch or periodically
                if batch_idx == 0:
                    logger.info(f"Gradient stats (batch {batch_idx}): "
                              f"norm={grad_stats['total_norm']:.6f}, "
                              f"max={grad_stats['max_grad']:.6e}, "
                              f"min={grad_stats['min_grad']:.6e}, "
                              f"mean={grad_stats['mean_grad']:.6e}, "
                              f"zero_ratio={grad_stats['zero_grad_ratio']:.3f}")
                elif log_interval > 0:
                    logger.debug(f"Gradient stats (batch {batch_idx}): "
                               f"norm={grad_stats['total_norm']:.6f}, "
                               f"zero_ratio={grad_stats['zero_grad_ratio']:.3f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics (apply sigmoid to convert logits to probabilities)
        probs = torch.sigmoid(predictions).detach().cpu().numpy()
        all_predictions.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
    
    # Check for empty dataloader
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - cannot compute metrics")
    
    # Compute metrics
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Debug: Check if predictions are varying
    if len(np.unique(all_predictions)) == 1:
        logger.warning(f"WARNING: All training predictions are identical ({all_predictions[0]:.6f}) - model may not be learning!")
    else:
        logger.debug(f"Training predictions range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}], std: {all_predictions.std():.6f}")
    
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(dataloader)
    
    # Compute average gradient statistics for the epoch
    if gradient_stats_list:
        avg_grad_norm = np.mean([s['total_norm'] for s in gradient_stats_list])
        avg_zero_ratio = np.mean([s['zero_grad_ratio'] for s in gradient_stats_list])
        metrics['avg_grad_norm'] = avg_grad_norm
        metrics['avg_zero_grad_ratio'] = avg_zero_ratio
        
        # Log summary
        logger.info(f"Epoch gradient summary: avg_norm={avg_grad_norm:.6f}, avg_zero_ratio={avg_zero_ratio:.3f}")
    
    return metrics


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Validate batch structure
            required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels']
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
            predictions = outputs['prediction'].squeeze(-1)  # Only squeeze last dimension
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            # Apply sigmoid to convert logits to probabilities for metrics
            probs = torch.sigmoid(predictions).cpu().numpy()
            all_predictions.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    # Check for empty dataloader
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - cannot compute metrics")
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Debug: Check if predictions are varying
    if len(np.unique(all_predictions)) == 1:
        logger.warning(f"WARNING: All validation predictions are identical ({all_predictions[0]:.6f}) - model may not be learning!")
    else:
        logger.debug(f"Validation predictions range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}], std: {all_predictions.std():.6f}")
    
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int = 100,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-5,
                early_stopping_patience: int = 10,
                early_stopping_min_delta: float = 0.001,
                gradient_clip_norm: float = 1.0,
                use_mixed_precision: bool = False,
                use_weighted_bce: bool = True,
                save_dir: Optional[Path] = None,
                seed: int = 42,
                log_gradients: bool = True,
                gradient_log_interval: int = 5,
                diagnose_gradients_epochs: Optional[list] = None,
                use_lr_scheduler: bool = True,
                lr_scheduler_factor: float = 0.5,
                lr_scheduler_patience: int = 7,
                lr_scheduler_min_lr: float = 1e-7,
                label_smoothing: float = 0.0) -> Dict:
    """
    Train model with early stopping, LR scheduling, and checkpointing.
    
    Args:
        log_gradients: Whether to log gradient statistics during training
        gradient_log_interval: Log gradients every N batches (0 = only first batch)
        diagnose_gradients_epochs: List of epoch numbers to run detailed gradient diagnosis
                                  (e.g., [0, 10, 50] to diagnose at epochs 0, 10, and 50)
        use_lr_scheduler: Whether to use learning rate scheduler
        lr_scheduler_factor: Factor to reduce LR by when plateau
        lr_scheduler_patience: Epochs to wait before reducing LR
        lr_scheduler_min_lr: Minimum learning rate
        label_smoothing: Label smoothing factor (0 = no smoothing, 0.1 = smooth to 0.1/0.9)
    
    Returns:
        Dict with training history and best model path
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Check for empty dataloaders
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty")
    if len(val_loader) == 0:
        raise ValueError("Validation dataloader is empty")
    
    # Loss function
    if use_weighted_bce:
        # Compute pos_weight from training labels
        train_labels = []
        for batch in train_loader:
            train_labels.append(batch['labels'].numpy())
        if len(train_labels) == 0:
            raise ValueError("No training labels found")
        train_labels = np.concatenate(train_labels)
        pos_weight = compute_pos_weight(torch.FloatTensor(train_labels))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        logger.info(f"Using weighted BCE with pos_weight={pos_weight:.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (ReduceLROnPlateau based on validation AUC)
    scheduler = None
    if use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize validation AUC
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr,
            verbose=True
        )
        logger.info(f"Using LR scheduler: ReduceLROnPlateau(factor={lr_scheduler_factor}, "
                   f"patience={lr_scheduler_patience}, min_lr={lr_scheduler_min_lr})")
    
    # Early stopping (use validation AUC for consistency with model saving)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode='auc'  # Use AUC instead of loss for consistency
    )
    
    # Training history
    history = {
        'train': {'loss': [], 'auc': [], 'accuracy': [], 'f1': []},
        'val': {'loss': [], 'auc': [], 'accuracy': [], 'f1': []},
        'learning_rate': []  # Track LR changes
    }
    
    # Add gradient history if logging gradients
    if log_gradients:
        history['train']['avg_grad_norm'] = []
        history['train']['avg_zero_grad_ratio'] = []
    
    # Log label smoothing if enabled
    if label_smoothing > 0:
        logger.info(f"Using label smoothing: {label_smoothing} (labels: 0->{label_smoothing}, 1->{1-label_smoothing})")
    
    best_val_auc = 0.0
    best_model_path = None
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    if log_gradients:
        logger.info(f"Gradient monitoring enabled (log interval: {gradient_log_interval} batches)")
    
    # Initialize diagnose_gradients_epochs if None
    if diagnose_gradients_epochs is None:
        diagnose_gradients_epochs = [0]  # Diagnose at epoch 0 by default
    
    # Initialize scaler for mixed precision training (if needed)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    for epoch in range(num_epochs):
        # Run detailed gradient diagnosis if requested
        if epoch in diagnose_gradients_epochs:
            # Need to do a forward-backward pass to get gradients
            # Use only 1 sample to save GPU memory
            model.train()
            sample_batch = next(iter(train_loader))
            
            # Reduce batch to single sample for memory efficiency
            # Get first subject's file count
            first_subject_file_count = sample_batch['subject_file_counts'][0]
            
            # Slice to get only first subject's data
            sample_batch['file_sequences'] = sample_batch['file_sequences'][:first_subject_file_count]
            sample_batch['frame_lengths'] = sample_batch['frame_lengths'][:first_subject_file_count]
            # Resize file_mask to match the new max_files (first subject only, with correct width)
            sample_batch['file_mask'] = sample_batch['file_mask'][:1, :first_subject_file_count]
            sample_batch['subject_file_counts'] = [first_subject_file_count]  # Only first subject
            sample_batch['labels'] = sample_batch['labels'][:1]  # Only first subject
            # file_to_subject should all be 0 since we only have one subject (index 0)
            if 'file_to_subject' in sample_batch:
                sample_batch['file_to_subject'] = [0] * first_subject_file_count
            if 'subject_ids' in sample_batch:
                sample_batch['subject_ids'] = sample_batch['subject_ids'][:1]
            if 'max_files' in sample_batch:
                sample_batch['max_files'] = first_subject_file_count
            
            sample_batch['file_sequences'] = sample_batch['file_sequences'].to(device)
            sample_batch['frame_lengths'] = sample_batch['frame_lengths'].to(device)
            sample_batch['file_mask'] = sample_batch['file_mask'].to(device)
            sample_labels = sample_batch['labels'].to(device)
            
            # Clear GPU cache before diagnosis to free memory
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(sample_batch)
                    predictions = outputs['prediction'].squeeze(-1)
                    loss = criterion(predictions, sample_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                outputs = model(sample_batch)
                predictions = outputs['prediction'].squeeze(-1)
                loss = criterion(predictions, sample_labels)
                loss.backward()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Detailed Gradient Diagnosis - Epoch {epoch+1}")
            logger.info(f"{'='*80}")
            diagnose_gradients(model, threshold=1e-6)
            
            optimizer.zero_grad()  # Clear gradients after diagnosis
            torch.cuda.empty_cache()  # Clear cache after diagnosis
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip_norm, use_mixed_precision,
            log_gradients=log_gradients,
            log_interval=gradient_log_interval,
            label_smoothing=label_smoothing
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f}"
        )
        
        # Update history
        for key in ['loss', 'auc', 'accuracy', 'f1']:
            history['train'][key].append(train_metrics[key])
            history['val'][key].append(val_metrics[key])
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Update gradient history if available
        if log_gradients and 'avg_grad_norm' in train_metrics:
            history['train']['avg_grad_norm'].append(train_metrics['avg_grad_norm'])
            history['train']['avg_zero_grad_ratio'].append(train_metrics['avg_zero_grad_ratio'])
        
        # Step the learning rate scheduler based on validation AUC
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['auc'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = save_dir / 'model_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'history': history
                }, best_model_path)
                logger.info(f"Saved best model (AUC={best_val_auc:.4f}) to {best_model_path}")
        
        # Early stopping (use validation AUC for consistency with model saving)
        if early_stopping(val_metrics['auc'], epoch):
            logger.info(f"Early stopping at epoch {epoch+1} (best epoch: {early_stopping.best_epoch+1}, best AUC: {early_stopping.best_metric:.4f})")
            break
    
    # Save training log
    if save_dir is not None:
        log_path = save_dir / 'train_log.json'
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training log to {log_path}")
    
    return {
        'history': history,
        'best_val_auc': best_val_auc,
        'best_model_path': best_model_path
    }


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Training module loaded. Import and use train_model() function.")

