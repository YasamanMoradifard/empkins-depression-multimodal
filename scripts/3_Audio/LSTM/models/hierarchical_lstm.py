"""
Complete hierarchical LSTM model for depression detection.

Integrates file-level LSTM, subject-level aggregation, and classification head.
See LSTM_ARCHITECTURE_GUIDE.md section 5 for complete model integration.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .file_lstm import FileLevelLSTM, SimpleFileLSTM
from .aggregation import AttentionAggregation, LSTMAggregation, SimpleAggregation
from .classifier import ClassificationHead


class HierarchicalLSTMDepression(nn.Module):
    """
    Complete hierarchical LSTM model for depression detection.
    
    Architecture:
        1. File-level LSTM: Encodes each file independently
        2. Subject-level aggregation: Combines file embeddings (attention/LSTM/mean)
        3. Classification head: Maps subject embedding to depression probability
    
    See LSTM_ARCHITECTURE_GUIDE.md for full architectural details.
    """
    
    def __init__(self,
                 n_features: int = 62,
                 lstm_hidden: int = 128,
                 file_embedding: int = 64,
                 subject_embedding: int = 64,
                 dropout: float = 0.3,
                 aggregation_method: str = 'attention',
                 classifier_hidden: int = 32,
                 classifier_hidden2: int = 16,
                 classifier_dropout: float = 0.5,
                 use_layer_norm: bool = False,
                 use_simple_lstm: bool = False):
        """
        Initialize hierarchical LSTM model.
        
        Args:
            n_features: Number of input features per frame (e.g., 62 for GeMAPS)
            lstm_hidden: Hidden dimension for file-level LSTM (default: 128)
            file_embedding: File embedding dimension (default: 64)
            subject_embedding: Subject embedding dimension (default: 64)
            dropout: Dropout rate for LSTM layers (default: 0.3)
            aggregation_method: 'attention', 'lstm', 'mean', 'max', or 'mean_max'
            classifier_hidden: First hidden layer dimension (default: 32)
            classifier_hidden2: Second hidden layer dimension (default: 16)
            classifier_dropout: Dropout rate for classifier (default: 0.5)
            use_layer_norm: Whether to use layer normalization in file LSTM
            use_simple_lstm: Whether to use simplified single-layer LSTM (recommended for small datasets)
        """
        super().__init__()
        
        # File-level encoder - use simple or full LSTM based on config
        if use_simple_lstm:
            self.file_encoder = SimpleFileLSTM(
                input_dim=n_features,
                hidden_dim=lstm_hidden,
                output_dim=file_embedding,
                dropout=dropout
            )
        else:
            self.file_encoder = FileLevelLSTM(
                input_dim=n_features,
                hidden_dim=lstm_hidden,
                output_dim=file_embedding,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
        
        # Subject-level aggregation
        if aggregation_method == 'attention':
            self.aggregation = AttentionAggregation(embedding_dim=file_embedding)
            aggregation_output_dim = file_embedding
        elif aggregation_method == 'lstm':
            self.aggregation = LSTMAggregation(
                file_embedding_dim=file_embedding,
                hidden_dim=subject_embedding
            )
            aggregation_output_dim = subject_embedding
        elif aggregation_method in ['mean', 'max']:
            self.aggregation = SimpleAggregation(
                embedding_dim=file_embedding,
                mode=aggregation_method
            )
            aggregation_output_dim = file_embedding
        elif aggregation_method == 'mean_max':
            self.aggregation = SimpleAggregation(
                embedding_dim=file_embedding,
                mode='mean_max'
            )
            aggregation_output_dim = file_embedding * 2
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        self.aggregation_method = aggregation_method
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=aggregation_output_dim,
            hidden_dim=classifier_hidden,
            hidden_dim2=classifier_hidden2,
            dropout=classifier_dropout
        )
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through complete model.
        
        Args:
            batch: Dict from collate_fn containing:
                - 'file_sequences': (total_files, max_timesteps, n_features)
                - 'frame_lengths': (total_files,)
                - 'file_mask': (batch_size, max_files)
                - 'subject_file_counts': List[int]
                - 'file_to_subject': List[int]
                - 'labels': (batch_size,)
                - 'subject_ids': List[str]
                - 'max_files': int
        
        Returns:
            Dict with:
                - 'prediction': (batch_size, 1) - depression probabilities
                - 'subject_embedding': (batch_size, embedding_dim)
                - 'attention_weights': (batch_size, max_files) or None
        """
        file_sequences = batch['file_sequences']
        frame_lengths = batch['frame_lengths']
        file_mask = batch['file_mask']
        file_to_subject = batch['file_to_subject']
        batch_size = file_mask.shape[0]
        max_files = batch['max_files']
        
        # Step 1: Encode all files
        file_embeddings = self.file_encoder(file_sequences, frame_lengths)
        # Shape: (total_files, file_embedding)
        
        # Step 2: Group file embeddings by subject
        # Reshape to (batch_size, max_files, file_embedding)
        file_emb_grouped = torch.zeros(
            batch_size, max_files, file_embeddings.shape[-1],
            device=file_embeddings.device
        )
        
        file_idx = 0
        for subject_idx in range(batch_size):
            n_files = batch['subject_file_counts'][subject_idx]
            for file_pos in range(n_files):
                file_emb_grouped[subject_idx, file_pos] = file_embeddings[file_idx]
                file_idx += 1
        
        # Step 3: Aggregate file embeddings to subject embeddings
        if self.aggregation_method == 'attention':
            subject_emb, attention_weights = self.aggregation(
                file_emb_grouped, file_mask
            )
        elif self.aggregation_method == 'lstm':
            file_lengths_tensor = torch.tensor(
                batch['subject_file_counts'],
                device=file_emb_grouped.device
            )
            subject_emb = self.aggregation(file_emb_grouped, file_lengths_tensor)
            attention_weights = None
        else:  # Simple aggregation
            subject_emb = self.aggregation(file_emb_grouped, file_mask)
            attention_weights = None
        
        # Step 4: Classify
        prediction = self.classifier(subject_emb)
        
        return {
            'prediction': prediction,
            'subject_embedding': subject_emb,
            'attention_weights': attention_weights
        }


if __name__ == '__main__':
    # Unit test
    print("Testing HierarchicalLSTMDepression...")
    
    from data.collate import collate_fn
    
    # Create dummy batch
    import numpy as np
    mock_batch = [
        {
            'files': [
                np.random.randn(100, 62).astype(np.float32),
                np.random.randn(80, 62).astype(np.float32),
            ],
            'file_lengths': [100, 80],
            'label': 1,
            'subject_id': 'subj_1'
        },
        {
            'files': [
                np.random.randn(120, 62).astype(np.float32),
            ],
            'file_lengths': [120],
            'label': 0,
            'subject_id': 'subj_2'
        }
    ]
    
    batch = collate_fn(mock_batch)
    
    # Create model
    model = HierarchicalLSTMDepression(
        n_features=62,
        aggregation_method='attention'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    print(f"\nBatch info:")
    print(f"  file_sequences: {batch['file_sequences'].shape}")
    print(f"  file_mask: {batch['file_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    print(f"\nModel outputs:")
    print(f"  prediction: {outputs['prediction'].shape}")
    print(f"  subject_embedding: {outputs['subject_embedding'].shape}")
    if outputs['attention_weights'] is not None:
        print(f"  attention_weights: {outputs['attention_weights'].shape}")
    
    print("\n✓ Test passed!")

