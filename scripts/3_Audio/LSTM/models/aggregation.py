"""
Subject-level aggregation modules for combining file embeddings.

Implements attention-based (recommended), LSTM-based, and simple aggregation methods.
See LSTM_ARCHITECTURE_GUIDE.md section 4.2 for details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionAggregation(nn.Module):
    """
    Attention-based aggregation of file embeddings.
    
    Learns importance weights for each file and computes weighted sum.
    This is the recommended method (see LSTM_ARCHITECTURE_GUIDE.md section 4.2).
    
    Architecture:
        - Linear layer: embedding_dim -> 32
        - Tanh activation
        - Linear layer: 32 -> 1 (attention score)
        - Softmax over files
        - Weighted sum of embeddings
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                file_embeddings: torch.Tensor,
                file_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted aggregation of file embeddings.
        
        Args:
            file_embeddings: (batch_size, max_files, embedding_dim)
            file_mask: (batch_size, max_files) - 1 for valid files, 0 for padding
        
        Returns:
            subject_embeddings: (batch_size, embedding_dim)
            attention_weights: (batch_size, max_files) - learned importance weights
        """
        # Compute attention scores
        scores = self.attention(file_embeddings).squeeze(-1)  # (batch, max_files)
        
        # Mask out padding files (set to large negative value)
        if file_mask is not None:
            scores = scores.masked_fill(~file_mask, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, max_files)
        
        # Weighted sum: (batch, 1, max_files) @ (batch, max_files, embedding_dim)
        subject_emb = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, max_files)
            file_embeddings  # (batch, max_files, embedding_dim)
        ).squeeze(1)  # (batch, embedding_dim)
        
        return subject_emb, attention_weights


class LSTMAggregation(nn.Module):
    """
    LSTM-based aggregation (useful if file order matters).
    
    Processes file embeddings sequentially with LSTM and returns final hidden state.
    See LSTM_ARCHITECTURE_GUIDE.md section 4.2 Option B.
    """
    
    def __init__(self, file_embedding_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            file_embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.hidden_dim = hidden_dim
    
    def forward(self,
                file_embeddings: torch.Tensor,
                file_lengths: torch.Tensor) -> torch.Tensor:
        """
        Aggregate file embeddings using LSTM.
        
        Args:
            file_embeddings: (batch_size, max_files, embedding_dim)
            file_lengths: (batch_size,) - number of files per subject
        
        Returns:
            subject_embeddings: (batch_size, hidden_dim)
        """
        # Pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            file_embeddings,
            file_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward
        _, (hidden, _) = self.lstm(packed)
        # hidden shape: (1, batch, hidden_dim)
        
        return hidden.squeeze(0)  # (batch, hidden_dim)


class SimpleAggregation(nn.Module):
    """
    Simple aggregation methods: mean, max, or concatenated mean+max.
    
    See LSTM_ARCHITECTURE_GUIDE.md section 4.2 Option C.
    """
    
    def __init__(self, embedding_dim: int = 64, mode: str = 'mean_max'):
        """
        Args:
            embedding_dim: Input embedding dimension
            mode: 'mean', 'max', or 'mean_max' (concatenate both)
        """
        super().__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim if mode != 'mean_max' else embedding_dim * 2
    
    def forward(self,
                file_embeddings: torch.Tensor,
                file_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate file embeddings using simple pooling.
        
        Args:
            file_embeddings: (batch_size, max_files, embedding_dim)
            file_mask: (batch_size, max_files) - 1 for valid files, 0 for padding
        
        Returns:
            subject_embeddings: (batch_size, output_dim)
        """
        if file_mask is not None:
            # Mask out padding files by setting to zero
            masked_embeddings = file_embeddings * file_mask.unsqueeze(-1)
            # Count valid files per subject
            n_files = file_mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
            n_files = torch.clamp(n_files, min=1.0)  # Avoid division by zero
        else:
            masked_embeddings = file_embeddings
            n_files = file_embeddings.shape[1]
        
        if self.mode == 'mean':
            subject_emb = masked_embeddings.sum(dim=1) / n_files
        elif self.mode == 'max':
            subject_emb = masked_embeddings.max(dim=1)[0]
        elif self.mode == 'mean_max':
            mean_emb = masked_embeddings.sum(dim=1) / n_files
            max_emb = masked_embeddings.max(dim=1)[0]
            subject_emb = torch.cat([mean_emb, max_emb], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return subject_emb


if __name__ == '__main__':
    # Unit tests
    print("Testing aggregation modules...")
    
    batch_size = 2
    max_files = 5
    embedding_dim = 64
    
    # Create dummy data
    file_embeddings = torch.randn(batch_size, max_files, embedding_dim)
    file_mask = torch.tensor([
        [1, 1, 1, 0, 0],  # Subject 1 has 3 files
        [1, 1, 0, 0, 0]   # Subject 2 has 2 files
    ]).bool()
    
    # Test AttentionAggregation
    print("\n1. Testing AttentionAggregation...")
    attn_agg = AttentionAggregation(embedding_dim=embedding_dim)
    subject_emb, attn_weights = attn_agg(file_embeddings, file_mask)
    print(f"   Input: {file_embeddings.shape}")
    print(f"   Output: {subject_emb.shape}")
    print(f"   Attention weights: {attn_weights.shape}")
    print(f"   Weights sum (should be 1.0): {attn_weights.sum(dim=1)}")
    
    # Test LSTMAggregation
    print("\n2. Testing LSTMAggregation...")
    lstm_agg = LSTMAggregation(file_embedding_dim=embedding_dim, hidden_dim=32)
    file_lengths = torch.tensor([3, 2])
    subject_emb = lstm_agg(file_embeddings, file_lengths)
    print(f"   Output: {subject_emb.shape}")
    
    # Test SimpleAggregation
    print("\n3. Testing SimpleAggregation (mean_max)...")
    simple_agg = SimpleAggregation(embedding_dim=embedding_dim, mode='mean_max')
    subject_emb = simple_agg(file_embeddings, file_mask)
    print(f"   Output: {subject_emb.shape} (should be {embedding_dim * 2})")
    
    print("\n✓ All tests passed!")

