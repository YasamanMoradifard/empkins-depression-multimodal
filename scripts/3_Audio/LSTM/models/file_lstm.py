"""
File-level LSTM encoder for processing individual audio file sequences.

Implements bidirectional LSTM layers to extract temporal patterns from each file.
See LSTM_ARCHITECTURE_GUIDE.md section 4.1 for architectural details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleFileLSTM(nn.Module):
    """
    Simplified single-layer LSTM encoder for small datasets.
    
    This is a much simpler architecture designed to prevent overfitting
    when training data is limited (~50-100 subjects).
    
    Architecture:
        - Single unidirectional LSTM layer
        - Dropout after LSTM
        - Linear projection to output embedding
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for LSTM (default: 32)
        output_dim: Output embedding dimension (default: 16)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(self,
                 input_dim: int = 188,
                 hidden_dim: int = 32,
                 output_dim: int = 16,
                 dropout: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Single unidirectional LSTM (much simpler)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0  # No dropout between layers (only 1 layer)
        )
        
        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Project to output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier for better gradients."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self,
                sequences: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simple LSTM.
        
        Args:
            sequences: Padded sequences (batch, max_timesteps, input_dim)
            lengths: Actual lengths of each sequence (batch,)
            
        Returns:
            File embeddings (batch, output_dim)
        """
        # Pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward - only need final hidden state
        _, (hidden, _) = self.lstm(packed)
        # hidden shape: (1, batch, hidden_dim)
        
        # Squeeze and apply dropout + projection
        hidden = hidden.squeeze(0)  # (batch, hidden_dim)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)  # (batch, output_dim)
        
        return output


class FileLevelLSTM(nn.Module):
    """
    File-level bidirectional LSTM encoder.
    
    Architecture:
        - Bidirectional LSTM Layer 1: hidden_dim units, returns sequences
        - Dropout
        - Bidirectional LSTM Layer 2: output_dim units, returns last state only
        - Output: File embedding of shape (batch, output_dim)
    
    Args:
        input_dim: Number of input features (e.g., 62 for GeMAPS)
        hidden_dim: Hidden dimension for first LSTM layer (default: 128)
        output_dim: Output embedding dimension (default: 64)
        dropout: Dropout rate (default: 0.3)
        use_layer_norm: Whether to use layer normalization (default: False)
    """
    
    def __init__(self,
                 input_dim: int = 62,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 dropout: float = 0.3,
                 use_layer_norm: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # First bidirectional LSTM (returns full sequence)
        self.lstm1 = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Layer normalization (optional)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2) if use_layer_norm else None
        
        # Second LSTM (returns only last hidden state)
        # Input is hidden_dim*2 because of bidirectional
        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            output_dim,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if dropout > 0 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                sequences: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through file-level LSTM.
        
        Args:
            sequences: Padded sequences (batch, max_timesteps, input_dim)
            lengths: Actual lengths of each sequence (batch,)
            
        Returns:
            File embeddings (batch, output_dim)
        """
        batch_size = sequences.shape[0]
        
        # Pack sequences to handle variable lengths efficiently
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(packed)
        # lstm1_out is PackedSequence
        
        # Unpack for layer norm (if used) or second LSTM
        lstm1_unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm1_out,
            batch_first=True
        )
        # Shape: (batch, max_timesteps, hidden_dim * 2)
        
        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            lstm1_unpacked = self.layer_norm(lstm1_unpacked)
        
        # Apply dropout
        lstm1_unpacked = self.dropout(lstm1_unpacked)
        
        # Pack again for second LSTM
        packed2 = nn.utils.rnn.pack_padded_sequence(
            lstm1_unpacked,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Second LSTM layer (returns only last hidden state)
        _, (hidden, _) = self.lstm2(packed2)
        # hidden shape: (1, batch, output_dim) for unidirectional
        
        # Squeeze to (batch, output_dim)
        file_embedding = hidden.squeeze(0)
        
        return file_embedding


if __name__ == '__main__':
    # Unit test / example
    print("Testing FileLevelLSTM...")
    
    # Create model
    model = FileLevelLSTM(
        input_dim=325,
        hidden_dim=128,
        output_dim=64,
        dropout=0.3
    )
    
    # Create dummy data
    batch_size = 4
    max_length = 100
    n_features = 325  
    
    # Variable-length sequences
    sequences = torch.randn(batch_size, max_length, n_features)
    lengths = torch.tensor([100, 80, 90, 70])  # Actual lengths
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(sequences, lengths)
    
    print(f"\nInput shape: {sequences.shape}")
    print(f"Lengths: {lengths}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: ({batch_size}, 64)")
    
    assert embeddings.shape == (batch_size, 64), "Output shape mismatch!"
    print("\n✓ Test passed!")

