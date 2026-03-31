"""
Classification head for mapping subject embeddings to depression probability.

Implements dense layers with dropout as described in LSTM_ARCHITECTURE_GUIDE.md section 4.3.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head with dense layers and dropout.
    
    Architecture:
        - Dense(32) + ReLU + Dropout(0.5)
        - Dense(16) + ReLU + Dropout(0.5)
        - Dense(1) (outputs logits, no sigmoid - BCEWithLogitsLoss handles sigmoid)
    
    Args:
        input_dim: Input embedding dimension (subject embedding size)
        hidden_dim: First hidden layer dimension (default: 32)
        hidden_dim2: Second hidden layer dimension (default: 16)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 32,
                 hidden_dim2: int = 16,
                 dropout: float = 0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
            # Note: No sigmoid here - BCEWithLogitsLoss expects logits and applies sigmoid internally
        )
        
        # Initialize final layer with small weights to prevent constant outputs
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.1)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Subject embeddings (batch_size, input_dim)
        
        Returns:
            Predictions (batch_size, 1) - logits (apply sigmoid for probabilities)
        """
        return self.classifier(x)


if __name__ == '__main__':
    # Unit test
    print("Testing ClassificationHead...")
    
    model = ClassificationHead(input_dim=64, hidden_dim=32, hidden_dim2=16)
    
    batch_size = 4
    x = torch.randn(batch_size, 64)
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Expected: logits (apply sigmoid for probabilities)")
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(predictions)
    print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    assert predictions.shape == (batch_size, 1), "Output shape mismatch!"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities not in [0, 1]!"
    print("\n✓ Test passed!")

