"""
Sanity check script to verify model components work correctly.

Creates toy data and runs a single forward pass through the complete model.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collate import collate_fn
from models.hierarchical_lstm import HierarchicalLSTMDepression


def create_toy_batch(batch_size=2, n_files_per_subject=[3, 2], seq_lengths=[100, 80, 90, 70, 120], n_features=62):
    """Create a toy batch for testing."""
    mock_batch = []
    
    file_idx = 0
    for subject_idx in range(batch_size):
        n_files = n_files_per_subject[subject_idx]
        files = []
        file_lengths = []
        
        for _ in range(n_files):
            length = seq_lengths[file_idx % len(seq_lengths)]
            files.append(np.random.randn(length, n_features).astype(np.float32))
            file_lengths.append(length)
            file_idx += 1
        
        mock_batch.append({
            'files': files,
            'file_lengths': file_lengths,
            'label': np.random.randint(0, 2),
            'subject_id': f'test_subject_{subject_idx}'
        })
    
    return collate_fn(mock_batch)


def test_file_lstm():
    """Test file-level LSTM."""
    print("Testing FileLevelLSTM...")
    from models.file_lstm import FileLevelLSTM
    
    model = FileLevelLSTM(input_dim=62, hidden_dim=128, output_dim=64)
    
    batch_size = 4
    max_length = 100
    sequences = torch.randn(batch_size, max_length, 62)
    lengths = torch.tensor([100, 80, 90, 70])
    
    with torch.no_grad():
        embeddings = model(sequences, lengths)
    
    assert embeddings.shape == (batch_size, 64), f"Expected (4, 64), got {embeddings.shape}"
    print(f"  ✓ File LSTM output shape: {embeddings.shape}")


def test_aggregation():
    """Test aggregation modules."""
    print("Testing Aggregation modules...")
    from models.aggregation import AttentionAggregation, SimpleAggregation
    
    batch_size = 2
    max_files = 5
    embedding_dim = 64
    
    file_embeddings = torch.randn(batch_size, max_files, embedding_dim)
    file_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]).bool()
    
    # Test attention
    attn = AttentionAggregation(embedding_dim=embedding_dim)
    subject_emb, attn_weights = attn(file_embeddings, file_mask)
    assert subject_emb.shape == (batch_size, embedding_dim)
    assert attn_weights.shape == (batch_size, max_files)
    print(f"  ✓ Attention output shape: {subject_emb.shape}")
    
    # Test simple aggregation
    simple = SimpleAggregation(embedding_dim=embedding_dim, mode='mean_max')
    subject_emb = simple(file_embeddings, file_mask)
    assert subject_emb.shape == (batch_size, embedding_dim * 2)
    print(f"  ✓ Simple aggregation output shape: {subject_emb.shape}")


def test_classifier():
    """Test classification head."""
    print("Testing ClassificationHead...")
    from models.classifier import ClassificationHead
    
    model = ClassificationHead(input_dim=64)
    x = torch.randn(4, 64)
    
    with torch.no_grad():
        predictions = model(x)
    
    assert predictions.shape == (4, 1)
    assert (predictions >= 0).all() and (predictions <= 1).all()
    print(f"  ✓ Classifier output shape: {predictions.shape}, range: [{predictions.min():.3f}, {predictions.max():.3f}]")


def test_full_model():
    """Test complete hierarchical model."""
    print("Testing HierarchicalLSTMDepression...")
    
    # Create toy batch
    batch = create_toy_batch(batch_size=2, n_files_per_subject=[3, 2])
    
    print(f"  Batch shapes:")
    print(f"    file_sequences: {batch['file_sequences'].shape}")
    print(f"    file_mask: {batch['file_mask'].shape}")
    print(f"    labels: {batch['labels'].shape}")
    
    # Create model
    model = HierarchicalLSTMDepression(
        n_features=62,
        lstm_hidden=128,
        file_embedding=64,
        aggregation_method='attention'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    print(f"  Model outputs:")
    print(f"    prediction: {outputs['prediction'].shape}")
    print(f"    subject_embedding: {outputs['subject_embedding'].shape}")
    if outputs['attention_weights'] is not None:
        print(f"    attention_weights: {outputs['attention_weights'].shape}")
    
    assert outputs['prediction'].shape == (2, 1)
    assert outputs['subject_embedding'].shape == (2, 64)
    print("  ✓ Full model forward pass successful!")


def test_different_aggregations():
    """Test model with different aggregation methods."""
    print("Testing different aggregation methods...")
    
    batch = create_toy_batch(batch_size=2)
    
    for method in ['attention', 'mean', 'max', 'mean_max']:
        model = HierarchicalLSTMDepression(
            n_features=62,
            aggregation_method=method
        )
        
        with torch.no_grad():
            outputs = model(batch)
        
        assert outputs['prediction'].shape == (2, 1)
        print(f"  ✓ {method} aggregation works")


def main():
    """Run all sanity checks."""
    print("="*80)
    print("Running Sanity Checks")
    print("="*80)
    print()
    
    try:
        test_file_lstm()
        print()
        test_aggregation()
        print()
        test_classifier()
        print()
        test_full_model()
        print()
        test_different_aggregations()
        print()
        
        print("="*80)
        print("✓ All sanity checks passed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

