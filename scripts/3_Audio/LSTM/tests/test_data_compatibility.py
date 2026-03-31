"""
Test data format compatibility between data loading and training/evaluation scripts.

Verifies that:
1. dataset.py produces correct format
2. collate.py transforms it correctly
3. train.py can consume the batches
4. evaluate.py can consume the batches
5. model can process the batches
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SubjectDataset
from data.collate import collate_fn
from models.hierarchical_lstm import HierarchicalLSTMDepression
from torch.utils.data import DataLoader


def test_dataset_output():
    """Test that dataset.py produces correct format."""
    print("="*80)
    print("Test 1: Dataset Output Format")
    print("="*80)
    
    # Create mock subjects
    mock_subjects = [
        {
            'subject_id': 'test_001',
            'file_paths': ['/fake/path/file1.csv', '/fake/path/file2.csv'],
            'label': 1
        },
        {
            'subject_id': 'test_002',
            'file_paths': ['/fake/path/file3.csv'],
            'label': 0
        }
    ]
    
    dataset = SubjectDataset(mock_subjects, max_length=None, normalize=False)
    
    # Check __getitem__ output
    sample = dataset[0]
    
    required_keys = ['files', 'file_lengths', 'label', 'subject_id']
    missing = [k for k in required_keys if k not in sample]
    
    if missing:
        print(f"  ✗ Missing keys: {missing}")
        return False
    
    # Check types
    assert isinstance(sample['files'], list), "files should be list"
    assert isinstance(sample['file_lengths'], list), "file_lengths should be list"
    assert isinstance(sample['label'], (int, np.integer)), f"label should be int, got {type(sample['label'])}"
    assert isinstance(sample['subject_id'], str), "subject_id should be str"
    
    # Check file data types
    if len(sample['files']) > 0:
        assert isinstance(sample['files'][0], np.ndarray), "files should contain numpy arrays"
        assert sample['files'][0].dtype == np.float32, "files should be float32"
    
    print("  ✓ Dataset output format is correct")
    print(f"    Sample keys: {list(sample.keys())}")
    print(f"    Files: {len(sample['files'])} files")
    print(f"    Label: {sample['label']} (type: {type(sample['label'])})")
    return True


def test_collate_output():
    """Test that collate.py produces correct batch format."""
    print("\n" + "="*80)
    print("Test 2: Collate Function Output Format")
    print("="*80)
    
    # Create mock batch (simulating dataset output)
    mock_batch = [
        {
            'files': [
                np.random.randn(100, 304).astype(np.float32),
                np.random.randn(80, 304).astype(np.float32),
            ],
            'file_lengths': [100, 80],
            'label': 1,
            'subject_id': 'test_001'
        },
        {
            'files': [
                np.random.randn(120, 304).astype(np.float32),
            ],
            'file_lengths': [120],
            'label': 0,
            'subject_id': 'test_002'
        }
    ]
    
    batch = collate_fn(mock_batch)
    
    # Required keys for training
    train_required = ['file_sequences', 'frame_lengths', 'file_mask', 'labels']
    train_missing = [k for k in train_required if k not in batch]
    
    if train_missing:
        print(f"  ✗ Missing keys for training: {train_missing}")
        return False
    
    # Required keys for evaluation
    eval_required = ['file_sequences', 'frame_lengths', 'file_mask', 'labels', 
                     'subject_ids', 'subject_file_counts']
    eval_missing = [k for k in eval_required if k not in batch]
    
    if eval_missing:
        print(f"  ✗ Missing keys for evaluation: {eval_missing}")
        return False
    
    # Required keys for model
    model_required = ['file_sequences', 'frame_lengths', 'file_mask', 
                       'subject_file_counts', 'file_to_subject', 'max_files']
    model_missing = [k for k in model_required if k not in batch]
    
    if model_missing:
        print(f"  ✗ Missing keys for model: {model_missing}")
        return False
    
    # Check shapes
    assert batch['file_sequences'].shape[0] == 3, "Should have 3 total files"
    assert batch['file_mask'].shape[0] == 2, "Should have 2 subjects"
    assert batch['labels'].shape[0] == 2, "Should have 2 labels"
    assert len(batch['subject_ids']) == 2, "Should have 2 subject IDs"
    
    print("  ✓ Collate output format is correct")
    print(f"    file_sequences shape: {batch['file_sequences'].shape}")
    print(f"    frame_lengths shape: {batch['frame_lengths'].shape}")
    print(f"    file_mask shape: {batch['file_mask'].shape}")
    print(f"    labels shape: {batch['labels'].shape}")
    print(f"    subject_ids: {batch['subject_ids']}")
    print(f"    subject_file_counts: {batch['subject_file_counts']}")
    print(f"    max_files: {batch['max_files']}")
    return True


def test_train_compatibility():
    """Test that train.py can consume batches."""
    print("\n" + "="*80)
    print("Test 3: Training Script Compatibility")
    print("="*80)
    
    # Create mock batch
    mock_batch = [
        {
            'files': [np.random.randn(100, 304).astype(np.float32)],
            'file_lengths': [100],
            'label': 1,
            'subject_id': 'test_001'
        }
    ]
    
    batch = collate_fn(mock_batch)
    
    # Check required keys (as in train.py line 92)
    required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels']
    missing = [k for k in required_keys if k not in batch]
    
    if missing:
        print(f"  ✗ Missing required keys: {missing}")
        return False
    
    # Check that tensors can be moved to device
    device = torch.device('cpu')
    try:
        batch['file_sequences'] = batch['file_sequences'].to(device)
        batch['frame_lengths'] = batch['frame_lengths'].to(device)
        batch['file_mask'] = batch['file_mask'].to(device)
        labels = batch['labels'].to(device)
    except Exception as e:
        print(f"  ✗ Error moving to device: {e}")
        return False
    
    print("  ✓ Training script can consume batches")
    print(f"    All required keys present: {required_keys}")
    print(f"    Tensors successfully moved to device")
    return True


def test_evaluate_compatibility():
    """Test that evaluate.py can consume batches."""
    print("\n" + "="*80)
    print("Test 4: Evaluation Script Compatibility")
    print("="*80)
    
    # Create mock batch
    mock_batch = [
        {
            'files': [np.random.randn(100, 304).astype(np.float32)],
            'file_lengths': [100],
            'label': 1,
            'subject_id': 'test_001'
        }
    ]
    
    batch = collate_fn(mock_batch)
    
    # Check required keys (as in evaluate.py line 59)
    required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels', 
                     'subject_ids', 'subject_file_counts']
    missing = [k for k in required_keys if k not in batch]
    
    if missing:
        print(f"  ✗ Missing required keys: {missing}")
        return False
    
    # Check that tensors can be moved to device
    device = torch.device('cpu')
    try:
        batch['file_sequences'] = batch['file_sequences'].to(device)
        batch['frame_lengths'] = batch['frame_lengths'].to(device)
        batch['file_mask'] = batch['file_mask'].to(device)
        labels = batch['labels'].to(device)
    except Exception as e:
        print(f"  ✗ Error moving to device: {e}")
        return False
    
    # Check that subject_file_counts can be accessed
    try:
        n_files = batch['subject_file_counts'][0]
        assert isinstance(n_files, int), "subject_file_counts should contain ints"
    except Exception as e:
        print(f"  ✗ Error accessing subject_file_counts: {e}")
        return False
    
    print("  ✓ Evaluation script can consume batches")
    print(f"    All required keys present: {required_keys}")
    print(f"    Tensors successfully moved to device")
    print(f"    subject_file_counts accessible: {batch['subject_file_counts']}")
    return True


def test_model_compatibility():
    """Test that model can process batches."""
    print("\n" + "="*80)
    print("Test 5: Model Compatibility")
    print("="*80)
    
    # Create mock batch
    mock_batch = [
        {
            'files': [
                np.random.randn(100, 304).astype(np.float32),
                np.random.randn(80, 304).astype(np.float32),
            ],
            'file_lengths': [100, 80],
            'label': 1,
            'subject_id': 'test_001'
        },
        {
            'files': [
                np.random.randn(120, 304).astype(np.float32),
            ],
            'file_lengths': [120],
            'label': 0,
            'subject_id': 'test_002'
        }
    ]
    
    batch = collate_fn(mock_batch)
    
    # Check required keys (as in model forward docstring)
    required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 
                     'subject_file_counts', 'file_to_subject', 'max_files']
    missing = [k for k in required_keys if k not in batch]
    
    if missing:
        print(f"  ✗ Missing required keys: {missing}")
        return False
    
    # Create model
    model = HierarchicalLSTMDepression(
        n_features=304,  # After excluding 5 metadata columns
        lstm_hidden=128,
        file_embedding=64,
        subject_embedding=64,
        aggregation_method='attention'
    )
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(batch)
    except Exception as e:
        print(f"  ✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check output format
    if 'prediction' not in outputs:
        print("  ✗ Model output missing 'prediction' key")
        return False
    
    assert outputs['prediction'].shape[0] == 2, "Should have predictions for 2 subjects"
    
    print("  ✓ Model can process batches")
    print(f"    Model output keys: {list(outputs.keys())}")
    print(f"    Prediction shape: {outputs['prediction'].shape}")
    print(f"    Subject embedding shape: {outputs['subject_embedding'].shape}")
    if outputs['attention_weights'] is not None:
        print(f"    Attention weights shape: {outputs['attention_weights'].shape}")
    return True


def test_end_to_end():
    """Test complete pipeline with DataLoader."""
    print("\n" + "="*80)
    print("Test 6: End-to-End Pipeline")
    print("="*80)
    
    # Create mock subjects (simulating load_train_val_test output)
    mock_subjects = [
        {
            'subject_id': 'test_001',
            'file_paths': ['/fake/path/file1.csv'],
            'label': 1
        },
        {
            'subject_id': 'test_002',
            'file_paths': ['/fake/path/file2.csv'],
            'label': 0
        }
    ]
    
    # Create dataset
    dataset = SubjectDataset(mock_subjects, max_length=None, normalize=False)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get a batch
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"  ✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify batch structure
    required_keys = ['file_sequences', 'frame_lengths', 'file_mask', 'labels']
    missing = [k for k in required_keys if k not in batch]
    
    if missing:
        print(f"  ✗ Missing keys: {missing}")
        return False
    
    print("  ✓ End-to-end pipeline works")
    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Batch size: {batch['file_mask'].shape[0]}")
    return True


def main():
    """Run all compatibility tests."""
    print("\n" + "="*80)
    print("DATA FORMAT COMPATIBILITY TESTS")
    print("="*80)
    
    tests = [
        test_dataset_output,
        test_collate_output,
        test_train_compatibility,
        test_evaluate_compatibility,
        test_model_compatibility,
        test_end_to_end
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All compatibility tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

