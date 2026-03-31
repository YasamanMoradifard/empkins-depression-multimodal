"""
Custom collate function for batching variable-length sequences.

Handles padding and masking for both file-level sequences and subject-level
file groupings. See LSTM_ARCHITECTURE_GUIDE.md section 8.2 for details.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader that handles variable-length sequences.
    
    This function:
    1. Pads frame-level sequences to max length in batch (per file)
    2. Groups files by subject
    3. Pads subject-level file lists to max files per subject
    4. Creates masks for valid files and frames
    
    Args:
        batch: List of dicts from SubjectDataset.__getitem__()
            Each dict has 'files' (List[np.ndarray]), 'file_lengths', 'label', 'subject_id'
    
    Returns:
        Dict with:
            - 'file_sequences': Tensor (total_files_in_batch, max_timesteps, n_features)
            - 'frame_lengths': Tensor (total_files_in_batch,) - actual length of each file
            - 'file_embeddings_grouped': Tensor (batch_size, max_files, embedding_dim)
                (will be filled after file-level encoding)
            - 'file_mask': Tensor (batch_size, max_files) - 1 for valid files, 0 for padding
            - 'subject_file_counts': List[int] - number of files per subject
            - 'labels': Tensor (batch_size,) - subject labels
            - 'subject_ids': List[str] - subject identifiers
    """
    batch_size = len(batch)
    
    # Step 1: Flatten all files from all subjects
    all_files = []
    all_lengths = []
    subject_file_counts = []
    labels = []
    subject_ids = []
    
    for sample in batch:
        n_files = len(sample['files'])
        subject_file_counts.append(n_files)
        labels.append(sample['label'])
        subject_ids.append(sample['subject_id'])
        
        for file_data, length in zip(sample['files'], sample['file_lengths']):
            all_files.append(torch.FloatTensor(file_data))
            all_lengths.append(length)
    
    # Step 2: Pad all file sequences to max length in batch
    if not all_files:
        raise ValueError("Empty batch - no files found")
    
    # Validate feature count consistency before padding
    if all_files:
        feature_counts = [f.shape[1] for f in all_files]
        unique_counts = set(feature_counts)
        if len(unique_counts) > 1:
            # Find which files have different counts
            count_to_files = {}
            for idx, count in enumerate(feature_counts):
                if count not in count_to_files:
                    count_to_files[count] = []
                count_to_files[count].append(idx)
            
            error_msg = (
                f"\n{'='*80}\n"
                f"FEATURE COUNT MISMATCH DETECTED!\n"
                f"{'='*80}\n"
                f"Found {len(unique_counts)} different feature counts: {sorted(unique_counts)}\n\n"
            )
            
            for count, file_indices in count_to_files.items():
                error_msg += f"  {count} features: {len(file_indices)} file(s)\n"
                if len(file_indices) <= 5:
                    for fidx in file_indices:
                        # Try to get subject info if available
                        subject_idx = 0
                        file_idx_in_subject = fidx
                        for sidx, n_files in enumerate(subject_file_counts):
                            if file_idx_in_subject < n_files:
                                subject_idx = sidx
                                break
                            file_idx_in_subject -= n_files
                        error_msg += f"    - File index {fidx} (Subject: {subject_ids[subject_idx] if subject_ids else 'unknown'})\n"
            
            error_msg += (
                f"\nThis usually means CSV files have inconsistent column counts.\n"
                f"Please ensure all CSV files have exactly the same feature columns.\n"
                f"{'='*80}\n"
            )
            
            raise RuntimeError(error_msg)
    
    # Find max sequence length
    max_timesteps = max(all_lengths) if all_lengths else 1
    
    # Pad sequences
    padded_files = nn.utils.rnn.pad_sequence(
        all_files, 
        batch_first=True, 
        padding_value=0.0
    )
    # Shape: (total_files, max_timesteps, n_features)
    
    frame_lengths = torch.LongTensor(all_lengths)
    # Shape: (total_files,)
    
    # Step 3: Group files back by subject and pad to max files per subject
    max_files = max(subject_file_counts) if subject_file_counts else 1
    
    # We'll create a placeholder for file embeddings (will be filled after encoding)
    # For now, we just need the structure
    n_features = padded_files.shape[-1]
    
    # Create file mask: 1 for valid files, 0 for padding
    file_mask = torch.zeros(batch_size, max_files, dtype=torch.bool)
    
    for i, n_files_subject in enumerate(subject_file_counts):
        file_mask[i, :n_files_subject] = 1
    
    # Create indices to group files by subject (for later use)
    file_to_subject = []
    for i, n_files_subject in enumerate(subject_file_counts):
        for _ in range(n_files_subject):
            file_to_subject.append(i)
    
    return {
        'file_sequences': padded_files,  # (total_files, max_timesteps, n_features)
        'frame_lengths': frame_lengths,  # (total_files,)
        'file_mask': file_mask,  # (batch_size, max_files)
        'subject_file_counts': subject_file_counts,  # List[int]
        'file_to_subject': file_to_subject,  # List[int] - maps file idx to subject idx
        'labels': torch.FloatTensor(labels),  # (batch_size,)
        'subject_ids': subject_ids,  # List[str]
        'max_files': max_files,  # int
    }


def group_file_embeddings(file_embeddings: torch.Tensor, 
                         file_to_subject: List[int],
                         max_files: int,
                         batch_size: int) -> torch.Tensor:
    """
    Group file embeddings back by subject.
    
    After file-level encoding, we have embeddings for each file. This function
    groups them back by subject and pads to max_files.
    
    Args:
        file_embeddings: Tensor (total_files, embedding_dim)
        file_to_subject: List mapping file index to subject index
        max_files: Maximum number of files per subject
        batch_size: Number of subjects in batch
    
    Returns:
        Tensor (batch_size, max_files, embedding_dim)
    """
    embedding_dim = file_embeddings.shape[-1]
    grouped = torch.zeros(batch_size, max_files, embedding_dim, 
                         device=file_embeddings.device)
    
    for file_idx, subject_idx in enumerate(file_to_subject):
        # Find position within subject's file list
        file_pos = sum(1 for i, s in enumerate(file_to_subject[:file_idx+1]) if s == subject_idx) - 1
        grouped[subject_idx, file_pos] = file_embeddings[file_idx]
    
    return grouped


if __name__ == '__main__':
    # Example usage
    print("Testing collate function...")
    
    # Mock batch
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
    
    collated = collate_fn(mock_batch)
    
    print(f"\nCollated batch shapes:")
    print(f"  file_sequences: {collated['file_sequences'].shape}")
    print(f"  frame_lengths: {collated['frame_lengths'].shape}")
    print(f"  file_mask: {collated['file_mask'].shape}")
    print(f"  labels: {collated['labels'].shape}")
    print(f"  subject_file_counts: {collated['subject_file_counts']}")
    print(f"  max_files: {collated['max_files']}")

