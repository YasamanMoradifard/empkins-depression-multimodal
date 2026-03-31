# LSTM Architecture for Time-Series Audio-Based Depression Detection

## Overview

This document explains the recommended LSTM architecture for depression detection using time-series audio features (OpenSMILE LLDs), supporting both condition-based and phase-based experiments.

---

## 1. Data Structure & Challenge

### Current Data Format
- **Per subject**: Multiple audio files (~20-40 files depending on phase)
- **Per file**: Variable-length time series (T_i × F dimensions)
  - T_i = number of frames (varies per file, typically 100-1000 frames)
  - F = number of features (62 for GeMAPS, 63 for eGeMAPS, etc.)
- **Labels**: One label per subject (depressed/healthy)

### Key Challenge
**Multiple variable-length sequences → Single subject-level prediction**

Each subject has multiple videos/audio files, each with different durations. We need to aggregate information from all files to make one prediction per subject.

---

## 2. Recommended Architecture: Hierarchical LSTM

### Architecture Type: **Two-Level Hierarchical LSTM**

This is the most theoretically sound approach for your data structure:

```
Level 1 (File-Level):  Process each file independently
                       ↓
Level 2 (Subject-Level): Aggregate all file representations
                       ↓
Classification:        Final prediction (depressed/healthy)
```

### Why Hierarchical?
1. **Preserves temporal structure**: Each file's temporal patterns are learned
2. **Handles multiple files**: Naturally aggregates information across files
3. **Interpretable**: Can analyze which files/phases contribute most
4. **Flexible**: Can handle missing files or variable numbers of files per subject

---

## 3. Detailed Architecture Design

### 3.1 Overall Pipeline

```python
Input: List of sequences per subject
  Subject_1: [Seq_1 (T1×F), Seq_2 (T2×F), ..., Seq_N1 (TN1×F)]
  Subject_2: [Seq_1 (T1×F), Seq_2 (T2×F), ..., Seq_N2 (TN2×F)]
  ...

Step 1: FILE-LEVEL LSTM (shared weights across all files)
  For each sequence → Extract embedding
  
Step 2: SUBJECT-LEVEL AGGREGATION
  Combine all file embeddings → Subject representation
  
Step 3: CLASSIFICATION
  Subject representation → Prediction (0/1)
```

### 3.2 Layer-by-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  Shape: (batch_files, timesteps, features)                 │
│  - batch_files: number of files in batch                   │
│  - timesteps: variable length (use padding/masking)        │
│  - features: 62 (GeMAPS), 63 (eGeMAPS), etc.              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              NORMALIZATION (Optional)                       │
│  - Layer Normalization or Batch Normalization              │
│  - Helps with different feature scales                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           FILE-LEVEL LSTM (Bidirectional)                   │
│  - LSTM Layer 1: 128 units, return_sequences=True          │
│  - Dropout: 0.3                                             │
│  - LSTM Layer 2: 64 units, return_sequences=False          │
│  - Dropout: 0.3                                             │
│                                                             │
│  Output: File embedding (batch_files, 64)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         SUBJECT-LEVEL AGGREGATION                           │
│                                                             │
│  Option A: ATTENTION-BASED (Recommended)                    │
│    - Learn importance weights for each file                │
│    - α_i = softmax(W * file_embedding_i)                   │
│    - subject_emb = Σ(α_i * file_embedding_i)              │
│                                                             │
│  Option B: LSTM-BASED                                       │
│    - LSTM over file embeddings                             │
│    - Handles temporal order of sessions                    │
│    - LSTM: 32 units, return_sequences=False                │
│                                                             │
│  Option C: SIMPLE AGGREGATION                               │
│    - Mean pooling: mean(file_embeddings)                   │
│    - Max pooling: max(file_embeddings)                     │
│    - Both: concatenate([mean, max])                        │
│                                                             │
│  Output: Subject embedding (batch_subjects, embedding_dim)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            CLASSIFICATION HEAD                              │
│  - Dense Layer: 32 units, ReLU activation                  │
│  - Dropout: 0.5                                             │
│  - Dense Layer: 16 units, ReLU activation                  │
│  - Dropout: 0.5                                             │
│  - Output Layer: 1 unit, Sigmoid activation                │
│                                                             │
│  Output: Probability (depressed)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Component Explanations

### 4.1 File-Level LSTM

**Purpose**: Extract temporal patterns from each audio file

**Architecture Choices**:
- **Bidirectional LSTM**: Captures both forward and backward temporal dependencies
  - Example: Depression markers might appear at beginning (low energy) or end (voice quality degradation)
- **Two layers**: First layer extracts low-level patterns, second layer learns higher-level representations
- **Return sequences**: First layer returns full sequence (for second layer), second layer returns only final state (embedding)
- **Units**: 128 → 64 (gradually reduce dimensionality)
- **Dropout**: 0.3 to prevent overfitting

**PyTorch Code Example**:
```python
class FileLevelLSTM(nn.Module):
    def __init__(self, input_dim=62, hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, 
                             bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim*2, output_dim, batch_first=True,
                             bidirectional=False, dropout=dropout)
        
    def forward(self, x, lengths):
        # x: (batch, seq_len, features)
        # Pack sequence (handle variable lengths)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, 
                                                    batch_first=True, 
                                                    enforce_sorted=False)
        
        # LSTM 1
        lstm1_out, _ = self.lstm1(packed)
        lstm1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm1_out, 
                                                          batch_first=True)
        
        # Pack again for LSTM 2
        packed2 = nn.utils.rnn.pack_padded_sequence(lstm1_out, lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        
        # LSTM 2 - take final hidden state
        _, (hidden, _) = self.lstm2(packed2)
        
        # hidden: (1, batch, output_dim)
        return hidden.squeeze(0)  # (batch, output_dim)
```

### 4.2 Subject-Level Aggregation

**Purpose**: Combine information from all files belonging to one subject

#### **Option A: Attention-Based (RECOMMENDED)**

**Why Attention?**
- Different files may have different importance
- Emotion induction videos might be more informative than training videos
- Files where person spoke more (longer speech segments) might be more reliable
- Attention learns these importance weights automatically

**How it Works**:
```
For Subject i with N files:
  file_embeddings = [emb_1, emb_2, ..., emb_N]  # Each (64,)
  
  1. Compute attention scores:
     scores = W_attention @ file_embeddings^T  # (N,)
     
  2. Normalize with softmax:
     attention_weights = softmax(scores)  # (N,), sums to 1
     
  3. Weighted sum:
     subject_embedding = Σ(attention_weights[i] * file_embeddings[i])  # (64,)
```

**PyTorch Code**:
```python
class AttentionAggregation(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, file_embeddings, file_mask=None):
        # file_embeddings: (batch_subjects, max_files, embedding_dim)
        # file_mask: (batch_subjects, max_files) - 1 for valid files, 0 for padding
        
        # Compute attention scores
        scores = self.attention(file_embeddings).squeeze(-1)  # (batch, max_files)
        
        # Mask padding files
        if file_mask is not None:
            scores = scores.masked_fill(file_mask == 0, -1e9)
        
        # Softmax to get weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, max_files)
        
        # Weighted sum
        subject_emb = torch.bmm(attention_weights.unsqueeze(1), 
                                 file_embeddings).squeeze(1)  # (batch, embedding_dim)
        
        return subject_emb, attention_weights  # Return weights for interpretability!
```

**Advantages**:
- ✅ Interpretable: Can visualize which files are important
- ✅ Flexible: Handles variable number of files per subject
- ✅ Adaptive: Learns importance automatically
- ✅ State-of-the-art: Used in many successful papers

#### **Option B: LSTM-Based**

**When to Use**: If the ORDER of files/sessions matters
- Example: If emotion_induction_1 → training_1 → emotion_induction_2 → training_2 sequence is meaningful
- Depression progression over time within experiment

**PyTorch Code**:
```python
class LSTMAggregation(nn.Module):
    def __init__(self, file_embedding_dim=64, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(file_embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, file_embeddings, lengths):
        # file_embeddings: (batch, max_files, embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(file_embeddings, lengths,
                                                    batch_first=True, 
                                                    enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden.squeeze(0)  # (batch, hidden_dim)
```

#### **Option C: Simple Aggregation (Baseline)**

**When to Use**: As a baseline comparison

```python
# Mean pooling
subject_emb = torch.mean(file_embeddings, dim=1)  # (batch, embedding_dim)

# Max pooling
subject_emb = torch.max(file_embeddings, dim=1)[0]  # (batch, embedding_dim)

# Both (concatenate)
mean_emb = torch.mean(file_embeddings, dim=1)
max_emb = torch.max(file_embeddings, dim=1)[0]
subject_emb = torch.cat([mean_emb, max_emb], dim=1)  # (batch, embedding_dim*2)
```

### 4.3 Classification Head

**Purpose**: Map subject embedding to depression probability

```python
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x)  # (batch, 1)
```

---

## 5. Complete Model Integration

```python
class HierarchicalLSTMDepression(nn.Module):
    def __init__(self, 
                 n_features=62,        # GeMAPS features
                 lstm_hidden=128,      # File-level LSTM hidden
                 file_embedding=64,    # File embedding size
                 subject_embedding=64, # Subject embedding size
                 dropout=0.3):
        super().__init__()
        
        # File-level encoder
        self.file_encoder = FileLevelLSTM(
            input_dim=n_features,
            hidden_dim=lstm_hidden,
            output_dim=file_embedding,
            dropout=dropout
        )
        
        # Subject-level aggregation (Attention)
        self.subject_aggregation = AttentionAggregation(
            embedding_dim=file_embedding
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=file_embedding,
            hidden_dim=32,
            dropout=0.5
        )
        
    def forward(self, batch_data):
        """
        batch_data: dict with keys:
          - 'files': list of tensors, each (seq_len, n_features)
          - 'file_lengths': list of sequence lengths
          - 'subject_ids': list of subject IDs for grouping
        """
        
        # Step 1: Encode all files
        file_embeddings = []
        for file_data, length in zip(batch_data['files'], batch_data['file_lengths']):
            emb = self.file_encoder(file_data.unsqueeze(0), [length])
            file_embeddings.append(emb)
        
        # Step 2: Group by subject and aggregate
        # (Implementation depends on batching strategy)
        # For simplicity, assume files are pre-grouped
        file_embeddings = torch.stack(file_embeddings)  # (n_files, embedding_dim)
        
        # Reshape to (batch_subjects, max_files, embedding_dim)
        # This requires custom collate function
        
        subject_emb, attention_weights = self.subject_aggregation(file_embeddings)
        
        # Step 3: Classify
        prediction = self.classifier(subject_emb)
        
        return {
            'prediction': prediction,
            'attention_weights': attention_weights,  # For interpretability
            'subject_embedding': subject_emb         # For visualization
        }
```

---

## 6. Training Strategy

### 6.1 Nested Cross-Validation (Like Current Code)

**Maintain the same evaluation protocol for fair comparison!**

```python
# Outer loop: 5-fold CV
for outer_fold in range(5):
    train_subjects, test_subjects = split_subjects(outer_fold)
    
    # Inner loop: Hyperparameter tuning (3-fold CV)
    best_model = hyperparameter_search(train_subjects)
    
    # Evaluate on test fold
    test_metrics = evaluate(best_model, test_subjects)
```

**Key Point**: Split at SUBJECT level, not file level (prevent data leakage)

### 6.2 Hyperparameters to Tune

**Architecture**:
- LSTM hidden dimensions: [64, 128, 256]
- File embedding size: [32, 64, 128]
- Number of LSTM layers: [1, 2, 3]
- Aggregation method: [attention, lstm, mean, max]
- Dropout rate: [0.2, 0.3, 0.5]

**Training**:
- Learning rate: [1e-5, 1e-4, 1e-3]
- Batch size: [4, 8, 16] subjects
- Optimizer: [Adam, AdamW]
- Weight decay: [0, 1e-5, 1e-4]

**Data**:
- Sequence length (max): [500, 1000, 2000] frames
- Padding strategy: [zero, last]
- Feature normalization: [per-file, global]

### 6.3 Loss Function & Metrics

**Loss Function** (for imbalanced data):
```python
# Option 1: Weighted Binary Cross-Entropy
pos_weight = (n_healthy / n_depressed)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Option 2: Focal Loss (handles class imbalance better)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

**Metrics** (same as current):
- AUC-ROC (primary)
- Accuracy
- F1-score
- Precision/Recall
- Confusion matrix

### 6.4 Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Get data
        files = [f.to(device) for f in batch['files']]
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        predictions = outputs['prediction'].squeeze()
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

---

## 7. Condition-Based & Phase-Based Experiments

### 7.1 Data Filtering

**Conditions**: Filter subjects before CV split
```python
def load_data_by_condition(condition):
    # condition: 'ALL', 'ADK', 'CR', 'CRADK', 'SHAM'
    metadata = pd.read_csv('participant_master_data.csv')
    
    if condition != 'ALL':
        subject_ids = metadata[metadata['condition'] == condition]['ID'].tolist()
    else:
        subject_ids = metadata['ID'].tolist()
    
    return load_subject_data(subject_ids)
```

**Phases**: Filter files when loading subject data
```python
def load_subject_files(subject_id, phase):
    # phase: 'all', 'emotion_induction_1', 'emotion_induction_2',
    #        'training_positive', 'training_negative', 'training_all'
    
    all_files = glob(f'{base_dir}/{subject_id}/timeseries_opensmile_features_vad/*.csv')
    
    if phase == 'all':
        return all_files
    elif phase == 'emotion_induction_1':
        return filter_by_phase(all_files, 'Belastungsphase', earliest=True)
    elif phase == 'emotion_induction_2':
        return filter_by_phase(all_files, 'Belastungsphase', earliest=False)
    # ... etc
```

### 7.2 Experiment Matrix

Run experiments for all combinations:

```python
conditions = ['ALL', 'ADK', 'CR', 'CRADK', 'SHAM']
phases = ['all', 'emotion_induction_1', 'emotion_induction_2', 
          'training_positive', 'training_negative', 'training_all']

for condition in conditions:
    for phase in phases:
        print(f"\n{'='*80}")
        print(f"Condition: {condition} | Phase: {phase}")
        print(f"{'='*80}\n")
        
        # Load data
        data = load_data(condition=condition, phase=phase)
        
        # Run nested CV
        results = run_nested_cv(data, model_class=HierarchicalLSTMDepression)
        
        # Save results
        save_results(results, condition=condition, phase=phase)
```

---

## 8. Practical Considerations

### 8.1 Computational Requirements

**Memory**:
- Each file: ~1000 frames × 62 features × 4 bytes = 250 KB
- 20 files/subject × 10 subjects/batch = 50 MB
- Model: ~5-10 MB
- **Total per batch**: ~100-200 MB GPU memory
- **Recommended GPU**: 8GB+ (e.g., RTX 3070, V100, A100)

**Training Time**:
- ~10-30 minutes per outer fold (depending on GPU)
- 5 outer folds × 1-5 hyperparameter combinations = **1-3 hours total**

### 8.2 Implementation Tips

**1. Custom DataLoader**
```python
class SubjectDataset(Dataset):
    def __init__(self, subject_ids, phase, condition, base_dir):
        self.subjects = []
        for sid in subject_ids:
            files = load_subject_files(sid, phase)
            label = get_label(sid)
            self.subjects.append({'id': sid, 'files': files, 'label': label})
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        # Load all files for this subject
        file_data = [load_csv(f) for f in subject['files']]
        return {
            'files': file_data,
            'label': subject['label'],
            'subject_id': subject['id']
        }
```

**2. Collate Function (Handle Variable Lengths)**
```python
def collate_fn(batch):
    # batch: list of dicts from __getitem__
    
    # Pad files within batch
    all_files = []
    all_lengths = []
    labels = []
    
    for sample in batch:
        for file_data in sample['files']:
            all_files.append(torch.FloatTensor(file_data))
            all_lengths.append(len(file_data))
        labels.append(sample['label'])
    
    # Pad to max length in batch
    padded_files = nn.utils.rnn.pad_sequence(all_files, batch_first=True)
    
    return {
        'files': padded_files,
        'lengths': torch.LongTensor(all_lengths),
        'labels': torch.FloatTensor(labels)
    }
```

**3. Early Stopping**
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop?
```

### 8.3 Debugging & Visualization

**1. Check Attention Weights**
```python
# After training, visualize which files are important
attention_weights = model(batch)['attention_weights']
for subject_id, weights in zip(batch['subject_ids'], attention_weights):
    print(f"Subject {subject_id}:")
    for file_name, weight in zip(batch['file_names'], weights):
        print(f"  {file_name}: {weight:.3f}")
```

**2. Visualize Embeddings (t-SNE/UMAP)**
```python
from sklearn.manifold import TSNE

subject_embeddings = []
labels = []
for batch in test_loader:
    outputs = model(batch)
    subject_embeddings.append(outputs['subject_embedding'].detach().cpu())
    labels.append(batch['labels'].cpu())

embeddings = torch.cat(subject_embeddings).numpy()
labels = torch.cat(labels).numpy()

# t-SNE
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
plt.title('Subject Embeddings (Depressed=1, Healthy=0)')
plt.show()
```

---

## 9. Alternative Architectures (For Comparison)

### 9.1 Transformer-Based

Replace LSTM with Transformer (better for longer sequences):

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, src_key_padding_mask=None):
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)
```

**Pros**: Better long-range dependencies, parallelizable
**Cons**: More parameters, may need more data

### 9.2 Conv1D + LSTM

Add convolutional layers before LSTM (learn local patterns):

```python
class ConvLSTM(nn.Module):
    def __init__(self, n_features=62):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq_len, 64)
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)
```

**Pros**: Efficient, captures local patterns
**Cons**: May miss long-range dependencies

---

## 10. Expected Performance

### Baseline (Aggregation)
- **Current results**: AUC = 0.622 (mean), Best = 0.736
- Simple, fast, interpretable

### LSTM (This Architecture)
- **Expected improvement**: +5-15% AUC
- **Rationale**: Captures temporal dynamics (voice changes over time, prosody patterns)
- **Target**: AUC = 0.70-0.80

### Why LSTM Might Be Better
1. **Temporal patterns**: Depression affects speech timing, pauses, prosody
2. **Context**: Frame-level features have temporal context (pitch contour, energy trajectory)
3. **Richer representation**: Learns hierarchical features automatically

### Why LSTM Might Not Help Much
1. **Small dataset**: 236 subjects may not be enough for deep learning
2. **High variance**: Individual differences dominate temporal patterns
3. **Aggregation already captures much**: Mean/std/min/max might capture most information

**Bottom Line**: Try both! If LSTM doesn't significantly outperform, aggregation is sufficient (and more interpretable).

---

## 11. Recommended Workflow for Student

### Phase 1: Setup & Baseline (Week 1)
1. Implement simple LSTM (single layer, mean pooling)
2. Test on 'ALL' condition, 'all' phase
3. Compare with aggregation baseline

### Phase 2: Architecture Improvements (Week 2)
1. Add bidirectional LSTM
2. Try attention-based aggregation
3. Tune hyperparameters

### Phase 3: Comprehensive Experiments (Week 3)
1. Run all condition × phase combinations
2. Analyze which conditions/phases benefit most from LSTM
3. Visualize attention weights and embeddings

### Phase 4: Analysis & Writing (Week 4)
1. Statistical comparison with baseline
2. Interpretability analysis
3. Write report/paper

---

## 12. References & Further Reading

**LSTM for Depression**:
- Gong, Y., & Poellabauer, C. (2017). "Topic modeling based multi-modal depression detection." ACM Multimedia.
- Alhanai, T., et al. (2018). "Detecting Depression with Audio/Text Sequence Modeling of Interviews." Interspeech.

**Hierarchical Models**:
- Yang, Z., et al. (2016). "Hierarchical Attention Networks for Document Classification." NAACL.
- Chao, L., et al. (2015). "Long short term memory recurrent neural network based multimodal dimensional emotion recognition." AVEC Workshop.

**Audio-Based Mental Health**:
- Cummins, N., et al. (2015). "A review of depression and suicide risk assessment using speech analysis." Speech Communication.
- Low, L., et al. (2020). "Automated assessment of psychiatric disorders using speech: A systematic review." Laryngoscope.

---

## 13. Summary & Key Takeaways

### Architecture Choice
**Hierarchical LSTM with Attention** ← RECOMMENDED
- Handles multiple files per subject
- Learns temporal patterns
- Interpretable (attention weights)
- Flexible (variable number of files)

### Training Strategy
- Nested CV (same as baseline)
- Split at subject level (prevent leakage)
- Tune hyperparameters on inner CV
- Early stopping on validation AUC

### Experiments
- Test all conditions: ALL, ADK, CR, CRADK, SHAM
- Test all phases: all, emotion_induction_1/2, training_positive/negative/all
- Compare with aggregation baseline

### Expected Outcome
- Potential improvement: +5-15% AUC
- Best case: AUC = 0.75-0.80
- Even if no improvement: Still valuable to show aggregation is sufficient

### Next Steps
1. Implement basic architecture
2. Test on one condition/phase
3. Compare with baseline
4. If promising → expand to all experiments
5. If not → analyze why and iterate

Good luck with the LSTM model! 🚀

