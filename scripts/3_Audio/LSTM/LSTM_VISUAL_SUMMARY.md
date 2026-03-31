# LSTM Architecture - Quick Visual Summary

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUBJECT DATA                                 │
│  Subject_123 (Depressed):                                          │
│    ├─ emotion_induction_1_file1.csv (1000 frames × 62 features)   │
│    ├─ emotion_induction_1_file2.csv (800 frames × 62 features)    │
│    ├─ training_positive_file1.csv (1200 frames × 62 features)     │
│    ├─ training_positive_file2.csv (900 frames × 62 features)      │
│    └─ ... (15 more files)                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVEL 1: FILE ENCODING                           │
│                    (Shared Weights Across All Files)                │
│                                                                     │
│  ┌────────────────────┐       ┌────────────────────┐              │
│  │  File 1            │       │  File 2            │              │
│  │  (1000×62)         │       │  (800×62)          │   ...        │
│  │                    │       │                    │              │
│  │  ↓ BiLSTM(128)    │       │  ↓ BiLSTM(128)    │              │
│  │  ↓ BiLSTM(64)     │       │  ↓ BiLSTM(64)     │              │
│  │                    │       │                    │              │
│  │  Embedding (64)    │       │  Embedding (64)    │              │
│  └────────────────────┘       └────────────────────┘              │
│           ↓                            ↓                            │
│         emb_1                        emb_2              ... emb_N   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│              LEVEL 2: SUBJECT-LEVEL AGGREGATION                     │
│                                                                     │
│              ┌─────────────────────────────┐                       │
│              │  Attention Mechanism        │                       │
│              │                             │                       │
│              │  α₁ = 0.15  (emotion ind.)  │                       │
│              │  α₂ = 0.08  (emotion ind.)  │                       │
│              │  α₃ = 0.32  (training pos)  │ ← Learned weights     │
│              │  α₄ = 0.25  (training pos)  │                       │
│              │  ...                        │                       │
│              │  αₙ = 0.05  (other)         │                       │
│              │                             │                       │
│              │  Σ(αᵢ × embᵢ) = subject_emb │                       │
│              └─────────────────────────────┘                       │
│                          ↓                                          │
│                  Subject Embedding (64)                             │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  LEVEL 3: CLASSIFICATION                            │
│                                                                     │
│                  Dense(32) + ReLU + Dropout(0.5)                   │
│                          ↓                                          │
│                  Dense(16) + ReLU + Dropout(0.5)                   │
│                          ↓                                          │
│                  Dense(1) + Sigmoid                                 │
│                          ↓                                          │
│                  P(Depressed) = 0.78                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: Aggregation vs LSTM

### Current Approach (Aggregation)

```
Files → Concatenate all frames → Aggregate (mean/std/min/max) → Features → ML Model
   
Subject_123:
  File1: [frame1, frame2, ..., frame1000] ┐
  File2: [frame1, frame2, ..., frame800]  ├─→ Concatenate all → [10,000 frames × 62]
  ...                                      │                              ↓
  FileN: [frame1, frame2, ..., frame900]  ┘                    Aggregate each feature:
                                                                  mean_pitch = 150 Hz
                                                                  std_pitch = 25 Hz
                                                                  max_energy = 0.8
                                                                  ...
                                                                  (125 features total)
                                                                         ↓
                                                              Logistic Regression
                                                                         ↓
                                                                  P(Depressed)

Pros: ✅ Simple, fast, interpretable
Cons: ❌ Loses temporal information, treats all frames equally
```

### LSTM Approach

```
Files → LSTM per file → File embeddings → Attention → Subject embedding → Classification

Subject_123:
  File1: [frames] → LSTM → emb1 (64) ┐
  File2: [frames] → LSTM → emb2 (64) ├─→ Attention → subject_emb (64)
  ...                                 │                      ↓
  FileN: [frames] → LSTM → embN (64) ┘              Classification Head
                                                             ↓
                                                      P(Depressed)

Pros: ✅ Preserves temporal patterns, learns importance, captures dynamics
Cons: ❌ More complex, needs more data, harder to interpret
```

---

## Why Hierarchical? Why Not Just One Big LSTM?

### Option 1: Concatenate ALL frames → Single LSTM ❌

```
Subject_123:
  Concatenate all 20 files → [10,000 frames × 62] → Single LSTM → Prediction

Problems:
  ❌ Too long (10,000 timesteps) → Memory explosion, gradient vanishing
  ❌ Different files have different meanings (induction vs training)
  ❌ Temporal order across files may not be meaningful
  ❌ Padding/masking becomes complicated
```

### Option 2: Hierarchical (FILE → SUBJECT) ✅

```
Subject_123:
  File1 → LSTM → emb1 ┐
  File2 → LSTM → emb2 ├─→ Aggregate → Prediction
  ...                 │
  FileN → LSTM → embN ┘

Advantages:
  ✅ Shorter sequences (~1000 frames) → Manageable
  ✅ Each file processed independently → Parallelizable
  ✅ Can handle variable number of files per subject
  ✅ Attention learns which files matter → Interpretable
  ✅ Preserves temporal patterns within each file
```

---

## Attention Mechanism Explanation

### What Does Attention Do?

**Question**: Not all files are equally informative for depression. How do we weight them?

**Answer**: Learn importance weights automatically!

### Example

```
Subject_123 has 5 files:

File                            LSTM Embedding    Attention Weight    Interpretation
─────────────────────────────────────────────────────────────────────────────────
emotion_induction_1_task1       [0.1, -0.5, ...]     α₁ = 0.35      ← Most important!
emotion_induction_1_task2       [0.2, -0.3, ...]     α₂ = 0.25      ← Important
training_positive_task1         [-0.1, 0.2, ...]     α₃ = 0.15      ← Somewhat important
training_positive_task2         [0.0, 0.1, ...]      α₄ = 0.15      ← Somewhat important
training_positive_task3         [0.3, 0.4, ...]      α₅ = 0.10      ← Less important

Subject embedding = 0.35×emb₁ + 0.25×emb₂ + 0.15×emb₃ + 0.15×emb₄ + 0.10×emb₅
                  = [weighted average of all embeddings]

Prediction = Classifier(subject_embedding) = P(Depressed) = 0.78
```

**Insight**: Emotion induction files are more informative than training files for this subject!

### How Attention is Learned

```python
# During training:
for each subject:
    for each file:
        file_emb = LSTM(file_data)
    
    # Attention network
    scores = AttentionNetwork(file_embeddings)  # [s₁, s₂, ..., sₙ]
    weights = softmax(scores)                   # [α₁, α₂, ..., αₙ] (sum to 1)
    
    subject_emb = Σ(weights[i] × file_embeddings[i])
    
    prediction = Classifier(subject_emb)
    
    # Backpropagation adjusts:
    # - LSTM parameters (to extract better file embeddings)
    # - Attention parameters (to learn which files are important)
    # - Classifier parameters (to map embeddings to depression)
    
    loss = BCE(prediction, true_label)
    loss.backward()
```

---

## Training Example (Nested CV)

```
Dataset: 236 subjects (123 depressed, 113 healthy)

Outer Loop (5-fold CV):
┌─────────────────────────────────────────────────────────────────┐
│  Fold 1: Train on 188 subjects, Test on 48 subjects            │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  Inner Loop (3-fold CV for hyperparameter tuning):   │     │
│  │  ┌─────────────────────────────────────────────┐     │     │
│  │  │  Subfold 1: Train 125, Val 63               │     │     │
│  │  │    → Test different hyperparameters         │     │     │
│  │  │       - LSTM hidden: 64, 128, 256          │     │     │
│  │  │       - Learning rate: 1e-4, 1e-3          │     │     │
│  │  │       - Dropout: 0.3, 0.5                  │     │     │
│  │  │    → Select best hyperparameters            │     │     │
│  │  └─────────────────────────────────────────────┘     │     │
│  │  Best hyperparameters: LSTM=128, LR=1e-4, Drop=0.3  │     │
│  └───────────────────────────────────────────────────────┘     │
│  Train final model on all 188 subjects with best params        │
│  Test on 48 held-out subjects → AUC = 0.75                     │
└─────────────────────────────────────────────────────────────────┘

Fold 2: ... (similar)
Fold 3: ... (similar)
Fold 4: ... (similar)
Fold 5: ... (similar)

Final Result: Mean AUC = 0.73 ± 0.05
```

---

## Key Design Decisions

| Decision | Options | Recommendation | Rationale |
|----------|---------|----------------|-----------|
| **File Encoding** | LSTM / Transformer / Conv1D | Bidirectional LSTM | Standard, works well, handles variable length |
| **Aggregation** | Attention / LSTM / Mean/Max | **Attention** | Learns importance, interpretable, flexible |
| **Loss Function** | BCE / Weighted BCE / Focal | Weighted BCE | Handles class imbalance |
| **Optimizer** | Adam / AdamW / SGD | AdamW | Better generalization, weight decay |
| **Batch Size** | 4 / 8 / 16 subjects | 8 | Balance GPU memory and gradient stability |
| **Learning Rate** | 1e-5 / 1e-4 / 1e-3 | 1e-4 | Standard for LSTMs |
| **Dropout** | 0.2 / 0.3 / 0.5 | 0.3 (LSTM), 0.5 (Dense) | Prevent overfitting |

---

## Expected Results by Condition/Phase

Based on aggregation baseline results, we expect:

| Condition/Phase | Baseline AUC | Expected LSTM AUC | Why? |
|-----------------|--------------|-------------------|------|
| **ALL / all** | 0.62 | **0.70-0.75** | Large data, temporal patterns matter |
| **SHAM / all** | ? | 0.65-0.70 | Clean control group |
| **CRADK / all** | ? | 0.60-0.65 | Smaller sample |
| **ALL / emotion_induction** | ? | **0.72-0.78** | Standardized task, strong temporal patterns |
| **ALL / training_positive** | ? | 0.65-0.70 | More varied responses |

**Hypothesis**: LSTM will help most for emotion induction phases (standardized elicitation) and less for training phases (more variability).

---

## Troubleshooting Guide

### Issue 1: Model overfits (high train AUC, low val AUC)

**Solutions**:
- ✅ Increase dropout (0.3 → 0.5)
- ✅ Add L2 regularization (weight_decay=1e-4)
- ✅ Reduce model capacity (128 → 64 hidden units)
- ✅ Early stopping (patience=10 epochs)

### Issue 2: Model underfits (low train AUC)

**Solutions**:
- ✅ Increase model capacity (64 → 128 → 256)
- ✅ Add more layers (1 → 2 LSTM layers)
- ✅ Increase learning rate (1e-4 → 1e-3)
- ✅ Train longer (more epochs)

### Issue 3: Training is slow

**Solutions**:
- ✅ Reduce sequence length (max 1000 frames instead of 2000)
- ✅ Use smaller batch size if memory is bottleneck
- ✅ Use gradient accumulation for effective larger batch
- ✅ Use mixed precision training (fp16)

### Issue 4: LSTM doesn't outperform aggregation

**Analysis**:
- ❓ Are temporal patterns actually informative? Plot features over time for depressed vs healthy
- ❓ Is dataset too small? Try with more data or simpler model
- ❓ Is aggregation capturing most information? Check if aggregation includes temporal stats (std, range)

**Conclusion**: If aggregation works well, stick with it! Simpler is better if performance is similar.

---

## File Structure for Student

```
audio_lstm_depression/
├── data/
│   ├── load_data.py           # Load and filter by condition/phase
│   └── dataset.py              # PyTorch Dataset + DataLoader
├── models/
│   ├── file_lstm.py            # File-level LSTM encoder
│   ├── aggregation.py          # Attention / LSTM / Mean aggregation
│   ├── classifier.py           # Classification head
│   └── hierarchical_lstm.py    # Full model integration
├── training/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   └── nested_cv.py            # Nested cross-validation
├── utils/
│   ├── config.py               # Hyperparameters
│   └── visualization.py        # Plot attention, embeddings
├── experiments/
│   ├── run_all_experiments.py  # Run all condition×phase combos
│   └── compare_with_baseline.py
├── results/
│   └── [saved results]
└── main.py                     # Entry point
```

---

## Summary for Student Meeting

### 🎯 Goal
Build LSTM model for audio-based depression detection that:
1. Handles multiple audio files per subject
2. Preserves temporal information (unlike aggregation)
3. Supports condition/phase experiments
4. Compares fairly with aggregation baseline

### 🏗️ Architecture
**Hierarchical LSTM with Attention**
- Level 1: LSTM encodes each file → embedding
- Level 2: Attention aggregates file embeddings → subject embedding
- Level 3: Dense layers classify → depression probability

### 📊 Experiments
- 5 conditions × 6 phases = 30 experiments
- Nested 5-fold CV (same as baseline)
- Compare AUC with aggregation

### 📅 Timeline
- Week 1: Implement + test on 1 condition/phase
- Week 2: Tune hyperparameters + improve architecture
- Week 3: Run all 30 experiments
- Week 4: Analyze + write report

### 🎓 Learning Outcomes
- Deep learning for time-series health data
- Hierarchical models for multi-instance learning
- Attention mechanisms and interpretability
- Rigorous ML evaluation (nested CV)

Good luck! 🚀

