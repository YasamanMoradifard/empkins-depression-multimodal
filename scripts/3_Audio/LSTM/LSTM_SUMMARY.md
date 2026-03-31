# LSTM Depression Detection

This document explains what each folder provides and how all pieces fit together.

---

## 📁 **FOLDER 1: `data/` - Data Loading & Preprocessing**

### **Purpose**: Load and prepare data for training

### **What You Get**:

#### **1. `dataset.py`** - Main data module

**Data Loading Functions:**
- `load_subjects_from_processed(condition, phase, processed_audio_dir, metadata_csv)`
  - **Input**: Condition (CR/CRADK/ADK/SHAM/All), phase (training_pos/induction1/etc), paths
  - **Output**: `List[SubjectDict]` where each dict has:
    ```python
    {
        'subject_id': str,        # e.g., "4"
        'file_paths': List[str],   # List of CSV file paths for this subject
        'label': int               # 0=Healthy, 1=Depressed
    }
    ```
  - **What it does**: Reads metadata CSV, filters by condition/phase, finds matching CSV files

**Helper Functions:**
- `normalize_condition(condition)` - Normalizes condition names
- `diagnose_to_label(diagnose)` - Converts "Healthy"/"Depressed" to 0/1

**PyTorch Dataset Class:**
- `SubjectDataset(subjects, normalize=True, handle_nan_inf=True)`
  - **Input**: List of SubjectDict from `load_subjects_from_processed()`
  - **Output** (from `__getitem__(idx)`):
    ```python
    {
        'files': List[np.ndarray],      # Each shape (T_i, F) - variable length
        'file_lengths': List[int],       # [T₁, T₂, ...]
        'label': int,                   # 0 or 1
        'subject_id': str                # Subject identifier
    }
    ```
  - **What it does**: 
    - Loads CSV files per subject
    - Excludes metadata columns (ID, diagnose, condition, phase, aufgabe)
    - Handles NaN/Inf values (replaces with feature mean)
    - Optionally normalizes per-file (zero mean, unit variance)

#### **2. `collate.py`** - Batching function

- `collate_fn(batch)` - Custom collate for DataLoader
  - **Input**: List of dicts from `SubjectDataset` (one per subject)
  - **Output**:
    ```python
    {
        'file_sequences': Tensor,        # (total_files, max_timesteps, n_features) - padded
        'frame_lengths': Tensor,        # (total_files,) - actual lengths
        'file_mask': Tensor,            # (batch_size, max_files) - 1 for valid, 0 for padding
        'subject_file_counts': List[int],  # Files per subject [2, 1, 3, ...]
        'file_to_subject': List[int],   # Maps file index → subject index
        'labels': Tensor,               # (batch_size,) - subject labels
        'subject_ids': List[str],       # Subject identifiers
        'max_files': int                # Max files per subject in batch
    }
    ```
  - **What it does**: 
    - Flattens all files from all subjects
    - Pads sequences to max length in batch
    - Creates masks for valid files
    - Validates feature count consistency

- `group_file_embeddings(...)` - Groups file embeddings by subject (optional, not used by model)

#### **3. `__init__.py`** - Module exports
- Exports: `load_subjects_from_processed`, `SubjectDict`, `SubjectDataset`, `collate_fn`, `group_file_embeddings`

---

## 📁 **FOLDER 2: `models/` - Neural Network Architecture**

### **Purpose**: Define the hierarchical LSTM model architecture

### **What You Get**:

#### **1. `hierarchical_lstm.py`** - Main model class

- `HierarchicalLSTMDepression(...)` - Complete model
  - **Input** (in `forward()`): Batch dict from `collate_fn()`
  - **Output**:
    ```python
    {
        'prediction': Tensor,           # (batch_size, 1) - depression logits
        'subject_embedding': Tensor,    # (batch_size, embedding_dim)
        'attention_weights': Tensor or None  # (batch_size, max_files) if attention used
    }
    ```
  - **Architecture**:
    1. **File-level LSTM**: Encodes each file → file embeddings
    2. **Subject-level aggregation**: Combines file embeddings → subject embedding
    3. **Classification head**: Maps subject embedding → depression probability

#### **2. `file_lstm.py`** - File-level encoder

- `FileLevelLSTM(input_dim, hidden_dim, output_dim, dropout, use_layer_norm)`
  - **Input**: `(total_files, max_timesteps, n_features)`, `(total_files,)` lengths
  - **Output**: `(total_files, file_embedding_dim)` - one embedding per file
  - **Architecture**: 2-layer bidirectional LSTM

#### **3. `aggregation.py`** - Subject-level aggregation methods

- `AttentionAggregation(embedding_dim)` - **Recommended**
  - Learns importance weights for each file
  - Returns weighted sum + attention weights
  
- `LSTMAggregation(file_embedding_dim, hidden_dim)`
  - Uses LSTM to process file sequence
  
- `SimpleAggregation(embedding_dim, mode)` - mode: 'mean', 'max', 'mean_max'
  - Simple pooling operations

#### **4. `classifier.py`** - Classification head

- `ClassificationHead(input_dim, hidden_dim, hidden_dim2, dropout)`
  - **Input**: `(batch_size, subject_embedding_dim)`
  - **Output**: `(batch_size, 1)` - logits (no sigmoid, BCEWithLogitsLoss handles it)
  - **Architecture**: Dense(32) → ReLU → Dropout → Dense(16) → ReLU → Dropout → Dense(1)

#### **5. `__init__.py`** - Module exports
- Exports: `FileLevelLSTM`, `AttentionAggregation`, `LSTMAggregation`, `SimpleAggregation`, `ClassificationHead`, `HierarchicalLSTMDepression`

---

## 📁 **FOLDER 3: `training/` - Training & Evaluation**

### **Purpose**: Train models and evaluate performance

### **What You Get**:

#### **1. `train.py`** - Training functions

- `train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, ...)`
  - **Input**: 
    - Model instance
    - Train & validation DataLoaders
    - Training hyperparameters
  - **Output**:
    ```python
    {
        'history': {
            'train': {'loss': [...], 'auc': [...], 'accuracy': [...], 'f1': [...]},
            'val': {'loss': [...], 'auc': [...], 'accuracy': [...], 'f1': [...]}
        },
        'best_val_auc': float,
        'best_model_path': Path  # Path to saved best model
    }
    ```
  - **What it does**:
    - Trains model for N epochs
    - Validates after each epoch
    - Implements early stopping (based on validation AUC)
    - Saves best model checkpoint
    - Returns training history

**Helper Functions:**
- `train_epoch(...)` - Trains for one epoch, returns metrics
- `validate(...)` - Validates model, returns metrics
- `EarlyStopping` - Class for early stopping logic
- `compute_pos_weight(...)` - Computes class weights for weighted BCE loss

#### **2. `evaluate.py`** - Evaluation functions

- `evaluate_model(model, dataloader, device, save_predictions=True, output_dir=None)`
  - **Input**: Trained model, test DataLoader
  - **Output**:
    ```python
    {
        'auc': float,
        'accuracy': float,
        'f1': float,
        'precision': float,
        'recall': float,
        'y_true': List[int],
        'y_pred': List[float],
        'confusion_matrix': {'tn': int, 'fp': int, 'fn': int, 'tp': int}
    }
    ```
  - **What it does**:
    - Evaluates model on test set
    - Computes all metrics
    - Saves predictions CSV
    - Saves attention weights (if available)
    - Saves metrics JSON

- `load_model(model_path, device, **model_kwargs)`
  - **Input**: Path to saved model checkpoint, device, model architecture kwargs
  - **Output**: Loaded model ready for evaluation
  - **What it does**: Loads model state dict from checkpoint

#### **3. `__init__.py`** - Empty (no exports)

---

## 📁 **FOLDER 4: `utils/` - Utilities**

### **Purpose**: Configuration, metrics, and visualization helpers

### **What You Get**:

#### **1. `config.py`** - Configuration constants

**Path Configuration:**
- `PROCESSED_AUDIO_DIR` - Path to OpenSmile_data folder
- `MERGED_METADATA_CSV` - Path to merged_RCT_info.csv
- `RESULTS_DIR` - Path to save results

**Model Architecture:**
- `N_FEATURES` - Number of input features (auto-detected)
- `LSTM_HIDDEN_DIM` - LSTM hidden dimension (default: 128)
- `FILE_EMBEDDING_DIM` - File embedding size (default: 64)
- `SUBJECT_EMBEDDING_DIM` - Subject embedding size (default: 64)
- `AGGREGATION_METHOD` - 'attention', 'lstm', 'mean', 'max', 'mean_max'
- `DROPOUT_LSTM`, `DROPOUT_CLASSIFIER` - Dropout rates

**Training Hyperparameters:**
- `BATCH_SIZE` - Batch size (default: 4)
- `LEARNING_RATE` - Learning rate (default: 1e-4)
- `NUM_EPOCHS` - Max epochs (default: 100)
- `EARLY_STOPPING_PATIENCE` - Early stopping patience (default: 10)
- `USE_WEIGHTED_BCE` - Use class-weighted loss (default: True)

#### **2. `metrics.py`** - Metric computation

- `compute_metrics(y_true, y_pred)`
  - **Input**: True labels (0/1), predicted probabilities (0-1)
  - **Output**: Dict with `{'auc', 'accuracy', 'f1', 'precision', 'recall'}`
  - **What it does**: Computes classification metrics

- `compute_confusion_matrix(y_true, y_pred_binary)`
  - **Input**: True labels, binary predictions
  - **Output**: Dict with `{'tn', 'fp', 'fn', 'tp'}`
  - **What it does**: Computes confusion matrix

#### **3. `visualization.py`** - Plotting functions

- `plot_roc_curve(y_true, y_pred, save_path, title)`
  - Plots ROC curve with AUC score
  - Saves to file or displays

- `plot_training_history(history, save_path)`
  - Plots training curves (loss, AUC, accuracy, F1) for train/val
  - 2x2 subplot layout

- `plot_attention_weights(attention_weights, save_path, max_subjects)`
  - Plots attention weights for example subjects
  - Bar charts showing file importance

#### **4. `__init__.py`** - Empty (no exports)

---

## 🔗 **HOW IT ALL FITS TOGETHER**

### **Complete Pipeline Flow**:

```
1. DATA LOADING (data/)
   └─> load_subjects_from_processed()
       └─> Returns List[SubjectDict] with file paths and labels
   
   └─> SubjectDataset(subjects)
       └─> Loads CSV files, preprocesses
       └─> Returns dicts with 'files', 'file_lengths', 'label', 'subject_id'
   
   └─> DataLoader(..., collate_fn=collate_fn)
       └─> Batches subjects, pads sequences
       └─> Returns batched dict ready for model

2. MODEL (models/)
   └─> HierarchicalLSTMDepression(...)
       └─> Forward pass:
           ├─> FileLevelLSTM → file embeddings
           ├─> Aggregation → subject embedding
           └─> ClassificationHead → depression probability

3. TRAINING (training/)
   └─> train_model(model, train_loader, val_loader, ...)
       └─> Trains for N epochs
       └─> Returns history and saves best model

4. EVALUATION (training/)
   └─> load_model(model_path, ...)
       └─> Loads saved model
   
   └─> evaluate_model(model, test_loader, ...)
       └─> Computes metrics, saves predictions

5. VISUALIZATION (utils/)
   └─> plot_roc_curve(...)
   └─> plot_training_history(...)
   └─> plot_attention_weights(...)
```

---

## 📋 **SUMMARY TABLE**

| Folder | Main Purpose | Key Functions/Classes | Output Type |
|--------|-------------|----------------------|-------------|
| **data/** | Load & preprocess | `load_subjects_from_processed()`, `SubjectDataset`, `collate_fn()` | Batched tensors ready for model |
| **models/** | Define architecture | `HierarchicalLSTMDepression`, `FileLevelLSTM`, `AttentionAggregation`, `ClassificationHead` | Model predictions (logits) |
| **training/** | Train & evaluate | `train_model()`, `evaluate_model()`, `load_model()` | Trained models, metrics, predictions |
| **utils/** | Configuration & helpers | `config.py` constants, `compute_metrics()`, `plot_*()` functions | Metrics dicts, plots |

---

## 🎯 **WHAT YOU NEED FOR A NEW main.py**

### **Imports You'll Need**:

```python
# Data
from data.dataset import load_subjects_from_processed, SubjectDataset
from data.collate import collate_fn

# Models
from models.hierarchical_lstm import HierarchicalLSTMDepression

# Training
from training.train import train_model
from training.evaluate import evaluate_model, load_model

# Utils
from utils.config import (
    PROCESSED_AUDIO_DIR, MERGED_METADATA_CSV, RESULTS_DIR,
    N_FEATURES, LSTM_HIDDEN_DIM, FILE_EMBEDDING_DIM, 
    AGGREGATION_METHOD, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
from utils.metrics import compute_metrics
from utils.visualization import plot_roc_curve, plot_training_history, plot_attention_weights
```

### **Typical Workflow**:

1. **Load data** → `load_subjects_from_processed()` → `SubjectDataset()` → `DataLoader()`
2. **Create model** → `HierarchicalLSTMDepression(...)`
3. **Train** → `train_model(...)`
4. **Evaluate** → `load_model()` → `evaluate_model()`
5. **Visualize** → `plot_roc_curve()`, `plot_training_history()`

---
