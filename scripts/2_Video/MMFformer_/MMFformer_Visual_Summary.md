# MMFformer Architecture - Quick Visual Summary

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUBJECT DATA                                 │
│  Subject_123 (Depressed):                                          │
│    ├─ Audio features: (T frames × 161 features)                    │
│    │   └─ From: OpenSMILE/VGGish/LDDs (varies by dataset)          │
│    └─ Video features: (T frames × 136 features)                    │
│        └─ From: Facial landmarks (68 points × 2 coords = 136)      │
│    Combined: (T frames × 297 features)                             │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVEL 1: MODALITY SEPARATION                     │
│                                                                     │
│  Input: (batch, T, 297)                                            │
│    ├─ Split: Audio (batch, T, 161)                                 │
│    │   └─ Source: Audio features (OpenSMILE/VGGish/LDDs)           │
│    │       Note: Dimension varies by dataset:                       │
│    │       - d02: 128 (OpenSMILE)                                  │
│    │       - dvlog: 25 (LDDs)                                      │
│    │       - lmvd: 128 (VGGish)                                    │
│    │       - Default: 161 (model default)                          │
│    └─ Split: Video (batch, T, 136)                                 │
│        └─ Source: Facial landmarks from NPZ files                  │
│            - 68 facial landmark points                             │
│            - Each point: (x, y) coordinates                        │
│            - Total: 68 × 2 = 136 dimensions                        │
│                                                                     │
│  Projection:                                                       │
│    ├─ Conv1D: Audio 161 → 128                                      │
│    └─ Conv1D: Video 136 → 128                                      │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│              LEVEL 2: MODALITY-SPECIFIC ENCODING                    │
│                                                                     │
│  ┌────────────────────┐       ┌────────────────────┐              │
│  │  Audio Branch      │       │  Video Branch       │              │
│  │                    │       │                     │              │
│  │  AudioTransformer  │       │  VisualMAE          │              │
│  │  (AudioSet         │       │  (Pretrained        │              │
│  │   Pretrained)      │       │   Visual Encoder)    │              │
│  │                    │       │                     │              │
│  │  Output:           │       │  Output:            │              │
│  │  (T_a, 768)        │       │  (T_v, 768)         │              │
│  └────────────────────┘       └────────────────────┘              │
│           ↓                            ↓                            │
│    Conv1D Blocks                Conv1D Blocks                       │
│    768 → 512 → 256 → 128       768 → 512 → 256 → 128               │
│           ↓                            ↓                            │
│    Audio Features              Video Features                        │
│    (T_a, 128)                  (T_v, 128)                           │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│              LEVEL 3: MULTI-MODAL FUSION                            │
│                    (Depends on Fusion Type)                         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  FUSION TYPE: 'ia' (Intermediate Attention) [RECOMMENDED]  │   │
│  │                                                             │   │
│  │  1. Cross-Attention:                                       │   │
│  │     Audio queries Video → h_av                             │   │
│  │     Video queries Audio → h_va                             │   │
│  │                                                             │   │
│  │  2. Feature Modulation:                                    │   │
│  │     Audio' = h_va ⊙ Audio (element-wise)                    │   │
│  │     Video' = h_av ⊙ Video (element-wise)                   │   │
│  │                                                             │   │
│  │  3. Final Conv1D:                                          │   │
│  │     Audio' → (T_a, 128)                                    │   │
│  │     Video' → (T_v, 128)                                    │   │
│  │                                                             │   │
│  │  4. Temporal Pooling:                                      │   │
│  │     Mean pooling across time → (128) each                  │   │
│  │                                                             │   │
│  │  5. Concatenate:                                           │   │
│  │     [Audio_pooled, Video_pooled] → (256)                   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  FUSION TYPE: 'MT' (Mutual Transformer)                    │   │
│  │                                                             │   │
│  │  1. Encode: Audio & Video separately                       │   │
│  │  2. Cross-Attention: Audio→Video & Video→Audio             │   │
│  │  3. Self-Attention: [Audio+Video] concatenated             │   │
│  │  4. Concatenate all: [fav, fva, f_av] → (768)             │   │
│  │  5. Temporal Pooling: Mean → (768)                         │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  FUSION TYPE: 'lt' (Late Transformer)                      │   │
│  │                                                             │   │
│  │  1. Process Audio & Video separately                        │   │
│  │  2. Cross-Attention at the end                              │   │
│  │  3. Concatenate → (256)                                     │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  FUSION TYPE: 'it' (Intermediate Transformer)              │   │
│  │                                                             │   │
│  │  1. Cross-Attention in middle                              │   │
│  │  2. Residual connection                                     │   │
│  │  3. Continue processing                                     │   │
│  │  4. Concatenate → (256)                                     │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  LEVEL 4: CLASSIFICATION                            │
│                                                                     │
│                  Dropout(0.5)                                      │
│                          ↓                                          │
│                  Linear(256 → 1) or Linear(768 → 1)                │
│                          ↓                                          │
│                  Sigmoid                                            │
│                          ↓                                          │
│                  P(Depressed) = 0.78                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: Single Modality vs Multi-Modal Fusion

### Audio-Only Approach

```
Audio Features (T × 161)
    ↓
AudioTransformer (Pretrained)
    ↓
Conv1D Blocks (768 → 512 → 256 → 128)
    ↓
Temporal Pooling (Mean)
    ↓
Linear(768 → 1) → P(Depressed)

Pros: ✅ Simple, fast, uses pretrained audio features
Cons: ❌ Ignores visual cues (facial expressions, body language)
```

### Video-Only Approach

```
Video Features (T × 136)
    ↓
VisualMAE (Pretrained)
    ↓
Conv1D Blocks (768 → 512 → 256 → 128)
    ↓
Temporal Pooling (Mean)
    ↓
Linear(768 → 1) → P(Depressed)

Pros: ✅ Captures visual depression markers
Cons: ❌ Ignores vocal characteristics (prosody, tone)
```

### Multi-Modal Fusion (MMFformer) ✅

```
Audio + Video Features
    ↓
Separate Encoding (AudioTransformer + VisualMAE)
    ↓
Fusion Mechanism (Attention/Transformer)
    ↓
Combined Representation
    ↓
Classification → P(Depressed)

Pros: ✅ Leverages both modalities, complementary information
Cons: ❌ More complex, requires alignment
```

---

## Fusion Strategies Explained

### 1. Intermediate Attention (IA) - RECOMMENDED

**How it Works:**
```
Step 1: Encode Audio & Video separately
  Audio: (T_a, 128)
  Video: (T_v, 128)

Step 2: Cross-Attention
  h_av = Attention(Video, Audio)  # Video attends to Audio
  h_va = Attention(Audio, Video)  # Audio attends to Video

Step 3: Feature Modulation (Element-wise multiplication)
  Audio' = Audio ⊙ h_va  # Audio features weighted by Video attention
  Video' = Video ⊙ h_av  # Video features weighted by Audio attention

Step 4: Final Processing
  Audio' → Conv1D → Pool → (128)
  Video' → Conv1D → Pool → (128)
  Concatenate → (256)
```

**Why IA?**
- ✅ Learns which audio moments align with important video moments
- ✅ Modulates features based on cross-modal attention
- ✅ Preserves temporal structure
- ✅ Interpretable (can visualize attention weights)

### 2. Mutual Transformer (MT)

**How it Works:**
```
Step 1: Encode separately
  Audio_encoded: (T_a, 256)
  Video_encoded: (T_v, 256)

Step 2: Three Cross-Attention Operations
  MT-1: Audio queries Video → fav
  MT-2: Video queries Audio → fva
  MT-3: [Audio+Video] self-attention → f_av

Step 3: Concatenate all
  [fav, fva, f_av] → (T, 768)

Step 4: Temporal Pooling
  Mean → (768)
```

**Why MT?**
- ✅ Most sophisticated fusion
- ✅ Three-way attention (bidirectional + self)
- ✅ Captures complex interactions
- ⚠️ More parameters, needs more data

### 3. Late Transformer (LT)

**How it Works:**
```
Step 1: Process Audio & Video independently
  Audio → Conv1D blocks → (T_a, 128)
  Video → Conv1D blocks → (T_v, 128)

Step 2: Cross-Attention at the end
  h_av = Transformer(Video, Audio)
  h_va = Transformer(Audio, Video)

Step 3: Pool & Concatenate
  [Audio_pooled, Video_pooled] → (256)
```

**Why LT?**
- ✅ Simple fusion strategy
- ✅ Modalities processed independently first
- ✅ Good baseline for comparison

### 4. Intermediate Transformer (IT)

**How it Works:**
```
Step 1: Partial processing
  Audio → Conv1D (partial) → (T_a, 128)
  Video → Conv1D (partial) → (T_v, 128)

Step 2: Cross-Attention in middle
  h_av = Transformer(Video, Audio)
  h_va = Transformer(Audio, Video)

Step 3: Residual connection
  Audio' = Audio + h_av
  Video' = Video + h_va

Step 4: Continue processing
  Audio' → Conv1D → Pool → (128)
  Video' → Conv1D → Pool → (128)
  Concatenate → (256)
```

**Why IT?**
- ✅ Fusion happens during processing (not at end)
- ✅ Residual connections preserve original features
- ✅ Balance between early and late fusion

---

## Training Pipeline

```
Dataset: d02 (EKS) / dvlog-dataset / lmvd-dataset
    ↓
Data Loading:
  - Filter by condition (ALL, ADK, CR, CRADK, SHAM)
  - Filter by phase (all, emotion_induction_1/2, training_pos/neg)
  - Filter by modality (av, audio, video)
    ↓
DataLoader:
  - Batch size: 16
  - Variable length sequences (padding)
    ↓
Model: MultiModalDepDet
  - Fusion: ia / lt / it / MT
  - Audio: AudioTransformer (AudioSet pretrained)
  - Video: VisualMAE (pretrained)
    ↓
Training:
  - Optimizer: AdamW / Adam / SGD
  - Learning Rate: 1e-5 (typical)
  - Loss: CombinedLoss (Focal + L2)
  - Scheduler: CosineAnnealing / StepLR / Plateau
  - Early Stopping: Patience=10
    ↓
Evaluation:
  - Metrics: Accuracy, Precision, Recall, F1
  - Confusion Matrix
  - Training Curves
    ↓
Results:
  - Saved to runlog_{condition}_{phase}_{timestamp}/
  - Plots: train_val_loss.png, train_val_acc.png
  - Confusion matrices for val and test
```

---

## Dataset Splitting Strategy (d02_manifests.py)

### Overview

The `d02_manifests.py` script creates train/validation/test splits for the d02 dataset with proper stratification and data leakage prevention.

### Split Configuration

```
Split Ratios:
  - Train: 70% of participants
  - Validation: 10% of participants
  - Test: 20% of participants
  - Random Seed: 42 (for reproducibility)
```

### Splitting Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: LOAD DATA                               │
│                                                                     │
│  Input: merged_RCT_info.csv                                       │
│    - Contains participant metadata (ID, condition, Diagnose, etc.) │
│    - Conditions: ALL, CR, ADK, CRADK, SHAM                         │
│    - Labels: Depressed / Healthy                                   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 2: FILTER BY CONDITION                      │
│                                                                     │
│  For each condition (CR, ADK, CRADK, SHAM, All):                  │
│    - Filter participants by condition                               │
│    - Count: Total participants, Depressed, Healthy                  │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: STRATIFIED SPLIT                        │
│                                                                     │
│  Two-stage stratified split:                                         │
│                                                                     │
│  Stage 1:                                                           │
│    Train (70%) ←→ Temp (30%)                                        │
│    └─ Stratified by 'Diagnose' (Depressed/Healthy)                │
│                                                                     │
│  Stage 2:                                                           │
│    Temp (30%) → Val (10%) + Test (20%)                             │
│    └─ Stratified by 'Diagnose' (Depressed/Healthy)                │
│                                                                     │
│  Result:                                                            │
│    - Train: 70% (balanced Depressed/Healthy)                        │
│    - Validation: 10% (balanced Depressed/Healthy)                  │
│    - Test: 20% (balanced Depressed/Healthy)                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 4: CREATE MANIFEST FILES                    │
│                                                                     │
│  Output Structure:                                                  │
│    d02_manifests/                                                   │
│    ├── All/                                                         │
│    │   ├── All_train.csv                                           │
│    │   ├── All_validation.csv                                      │
│    │   └── All_test.csv                                            │
│    ├── CR/                                                          │
│    │   ├── CR_train.csv                                            │
│    │   ├── CR_validation.csv                                       │
│    │   └── CR_test.csv                                             │
│    ├── ADK/                                                         │
│    │   └── ...                                                     │
│    └── ...                                                          │
│                                                                     │
│  Each CSV contains:                                                 │
│    - Participant ID, condition, Diagnose                           │
│    - Other metadata columns                                        │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 5: ADD FILE PATHS (Optional)               │
│                                                                     │
│  For each manifest CSV:                                             │
│    1. Scan NPZ folder for video files                               │
│    2. Extract phase from filename:                                  │
│       - training1_pos / training2_pos → training_pos                │
│       - training1_neg / training2_neg → training_neg               │
│       - induction1 → induction1                                    │
│       - induction2 → induction2                                   │
│    3. Match audio CSV files by phase and Aufgabe                    │
│    4. Expand rows: 1 participant → N files (one row per file)      │
│                                                                     │
│  New columns added:                                                 │
│    - path_video: Path to .npz video feature file                    │
│    - path_audio: Path to .csv audio feature file                   │
│    - phase: training_pos, training_neg, induction1, induction2    │
│    - Aufgabe: Task number (1, 2, 3, ...)                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 6: VERIFY NO DATA LEAKAGE                   │
│                                                                     │
│  Checks:                                                            │
│    ✓ Train ∩ Validation = ∅                                        │
│    ✓ Train ∩ Test = ∅                                              │
│    ✓ Validation ∩ Test = ∅                                         │
│                                                                     │
│  Each participant appears in ONLY ONE split!                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

**1. Stratified Splitting**
- Maintains class balance (Depressed/Healthy) in each split
- Uses `sklearn.model_selection.train_test_split` with `stratify` parameter
- Ensures similar distribution of labels across train/val/test

**2. Condition-Based Splits**
- Creates separate splits for each condition (CR, ADK, CRADK, SHAM)
- Also creates "All" condition with all participants combined
- Allows condition-specific experiments

**3. Phase Extraction**
- Automatically extracts phase from NPZ filenames:
  - `training1_pos_aufgabe_10.npz` → `training_pos`, Aufgabe=10
  - `induction1.npz` → `induction1`, Aufgabe=1
- Uses `training_assignments.csv` to resolve generic training1/training2

**4. File Path Matching**
- Matches video NPZ files with corresponding audio CSV files
- For inductions: Matches by timestamp (earlier = induction1, later = induction2)
- For trainings: Matches by Training number and Aufgabe number

**5. Data Leakage Prevention**
- Participant-level splitting (not file-level)
- Each participant ID appears in only one split
- Assertions verify no overlap between splits

### Example Manifest File Structure

```csv
ID,condition,Diagnose,path_video,path_audio,phase,Aufgabe
4,CR,Depressed,/path/to/4_10_training1_neg_aufgabe_10.npz,/path/to/Training_1_Aufgabe_10.csv,training_neg,10
4,CR,Depressed,/path/to/4_11_training1_neg_aufgabe_11.npz,/path/to/Training_1_Aufgabe_11.csv,training_neg,11
4,CR,Depressed,/path/to/4_induction1.npz,/path/to/Belastungsphase_2023-01-15_10-30.csv,induction1,1
...
```

### Usage in Training

The manifest files are used by `get_eks_dataloader.py` to:
1. Load participants for train/validation/test splits
2. Filter by condition (if specified)
3. Filter by phase (if specified)
4. Load corresponding video and audio features
5. Create PyTorch DataLoader with proper batching

---

## Key Design Decisions

| Decision | Options | Recommendation | Rationale |
|----------|---------|----------------|-----------|
| **Fusion Strategy** | ia / lt / it / MT / add / multi / concat | **ia** (Intermediate Attention) | Learns cross-modal alignment, interpretable |
| **Audio Model** | AudioTransformer / Custom | AudioTransformer (AudioSet pretrained) | Leverages large-scale pretraining |
| **Video Model** | VisualMAE / Custom | VisualMAE (pretrained) | Strong visual encoder |
| **Loss Function** | BCE / Weighted BCE / Focal | CombinedLoss (Focal + L2) | Handles class imbalance + regularization |
| **Optimizer** | Adam / AdamW / SGD | AdamW | Better generalization with weight decay |
| **Batch Size** | 8 / 16 / 32 | 16 | Balance memory and gradient stability |
| **Learning Rate** | 1e-5 / 1e-4 / 1e-3 | 1e-5 | Standard for pretrained models |
| **Dropout** | 0.3 / 0.5 | 0.5 (fusion) | Prevent overfitting in fusion layers |

## Troubleshooting Guide

### Issue 1: Model overfits (high train acc, low val acc)

**Solutions**:
- ✅ Increase dropout (0.5 → 0.7)
- ✅ Add L2 regularization (weight_decay=1e-3)
- ✅ Use early stopping (patience=5)
- ✅ Reduce model capacity (fewer transformer layers)

### Issue 2: Model underfits (low train acc)

**Solutions**:
- ✅ Decrease dropout (0.5 → 0.3)
- ✅ Increase learning rate (1e-5 → 1e-4)
- ✅ Train longer (more epochs)
- ✅ Check if pretrained models are loading correctly

### Issue 3: Audio and video features misaligned

**Solutions**:
- ✅ Check temporal alignment in data preprocessing
- ✅ Use padding masks correctly
- ✅ Verify feature dimensions match expected sizes
- ✅ Check if sequence lengths are reasonable

### Issue 4: Fusion not improving over single modality

**Analysis**:
- ❓ Are audio and video features complementary? Check correlation
- ❓ Is fusion mechanism learning? Visualize attention weights
- ❓ Try different fusion strategies (ia vs MT vs lt)
- ❓ Check if modalities are synchronized temporally

---

## File Structure

```
MMFformer_/
├── models/
│   ├── MultiModalDepDet.py      # Main model architecture
│   ├── Generate_Audio_Model.py  # AudioTransformer wrapper
│   ├── Generate_Visual_Model.py # VisualMAE wrapper
│   ├── mutualtransformer.py     # MT fusion mechanism
│   └── transformer_timm.py      # Attention blocks
├── datasets_process/
│   └── get_eks_dataloader.py    # d02 dataset loader
├── train_val/
│   ├── train_val.py              # Training/validation loops
│   ├── losses.py                 # Loss functions
│   └── plotting.py               # Visualization utilities
├── scripts/
│   ├── main.py                   # Main training script
│   └── mainkfold.py              # K-fold CV script
├── configs/
│   └── config.yaml               # Configuration file
├── results/                       # Saved results
└── runlog_*/                     # Per-run logs and plots
```

---

## Complete File Reference

This section lists all Python files in the MMFformer_ directory and explains what each file does.

### 📁 Root Directory

| File | Purpose |
|------|---------|
| **WTF.py** | Utility script for file management (e.g., deleting files with specific suffixes from d02_npy folder) |

### 📁 scripts/ - Main Training Scripts

| File | Purpose |
|------|---------|
| **main.py** | Main training script for single train/val/test split. Handles model initialization, training loop, validation, testing, and result saving. Supports d02, dvlog, and lmvd datasets. |
| **mainkfold.py** | K-fold cross-validation training script. Runs training across multiple folds and aggregates results. Useful for robust evaluation. |
| **compute_results.py** | Utility script to compute mean and standard deviation of metrics across multiple folds. Reads result files and calculates statistics (Accuracy, Precision, Recall, F1, etc.). |

### 📁 models/ - Model Architectures

| File | Purpose |
|------|---------|
| **MultiModalDepDet.py** | Main multi-modal depression detection model. Implements audio-visual fusion with multiple fusion strategies (ia, lt, it, MT, add, multi, concat, tensor_fusion). |
| **DepMamba.py** | Alternative model architecture using Mamba (state-space model) for depression detection. Requires mamba_ssm package. |
| **Generate_Audio_Model.py** | Wrapper for AudioTransformer model. Loads pretrained AudioSet models for audio feature extraction. |
| **Generate_Visual_Model.py** | Wrapper for VisualMAE model. Loads pretrained visual encoders for video feature extraction. |
| **mutualtransformer.py** | Implements Mutual Transformer (MT) fusion mechanism. Performs bidirectional cross-attention between audio and video modalities. |
| **transformer_timm.py** | Contains attention blocks and transformer components (AttentionBlock, Attention classes) used in fusion mechanisms. |
| **base.py** | Base class for all models. Provides common interface and utility methods. |
| **__init__.py** | Module initialization. Exports MultiModalDepDet and lazy-loads DepMamba (only when needed). |

### 📁 models/dfer/ - VisualMAE Components

| File | Purpose |
|------|---------|
| **Temporal_Model.py** | Temporal modeling components for visual features. |
| **VisualMAE/visual_models_vit.py** | Vision Transformer (ViT) implementation for visual feature extraction. |
| **VisualMAE/audio_models_vit.py** | Audio Vision Transformer implementation. |
| **VisualMAE/util/misc.py** | Miscellaneous utility functions for VisualMAE. |
| **VisualMAE/util/stat.py** | Statistical utility functions. |
| **VisualMAE/util/patch_embed.py** | Patch embedding utilities for ViT. |
| **VisualMAE/util/pos_embed.py** | Positional embedding utilities. |
| **VisualMAE/util/lr_sched.py** | Learning rate scheduling utilities. |
| **VisualMAE/util/lr_decay.py** | Learning rate decay utilities. |
| **VisualMAE/util/datasets.py** | Dataset utilities for VisualMAE. |
| **VisualMAE/util/lars.py** | LARS (Layer-wise Adaptive Rate Scaling) optimizer utilities. |
| **VisualMAE/util/crop.py** | Image cropping utilities. |

### 📁 models/mamba/ - Mamba Model Components

| File | Purpose |
|------|---------|
| **bimamba.py** | Bidirectional Mamba implementation. |
| **mm_bimamba.py** | Multi-modal bidirectional Mamba. |
| **mamba_blocks.py** | Mamba block implementations. |
| **selective_scan_interface.py** | Interface for selective scan operations (core Mamba operation). |
| **__init__.py** | Module initialization for Mamba components. |

### 📁 datasets_process/ - Data Loading and Processing

| File | Purpose |
|------|---------|
| **EKSpression.py** | Main dataloader for d02 (EKS) dataset. Implements `get_eks_dataloader()` function that loads data with condition/phase/modality filtering. Handles NPZ video files and CSV audio files. |
| **EKSpression_prepare_labels.py** | Prepares labels for EKS dataset. Processes metadata and creates label files. |
| **EKSpression_extract_npy.py** | Extracts and processes NPY files for EKS dataset. Converts raw data to format needed for training. |
| **dvlog.py** | Dataloader for DVlog dataset. Implements `get_dvlog_dataloader()` function. Handles train/valid/test splits and gender filtering. |
| **lmvd.py** | Dataloader for LMVD dataset. Implements `get_lmvd_dataloader()` function. Similar structure to dvlog.py. |
| **lmvd_prepare_labels.py** | Prepares labels for LMVD dataset. Creates train/valid/test splits with stratification. |
| **lmvd_extract_npy.py** | Extracts visual features from LMVD dataset and saves as NPY files. |
| **__init__.py** | Module initialization. Exports all dataloader functions (get_dvlog_dataloader, get_lmvd_dataloader, get_eks_dataloader) and collate functions. |

### 📁 train_val/ - Training and Validation

| File | Purpose |
|------|---------|
| **train_val.py** | Core training and validation functions. Contains `train_epoch()` and `val()` functions that handle forward pass, loss computation, and metric calculation. |
| **losses.py** | Loss function implementations. Contains `CombinedLoss` class that combines Focal Loss and L2 regularization for handling class imbalance. |
| **plotting.py** | Visualization utilities. Functions for plotting training curves, confusion matrices, and data statistics. Used for generating plots saved in runlog directories. |
| **utils.py** | Training utilities. Contains helper functions like `EarlyStopping`, `adjust_learning_rate`, `LOG_INFO` for colored console output, etc. |

### 📁 d02_manifests/ - Dataset Splitting

| File | Purpose |
|------|---------|
| **d02_manifests.py** | Creates train/validation/test manifest files for d02 dataset. Performs stratified splitting (70/10/20), maintains class balance, prevents data leakage, and adds file paths. Generates CSV files for each condition (All, CR, ADK, CRADK, SHAM) with participant metadata and file paths. |

### 📊 Experiments
- 5 conditions × 6 phases × 3 modalities = 90 possible combinations
- Train/Val/Test splits (or K-fold CV)
- Compare fusion strategies: ia, lt, it, MT
- Compare with single-modality baselines
