# empkins-depression-multimodal

Depression detection from physiological and behavioral signals using multimodal deep learning and fusion architectures.

This project implements a complete pipeline, from raw signal processing through feature extraction, unimodal classification, and multimodal fusion — across six data modalities: ECG, RSP, EMG, Video, Audio, and Text. The study uses data from 259 RCT participants (132 healthy, 127 depressed) across four experimental conditions (CR, CRADK, ADK, SHAM).

**M.Sc. Thesis** · Machine Learning and Data Analytics Lab (MaD Lab) · Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
Supervised by M.Sc. Misha Sadeghi, M.Sc. Mahdis Habibpour, and Prof. Dr. Björn M. Eskofier 


## Dataset

All experiments use the **EmpkinS-EKSpression** dataset, a four-arm single-blind Randomized Controlled Trial developed at the EmpkinS Collaborative Research Center at FAU Erlangen-Nürnberg. Participants were either clinically diagnosed with a depressive disorder (ICD-10/SCID-5-CV) or matched healthy controls. The binary classification target is the diagnostic label; the regression target is the PHQ-9 severity score (0–27).

Raw data is not included in this repository. See `data/README.md` for cohort details: participant counts per condition, modality availability, and known missing data.

Dataset paper: https://link.springer.com/article/10.1186/s12888-024-06361-3
MaD Lab: https://www.mad.tf.fau.de/


## Repository Structure

```
empkins-depression-multimodal/
├── data/
│   └── README.md                               # Cohort statistics, modality availability, missing participants
└── scripts/
    ├── 1_BioSignal/                            # ECG, RSP, EMG processing and classification
    │   ├── data_exploration_notebook/           # EDA notebooks (one per signal type)
    │   └── code/
    │       ├── data_processing/                 # Raw to preprocessed CSV, signal processing modules
    │       └── ML_classification/              # Nested CV sklearn/XGBoost pipeline
    ├── 2_Video/                                # Video-based depression detection
    │   ├── ML/                                 # OpenDBM tabular features + nested CV
    │   └── MMFformer_/                         # Transformer/Mamba deep learning architecture
    │       ├── configs/config.yaml
    │       ├── scripts/
    │       ├── models/
    │       ├── datasets_process/
    │       ├── train_val/
    │       └── results/
    ├── 3_Audio/                                # Audio-based depression detection
    │   ├── Data/                               # OpenSMILE feature preparation
    │   ├── ML/                                 # Tabular ML pipeline (sklearn/XGBoost, nested CV)
    │   └── LSTM/                               # Hierarchical LSTM on audio sequences
    │       ├── main.py
    │       ├── data/
    │       ├── models/
    │       ├── training/
    │       └── utils/
    ├── 4_Text/                                 # Text-based depression detection
    │   ├── data_exploration.ipynb
    │   └── Text_ML.py
    └── 5_MultiModal/                           # Multimodal fusion experiments
        ├── MultiModal_Early_Classification.py
        ├── MultiModal_Late_Classification.py
        ├── MultiModal_Early_Regression.py
        ├── MultiModal_Late_Regression.py
        ├── report.py
        └── MultiModal.sh
```


## Methods

### Physiological signals (ECG, RSP, EMG)

Features are extracted per modality using NeuroKit2: HRV time- and frequency-domain metrics and nonlinear measures for ECG; breathing rate, amplitude, and variability for RSP; spectral power, RMS, and zero-crossing rate per facial muscle channel for EMG. Each signal is segmented by experimental phase and summarized with aggregation statistics. A nested cross-validation pipeline (5 outer / 3 inner GroupKFold folds) with Mann-Whitney U feature selection trains classifiers including Random Forest, AdaBoost, SVC-RBF, XGBoost, KNN, Logistic Regression, and Decision Tree.

### Video

Tabular features come from OpenDBM (facial expressivity, asymmetry, landmark coordinates) and OpenFace 2.0 (facial action units, head pose, gaze), yielding 4,632 features per task after aggregation. The same nested CV pipeline is applied for traditional ML. For deep learning, the MMFformer transformer architecture is adapted for single-modality video classification by discarding its audio branch and fusion modules.

MMFformer paper: https://arxiv.org/abs/2508.06701
MMFformer repository: https://github.com/rezwanh001/Large-Scale-Multimodal-Depression-Detection

### Audio

OpenSMILE extracts acoustic features covering prosody, MFCCs, spectral properties, and voice quality, each summarized with 19 aggregation statistics. Traditional ML follows the same nested CV pipeline used for other modalities. For deep learning, a hierarchical Bidirectional LSTM processes frame-level features through stacked BiLSTM layers with intra- and inter-segment attention, capturing both local and global temporal dynamics of speech.

### Text

Clinical interview transcripts are encoded using DepRoBERTa, a RoBERTa model fine-tuned for depression detection, producing 3 depression probability scores and 11 question-based features per participant. These 14 features are passed to the same set of ML classifiers used across the other modalities.


### Multimodal fusion

Both fusion strategies combine Audio, Video, ECG, EMG, and RSP features, optionally including Text, evaluated by participant ID (all phases pooled) or by individual experimental phase.

Early fusion concatenates features from all modalities into a single vector, applies joint Mann-Whitney U feature selection, and trains a single classifier end-to-end on the combined representation.

Late fusion trains independent classifiers per modality and combines their output probabilities using performance-weighted averaging. Weights are derived from out-of-fold F1 scores with a quadratic proportional scheme, so a modality with twice the F1 receives four times the fusion weight.


## Results

### Single-modality classification

| Modality | Best model | F1-weighted | ROC-AUC |
|----------|-----------|:-----------:|:-------:|
| Text | SVC-RBF | 0.87 +/- 0.08 | - |
| EMG | AdaBoost | 0.85 +/- 0.17 | 0.87 +/- 0.15 |
| ECG | Random Forest | 0.76 +/- 0.08 | 0.84 +/- 0.17 |
| Audio (ML) | SVC-RBF | 0.73 +/- 0.13 | 0.72 +/- 0.20 |
| RSP | Decision Tree | 0.73 +/- 0.09 | 0.71 +/- 0.17 |
| Video (ML) | Logistic Regression | 0.64 +/- 0.10 | 0.67 +/- 0.06 |
| Video (MMFformer) | - | 0.58 +/- 0.11 | - |
| Audio (LSTM) | - | 0.51 +/- 0.36 | - |

Text is the strongest single modality. Among physiological signals, EMG achieves the highest F1 while ECG yields the best ROC-AUC. Emotionally engaging phases (induction, training) consistently outperform resting phases across all modalities.

### Effect of adding text to multimodal fusion

| Fusion | Aggregation | F1 with text | F1 without text |
|--------|------------|:------------:|:---------------:|
| Early | by ID | 0.79 +/- 0.08 | 0.64 +/- 0.08 |
| Late | by ID | 0.86 +/- 0.06 | 0.62 +/- 0.16 |
| Early | by phase | 0.86 +/- 0.04 | 0.73 +/- 0.28 |
| Late | by phase | 0.90 +/- 0.08 | 0.69 +/- 0.06 |

Removing text reduces F1 by 13 to 24 percentage points across all settings.

### Best multimodal classification results (with text, by phase)

| Fusion | Condition-Phase | Model | F1-weighted | ROC-AUC |
|--------|----------------|-------|:-----------:|:-------:|
| Late | ADK_training_neg (n=56) | Random Forest | 0.90 +/- 0.08 | 0.94 +/- 0.04 |
| Early | all_induction1 | AdaBoost | 0.86 +/- 0.04 | 0.92 +/- 0.03 |
| Early | all_induction2 | XGBoost | 0.86 +/- 0.10 | 0.87 +/- 0.10 |
| Late | all_all_phases (by ID) | SVC-RBF | 0.86 +/- 0.06 | 0.91 +/- 0.04 |

Note: late fusion on ADK_induction2 reaches F1=0.93 with perfect AUC, but the sample size is only 13 participants and results likely reflect overfitting.

### Severity regression (PHQ-9, with text)

| Fusion | Condition-Phase | Model | MAE | RMSE |
|--------|----------------|-------|:---:|:----:|
| Late | all_induction1 (by phase) | Random Forest | 2.30 +/- 0.38 | 3.01 +/- 0.29 |
| Early | all_induction1 (by phase) | Random Forest | 2.37 +/- 0.44 | 3.14 +/- 0.34 |
| Late | all_all_phases (by ID) | Random Forest | 2.54 +/- 0.56 | 3.31 +/- 0.81 |
| Early | all_all_phases (by ID) | Random Forest | 2.62 +/- 0.58 | 3.47 +/- 0.86 |

Without text, the best MAE rises to approximately 5.1, roughly double the text-included result.


## Installation

Dependencies vary by modality. There is no single root-level requirements file.

For all tabular ML pipelines (BioSignal, Audio ML, Video ML, Text, MultiModal):

```bash
python -m venv venv
source venv/bin/activate
pip install -r scripts/1_BioSignal/code/ML_classification/requirements.txt
```

Core packages: numpy, pandas, scikit-learn, xgboost, neurokit2, biosppy, matplotlib, scipy.

For MMFformer, additionally install:

```bash
pip install torch torchvision torchaudio timm wandb pyyaml speechbrain
```

For the Audio LSTM, PyTorch and standard scientific Python packages are sufficient.


## Running Experiments

All tabular ML scripts follow the same nested cross-validation structure and can be run directly:

```bash
python scripts/1_BioSignal/code/ML_classification/BioSig_ML.py
python scripts/2_Video/ML/Video_ML.py
python scripts/3_Audio/ML/Audio_ML.py
python scripts/4_Text/Text_ML.py
python scripts/5_MultiModal/MultiModal_Early_Classification.py
python scripts/5_MultiModal/MultiModal_Late_Classification.py
```

`BioSig_ML.py` accepts arguments for modality, condition, and phase. See `BioSig_ML.sh` for the full parameter grid.

For MMFformer, set your data paths in `scripts/2_Video/MMFformer_/configs/config.yaml`, then:

```bash
python scripts/2_Video/MMFformer_/scripts/main.py
```

For k-fold evaluation use `mainkfold.py`; for fine-tuning use `MMFformer_finetuning.py`. SLURM wrappers are available in `scripts/2_Video/MMFformer_/scripts/`.

For the Audio LSTM:

```bash
python scripts/3_Audio/LSTM/main.py
```

BioSignal preprocessing runs through the notebooks in `scripts/1_BioSignal/code/data_processing/` in order, or using the standalone processing scripts directly.


## HPC / SLURM

Shell scripts are provided for SLURM-based clusters. Cluster-specific paths (virtualenv locations, data directories) must be updated before use.

| Script | Location |
|--------|----------|
| BioSig_ML.sh | scripts/1_BioSignal/code/ML_classification/ |
| run.sh, run_multiple_phases.sh, run_kfold_multiple_phases.sh | scripts/2_Video/MMFformer_/scripts/ |
| job_run_lstm.sh | scripts/3_Audio/LSTM/ |
| MultiModal.sh | scripts/5_MultiModal/ |


## Additional Documentation

More detailed guides are available within individual modules:

- BioSignal preprocessing: `scripts/1_BioSignal/code/data_processing/README.md`
- BioSignal EDA: `scripts/1_BioSignal/data_exploration_notebook/README.md`
- Audio data preparation: `scripts/3_Audio/Data/README.md`
- Audio ML pipeline: `scripts/3_Audio/ML/README.md`
- Audio LSTM architecture: `scripts/3_Audio/LSTM/LSTM_ARCHITECTURE_GUIDE.md`
- MMFformer overview: `scripts/2_Video/MMFformer_/MMFformer_Visual_Summary.md`
- MMFformer fine-tuning: `scripts/2_Video/MMFformer_/FineTune_Results/HOW_TO_RUN.md`
- Cohort information: `data/README.md`


## Acknowledgements

This work was conducted at the Machine Learning and Data Analytics Lab (MaD Lab) at FAU Erlangen-Nürnberg, supervised by M.Sc. Misha Sadeghi, M.Sc. Mahdis Habibpour, M.Sc. Lydia Rupp, Prof. Dr. Björn M. Eskofier and Prof. Dr. Matthias Berking. The EmpkinS-EKSpression dataset was developed as part of the EmpkinS Collaborative Research Center.