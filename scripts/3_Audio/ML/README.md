# Audio ML Classification Pipeline

**Depression Detection from Audio Features**

This pipeline classifies audio recordings as **healthy (0)** vs **depressed (1)** using OpenSmile-extracted acoustic features.

---

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIO ML CLASSIFICATION PIPELINE                     │
│                    (Depression Detection from Audio Features)                │
└─────────────────────────────────────────────────────────────────────────────┘

                                    INPUT
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING                                                             │
│     • Load OpenSmile CSVs (frame-level audio features)                       │
│     • Filter by condition (CR/CRADK/ADK/SHAM) & phase (training_pos/neg)     │
│     • Parse labels from filenames: healthy (0) vs depressed (1)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. FEATURE AGGREGATION                                                      │
│     • Group by: (Participant ID, condition, phase, aufgabe)                  │
│     • Compute 19 statistics per feature:                                     │
│       mean, std, min, max, skew, kurtosis, entropy, rate_of_change,         │
│       peaks_count, median, percentiles, IQR, slope, intercept, etc.          │
│     • Result: 1 row per participant/aufgabe (~multiple rows per person)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DATA CLEANING                                                            │
│     • Remove constant columns (zero variance)                                │
│     • Sanitize feature names (XGBoost compatibility)                         │
│     • Keep only numeric features                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. OUTER CROSS-VALIDATION LOOP                                              │
│     • GroupKFold (default) or StratifiedKFold                                │
│     • GroupKFold ensures same participant never in both train & test         │
│     • k_fold = 3 (simple CV) or k_outer_fold = 5 (nested CV)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
     ┌─────────────────┐                           ┌─────────────────┐
     │   OUTER TRAIN   │                           │   OUTER TEST    │
     │   (X_train)     │                           │   (X_test)      │
     └─────────────────┘                           └─────────────────┘
              │                                               │
              ▼                                               │
┌─────────────────────────────────────────────────┐           │
│  5. PREPROCESSING (for Feature Selection only)  │           │
│     • SimpleImputer(median) - fit on train      │           │
│     • Scaler (minmax/standard/robust)           │           │
│     → X_train_norm, X_test_norm                 │           │
└─────────────────────────────────────────────────┘           │
              │                                               │
              ▼                                               │
┌─────────────────────────────────────────────────┐           │
│  6. FEATURE SELECTION (if difficulty ≥ 3)       │           │
│     Methods:                                     │           │
│     • skbest: SelectKBest(ANOVA F-test)         │           │
│     • rfe: Recursive Feature Elimination         │           │
│     • combined: Mann-Whitney → RFE               │           │
│                                                  │           │
│     Internal CV with Pipeline to find best k    │           │
│     → Returns: feat_names (selected features)   │           │
└─────────────────────────────────────────────────┘           │
              │                                               │
              ▼                                               │
       Select columns                                         │
       X_train_raw_sel = X_train[feat_names]                  │
       X_test_raw_sel = X_test[feat_names] ◄──────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. MODEL TRAINING (with or without HPO)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IF HPO (difficulty 5-6):                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Pipeline:                                                          │     │
│  │    1. SimpleImputer(median)                                        │     │
│  │    2. Scaler (minmax/standard/robust)                              │     │
│  │    3. Classifier                                                   │     │
│  │                                                                    │     │
│  │  GridSearchCV / RandomizedSearchCV:                                │     │
│  │    • Inner CV: GroupKFold (k_inner_fold = 3)                      │     │
│  │    • Scoring: f1_weighted                                          │     │
│  │    • Returns: best_model (Pipeline with best params)               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  IF NO HPO (difficulty 1-4):                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Create Pipeline directly:                                          │     │
│  │    1. SimpleImputer(median)                                        │     │
│  │    2. Scaler                                                       │     │
│  │    3. Classifier (with default or regularized params)              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Final Training:                                                             │
│    best_model.fit(X_train_raw_sel, y_train)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. EVALUATION                                                               │
│     • Predict on X_test_raw_sel                                              │
│     • Metrics: Accuracy, Precision, Recall, Specificity, F1 (multiple),     │
│       ROC-AUC, Confusion Matrix                                              │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  9. AGGREGATE RESULTS                                                        │
│     • Mean ± std across all outer folds                                      │
│     • Save: CSV results, confusion matrices, best hyperparameters,           │
│       selected features                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Classifiers Used

| Classifier | Regularized Version (difficulty ≥ 2) |
|------------|--------------------------------------|
| Logistic Regression | C=0.1, class_weight="balanced" |
| SVC | C=0.1, kernel="linear", class_weight="balanced" |
| Random Forest | max_depth=5, n_estimators=200 |
| AdaBoost | learning_rate=0.5, n_estimators=100 |
| Decision Tree | max_depth=4, min_samples_leaf=2 |
| KNN | n_neighbors=7, weights="distance" |
| XGBoost | max_depth=3, learning_rate=0.05 |

---

## 🔧 Difficulty Levels

| Level | Feature Selection | Regularization | HPO | Nested CV |
|-------|-------------------|----------------|-----|-----------|
| 1 | ❌ | ❌ | ❌ | ❌ |
| 2 | ❌ | ✅ | ❌ | ❌ |
| 3 | ✅ | ❌ | ❌ | ❌ |
| 4 | ✅ | ✅ | ❌ | ❌ |
| 5 | ✅ | ✅ | ✅ | ❌ |
| 6 | ✅ | ✅ | ✅ | ✅ |

---

## 🔄 Feature Selection Methods

### **skbest** (default)
```
For k in 1..50:
    Pipeline: SelectKBest(k features) → Model
    Evaluate with GroupKFold CV
→ Return k with best F1 score
```

### **rfe**
```
For n in 1..15:
    Pipeline: RFE(n features) → Model
    Evaluate with KFold CV
→ Return n with best F1 score
```

### **combined**
```
1. Mann-Whitney U test → top 50 features by p-value
2. RFE on those 50 features → find best subset
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `*_test_results.csv` | Per-fold metrics |
| `*_summary_results.csv` | Mean/std across folds |
| `*_confusion_matrix.png` | Confusion matrix plots |
| `*_best_hyperparameters.json` | Best HPO params (if HPO enabled) |
| `*_selected_features.csv` | Selected feature statistics |
| `*_args.json` | Run configuration |

---

## 🎓 Key Design Decisions

1. **GroupKFold** ensures participant-level independence (no same person in train & test)
2. **Pipeline wraps preprocessing** so imputation/scaling happens correctly inside CV
3. **Multiple normalization methods** can be compared (minmax, standard, robust)
4. **Binary classification**: healthy (0) vs depressed (1)
5. **Primary metric**: F1-weighted score

---

## 🛡️ Data Leakage Prevention

The pipeline implements several safeguards against data leakage:

### Leakage A: Group-Aware Inner CV
- Inner HPO uses `GroupKFold` with participant groups
- Ensures samples from the same participant stay together in inner folds

### Leakage B: Preprocessing Inside Pipeline
- Imputation and scaling wrapped in sklearn `Pipeline`
- For each inner/outer fold, statistics computed only from training data

### Leakage C: Feature Selection Inside CV
- Feature selector wrapped in `Pipeline` with model
- `cross_val_score(Pipeline)` ensures selection happens per fold

---

## 🚀 Usage

```bash
python Audio_nested_CV_skbest.py \
    --condition CR \
    --phase training_pos \
    --feature_selection skbest \
    --cv_splitter groupkfold \
    --difficulty_level 5 \
    --k_fold 3 \
    --k_inner_fold 3 \
    --normalization minmax \
    --search_method random \
    --n_iter 50 \
    --opensmile_data_dir /path/to/opensmile/csvs
```

Or use the shell script:
```bash
./Audio_ML.sh
```

---

## 📊 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1 Score**: Harmonic mean of precision and recall
- **F1 Weighted**: F1 weighted by class support
- **F1 Macro**: Unweighted mean of F1 per class
- **ROC-AUC**: Area under the ROC curve

---

## 📋 Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn (optional)
- xgboost (optional)

