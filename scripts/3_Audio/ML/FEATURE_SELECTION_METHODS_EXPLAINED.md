# Feature Selection Methods Explained

This document explains how each feature selection method works in detail, including the algorithms, statistical tests, and evaluation strategies.

---

## Overview

All feature selection methods are applied **separately in each CV fold** to prevent data leakage. They search for the optimal number of features by testing different feature counts and selecting the one that yields the best cross-validated performance.

---

## Method 1: `skbest` (SelectKBest with ANOVA F-test)

### Algorithm Type
**Univariate filter method** - Evaluates each feature independently using a statistical test.

### How It Works

#### Step 1: Statistical Scoring
- Uses **ANOVA F-test** (`f_classif` from scikit-learn)
- For each feature, computes an F-statistic:
  ```
  F = (Between-group variance) / (Within-group variance)
  ```
- **Intuition:** Features with higher F-scores have larger differences between classes (healthy vs depressed) relative to within-class variation

#### Step 2: Feature Count Search
- Tests feature counts from **1 to min(50, total_features)**
- For each k (e.g., k=1, k=2, ..., k=50):
  1. **Selects top k features** based on F-test scores (highest F-scores = most discriminative)
  2. **Runs 5-fold GroupKFold CV** on the training data with only these k features
  3. **Evaluates F1-weighted score** across all CV folds
  4. Records the average F1 score

#### Step 3: Optimal Selection
- **Selects the k that gives the best average F1-weighted score**
- Returns the feature indices for that optimal k

### Code Flow:
```python
# For k = 1, 2, 3, ..., 50:
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_train, y_train)  # Selects top k features

# Evaluate with 5-fold CV
scores = cross_val_score(model, X_selected, y_train, cv=5, scoring="f1_weighted")
avg_f1 = scores.mean()

# Track best k
if avg_f1 > best_f1:
    best_k = k
    best_features = top k features
```

### Key Characteristics:
- ✅ **Fast** - Only evaluates features statistically, no model training for ranking
- ✅ **Model-agnostic** - Doesn't depend on the specific classifier being used
- ✅ **Univariate** - Evaluates each feature independently (may miss feature interactions)
- ✅ **Uses GroupKFold** - Respects participant groups to prevent leakage
- ⚠️ **May miss interactions** - Two weak features together might be strong, but won't be discovered

### Statistical Test Details:
- **ANOVA F-test:** Compares means between two groups (healthy vs depressed)
- **Null hypothesis:** Feature means are equal across classes
- **Higher F-score:** Stronger evidence that feature distinguishes classes
- **Assumes:** Normal distribution (but robust to violations for large samples)

---

## Method 2: `rfe` (Recursive Feature Elimination)

### Algorithm Type
**Wrapper method** - Uses model performance to rank features, recursively removes least important ones.

### How It Works

#### Step 1: Model Compatibility Check
- Some models don't expose feature importance/coefficients:
  - **SVC:** Switches to `RandomForestClassifier` for RFE
  - **KNeighborsClassifier:** Switches to `DecisionTreeClassifier` for RFE
  - Other models use themselves as the base estimator

#### Step 2: Feature Count Search
- Tests feature counts from **1 to min(15, total_features)**
- For each target count (e.g., 1, 2, 3, ..., 15):
  1. **Runs RFE algorithm:**
     - Starts with all features
     - Trains the model and gets feature importances/coefficients
     - Removes the least important feature
     - Repeats until only `num_features` remain
     - This is done by scikit-learn's `RFE` class
   
  2. **Evaluates with 5-fold KFold CV:**
     - Uses the selected features from RFE
     - Trains the model on each CV fold
     - Computes F1-weighted scores
   
  3. **Records average F1 score** for this feature count

#### Step 3: Optimal Selection
- **Selects the feature count that gives the best average F1-weighted score**
- Returns the actual feature names selected by RFE

### RFE Algorithm Details:
```
RFE works backwards:
1. Start with all N features
2. Train model → get feature importances (coef_ or feature_importances_)
3. Remove least important feature → now have N-1 features
4. Train again → remove least important → N-2 features
5. Continue until you have the target number of features
```

**Example:**
- Target: 5 features
- Start: 1000 features
- Iteration 1: Remove worst → 999 features
- Iteration 2: Remove worst → 998 features
- ...
- Iteration 995: Remove worst → 5 features remaining

### Code Flow:
```python
# For num_features = 1, 2, 3, ..., 15:
selector = RFE(estimator=model, n_features_to_select=num_features)
selector.fit(X_train, y_train)  # Recursively removes features
X_selected = selector.transform(X_train)  # Gets final selected features

# Evaluate with 5-fold CV
scores = cross_val_score(model, X_selected, y_train, cv=5, scoring="f1_weighted")
avg_f1 = scores.mean()

# Track best
if avg_f1 > best_f1:
    best_features = selector-selected features
```

### Key Characteristics:
- ✅ **Model-aware** - Uses actual classifier to determine importance
- ✅ **Captures interactions** - Features are evaluated in context of the model
- ✅ **Wrapper method** - Directly optimizes for model performance
- ⚠️ **Slower** - Must train model multiple times during RFE process
- ⚠️ **More feature counts tested** - Tests fewer values (1-15) but each takes longer
- ⚠️ **Model-dependent** - Results vary by classifier type

### Feature Importance Extraction:
- **LogisticRegression, SVC (linear):** Uses `coef_` (coefficient magnitudes)
- **RandomForest, DecisionTree, AdaBoost, XGBoost:** Uses `feature_importances_`
- **KNN, SVC (non-linear):** Cannot extract importance → uses surrogate model

---

## Method 3: `combined` (Mann-Whitney U Test + RFE)

### Algorithm Type
**Hybrid filter + wrapper method** - Two-stage feature selection.

### How It Works

#### Stage 1: Mann-Whitney U Test (Filter)
**Purpose:** Quickly filter to top 50 most promising features using a non-parametric statistical test.

**Algorithm:**
1. For each feature:
   - Split data by class: `data0 = X[y == 0]` (healthy), `data1 = X[y == 1]` (depressed)
   - Run **Mann-Whitney U test** (two-sided):
     ```
     U = rank-based test statistic
     H0: Distributions are equal
     H1: Distributions differ
     ```
   - Get p-value (lower = more significant difference)
   
2. Rank all features by p-value (smallest = most discriminative)
3. Select **top 50 features** with smallest p-values
4. These 50 features proceed to Stage 2

**Mann-Whitney U Test Details:**
- **Non-parametric** - Doesn't assume normal distribution
- **Ranks-based** - Compares ranks rather than means
- **Good for:** Non-normal data, ordinal data, robust to outliers
- **Interpretation:** Low p-value = feature distributions differ significantly between classes

#### Stage 2: RFE on Filtered Features (Wrapper)
**Purpose:** Further refine from 50 features to optimal count (1-15).

1. Takes the 50 features from Stage 1
2. Runs **RFE** on just these 50 features (same as Method 2)
3. Tests feature counts from 1 to 15
4. Selects the count with best 5-fold CV F1 score

### Complete Flow:
```
All Features (e.g., 1000)
    ↓
Stage 1: Mann-Whitney U Test
    ├─ Test each feature independently
    ├─ Rank by p-value
    └─ Select top 50 features
    ↓
50 Features
    ↓
Stage 2: RFE
    ├─ Test k = 1, 2, 3, ..., 15
    ├─ For each k: RFE selects k features
    ├─ Evaluate with 5-fold CV
    └─ Select best k
    ↓
Final Selected Features (e.g., 8 features)
```

### Code Flow:
```python
# Stage 1: Mann-Whitney filter
filtered_features = mannwhitney_feature_selection(X, y, all_features, top_k=50)
# Returns: 50 feature names with smallest p-values

# Stage 2: RFE on filtered features
X_filtered = X[filtered_features]  # Only 50 features now
rfe_selected = rfe_feature_selection(
    X_filtered, y, model, 
    num_features_range=range(1, 15)  # Tests 1-15 features
)
# Returns: Final selected feature names
```

### Key Characteristics:
- ✅ **Best of both worlds** - Fast filtering + model-aware selection
- ✅ **Handles high-dimensional data** - Reduces 1000s of features to 50 quickly
- ✅ **Non-parametric filter** - Mann-Whitney doesn't assume normality
- ✅ **More robust** - Two-stage approach reduces noise
- ⚠️ **May lose important features** - If a feature is weak alone but strong with others, might be filtered out in Stage 1
- ⚠️ **Computational trade-off** - Slower than skbest, but faster than full RFE on all features

---

## Comparison Table

| Aspect | `skbest` | `rfe` | `combined` |
|--------|----------|-------|------------|
| **Type** | Univariate Filter | Wrapper | Hybrid (Filter + Wrapper) |
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Medium | ⚡⚡ Medium |
| **Feature Range Tested** | 1-50 | 1-15 | 1-15 (after filtering to 50) |
| **Model Dependency** | ❌ Model-agnostic | ✅ Model-specific | ✅ Model-specific |
| **Captures Interactions** | ❌ No | ✅ Yes | ✅ Yes (Stage 2) |
| **Statistical Test** | ANOVA F-test | Model importance | Mann-Whitney + Model importance |
| **CV Used** | GroupKFold (5-fold) | KFold (5-fold) | KFold (5-fold) |
| **Best For** | Quick selection, many features | Model-optimized selection | High-dim data with robustness |

---

## When to Use Each Method

### Use `skbest` when:
- ✅ You have many features (1000+) and need fast selection
- ✅ Features are likely independent/univariate
- ✅ You want model-agnostic feature selection
- ✅ Computational time is limited

### Use `rfe` when:
- ✅ You want features optimized for your specific model
- ✅ Feature interactions might be important
- ✅ You have moderate number of features (< 500)
- ✅ You have time for model training during selection

### Use `combined` when:
- ✅ You have very high-dimensional data (1000s of features)
- ✅ You want robustness (two-stage filtering)
- ✅ Data might not be normally distributed (Mann-Whitney is non-parametric)
- ✅ You want model-optimized selection but need initial filtering

---

## Important Implementation Details

### 1. **Data Leakage Prevention**
All methods are applied **only on training folds** within each CV iteration:
```python
# In each outer CV fold:
X_train, X_test = split_data()
selected_features = feature_selection(X_train, y_train)  # Only on train!
X_test_selected = X_test[selected_features]  # Apply same selection to test
```

### 2. **Group-Aware CV**
- `skbest` uses `GroupKFold` - prevents participant-level leakage
- `rfe` uses `KFold` - simpler split (no group info needed for RFE itself)

### 3. **Feature Range Limits**
- `skbest`: Tests up to 50 features (configurable: `num_features_range`)
- `rfe`: Tests up to 15 features
- `combined`: First filters to 50, then tests 1-15

### 4. **Model Compatibility**
- RFE requires models with `coef_` or `feature_importances_`
- KNN and non-linear SVC automatically switch to DecisionTree/RandomForest for RFE

### 5. **Performance Metric**
- All methods use **F1-weighted score** as the evaluation metric
- Selects feature count that maximizes average F1-weighted across CV folds

---

## Computational Complexity

### `skbest`:
- **Time:** O(n_features × k_max × n_samples)
  - n_features: total features
  - k_max: max features tested (50)
  - n_samples: training samples
- **Approx:** ~50 × 5 CV folds = 250 model evaluations

### `rfe`:
- **Time:** O(n_features × k_max × model_train_time)
  - RFE itself: trains model ~(n_features - k) times per k
  - Then 5 CV folds for evaluation
- **Approx:** For k=15: ~15 RFE runs × ~1000 feature removals × 5 CV = expensive!

### `combined`:
- **Time:** O(n_features × n_samples) + O(50 × 15 × model_train_time)
  - Stage 1: Fast (just statistical tests)
  - Stage 2: RFE on only 50 features
- **Approx:** Much faster than full RFE, slower than skbest

---

## Example Execution Flow

### Example: 1000 features, selecting 8 features

**`skbest`:**
```
1. Compute F-scores for all 1000 features (fast)
2. Test k=1: Select top 1 feature → CV → F1=0.65
3. Test k=2: Select top 2 features → CV → F1=0.72
...
8. Test k=8: Select top 8 features → CV → F1=0.85 ← Best!
9. Test k=9: Select top 9 features → CV → F1=0.84
...
50. Test k=50: Select top 50 features → CV → F1=0.82
Result: 8 features selected (top 8 by F-score)
```

**`rfe`:**
```
1. Test num_features=1: RFE removes 999 features → CV → F1=0.60
2. Test num_features=2: RFE removes 998 features → CV → F1=0.70
...
8. Test num_features=8: RFE removes 992 features → CV → F1=0.88 ← Best!
...
15. Test num_features=15: RFE removes 985 features → CV → F1=0.87
Result: 8 features selected (specific 8 chosen by RFE)
```

**`combined`:**
```
Stage 1: Mann-Whitney
  1. Test all 1000 features → get p-values
  2. Select top 50 features (smallest p-values)
  
Stage 2: RFE on 50 features
  1. Test num_features=1: RFE removes 49 features → CV → F1=0.62
  2. Test num_features=2: RFE removes 48 features → CV → F1=0.75
  ...
  8. Test num_features=8: RFE removes 42 features → CV → F1=0.86 ← Best!
  ...
  15. Test num_features=15: RFE removes 35 features → CV → F1=0.85
  
Result: 8 features selected (from the pre-filtered 50)
```

---

## Output Files

After running feature selection, you get:
- `*_selected_features.csv`: Contains statistics for each selected feature:
  - Feature name
  - Min, max, mean, std
  - Correlation with label
  - Normalization method
  - Model name

This helps you understand which features are most important for each model!
