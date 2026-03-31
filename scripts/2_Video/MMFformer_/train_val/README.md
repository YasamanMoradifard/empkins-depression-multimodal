# Metrics Explanation: Class-Specific, Weighted, and Unweighted Averages

This document explains the three types of metrics calculated in the validation function: **class-specific metrics**, **weighted averages**, and **unweighted averages**.

## Overview

For binary classification (classes 0 and 1), the code calculates metrics in three different ways to provide a comprehensive view of model performance, especially when dealing with imbalanced datasets.

---

## 1. Class-Specific Metrics

**What they are:** Metrics calculated separately for each class (0 and 1).

**How they're calculated:**
- For each class, the code computes:
  - **Precision**: `TP / (TP + FP)` - Of all predictions for this class, how many were correct?
  - **Recall**: `TP / (TP + FN)` - Of all true instances of this class, how many did we find?
  - **F1-Score**: `2 × (Precision × Recall) / (Precision + Recall)` - Harmonic mean of precision and recall
  - **Accuracy** (class-specific): `TP / (TP + FN)` - Same as recall for this class

**Example:**
```
Class 0: Precision = 0.85, Recall = 0.90, F1 = 0.87
Class 1: Precision = 0.75, Recall = 0.60, F1 = 0.67
```

**When to use:**
- When you need to understand performance for each class individually
- When classes have different importance or different costs of misclassification
- To identify which class the model struggles with more

**Key insight:** These metrics show you the raw performance for each class without any averaging.

---

## 2. Weighted Average Metrics

**What they are:** Averages of class-specific metrics, weighted by the number of samples in each class.

**How they're calculated:**
```
Weighted_Metric = Σ(Class_Metric[i] × Class_Count[i]) / Total_Samples
```

For example:
```
Weighted_Precision = (Precision_0 × Count_0 + Precision_1 × Count_1) / Total_Samples
```

**Example:**
If you have:
- Class 0: 800 samples, Precision = 0.85
- Class 1: 200 samples, Precision = 0.75

Then:
```
Weighted_Precision = (0.85 × 800 + 0.75 × 200) / 1000
                   = (680 + 150) / 1000
                   = 0.83
```

**When to use:**
- **For imbalanced datasets** - This gives more weight to the majority class
- When you want metrics that reflect the actual distribution of your data
- When the class distribution in your dataset matches the real-world distribution
- **Most common use case** - Often used as the primary metric in imbalanced classification

**Key insight:** The majority class has more influence on the final metric. If Class 0 has 90% of samples and Class 1 has 10%, the weighted metric will be heavily influenced by Class 0's performance.

---

## 3. Unweighted Average Metrics

**What they are:** Simple arithmetic averages of class-specific metrics, treating each class equally regardless of sample count.

**How they're calculated:**
```
Unweighted_Metric = Σ(Class_Metric[i]) / Number_of_Classes
```

For example:
```
Unweighted_Precision = (Precision_0 + Precision_1) / 2
```

**Example:**
If you have:
- Class 0: Precision = 0.85
- Class 1: Precision = 0.75

Then:
```
Unweighted_Precision = (0.85 + 0.75) / 2
                     = 0.80
```

**When to use:**
- **For balanced evaluation** - When you want to treat all classes equally
- When classes have equal importance regardless of their frequency
- When you want to ensure the model performs well on both classes, not just the majority class
- **Important for imbalanced datasets** - This metric won't be skewed by class imbalance

**Key insight:** Each class contributes equally to the final metric. This is particularly useful when you have imbalanced data but want to ensure good performance on the minority class.

---

## Comparison Example

Let's say you have an imbalanced dataset:
- **Class 0**: 900 samples (90%)
- **Class 1**: 100 samples (10%)

And your model performs:
- **Class 0**: Precision = 0.95, Recall = 0.90, F1 = 0.92
- **Class 1**: Precision = 0.60, Recall = 0.50, F1 = 0.55

### Results:

**Class-Specific:**
- Class 0 F1: 0.92
- Class 1 F1: 0.55

**Weighted F1:**
```
(0.92 × 900 + 0.55 × 100) / 1000 = 0.883
```
- Heavily influenced by Class 0 (majority class)
- Looks good because Class 0 performs well

**Unweighted F1:**
```
(0.92 + 0.55) / 2 = 0.735
```
- Treats both classes equally
- Reveals that Class 1 (minority class) performance is poor

---

## Which Metric Should You Use?

### Use **Class-Specific Metrics** when:
- You need detailed performance breakdown per class
- Different classes have different business importance
- You're debugging which class is causing issues

### Use **Weighted Averages** when:
- Your dataset is imbalanced and you want metrics that reflect real-world distribution
- The majority class performance is more important
- You're comparing models on the same imbalanced dataset

### Use **Unweighted Averages** when:
- You want to ensure balanced performance across all classes
- Classes have equal importance regardless of frequency
- You're working with imbalanced data but want to avoid being misled by majority class dominance
- **This is often preferred for imbalanced datasets** to ensure minority class performance is considered

---

## Important Notes

1. **Binary Classification Context**: In this code, we have two classes (0 and 1), but the same principles apply to multi-class problems.

2. **Class-Specific Accuracy**: In this implementation, `class_accuracy` is calculated as `TP / (TP + FN)`, which is actually the same as recall for that class. This measures how well the model identifies samples of that specific class.

3. **Overall Accuracy**: The `accuracy` metric (calculated as `(TP + TN) / Total_Samples`) is different from weighted/unweighted averages - it's the overall classification accuracy across all samples.

4. **Imbalanced Datasets**: When dealing with imbalanced data, **unweighted averages are often more informative** because they prevent the majority class from masking poor performance on the minority class.

---

## Summary Table

| Metric Type | Calculation | Best For |
|------------|-------------|----------|
| **Class-Specific** | Individual metrics per class | Understanding per-class performance |
| **Weighted Average** | `Σ(Metric_i × Count_i) / Total` | Imbalanced datasets, real-world distribution |
| **Unweighted Average** | `Σ(Metric_i) / Num_Classes` | Balanced evaluation, equal class importance |

---

## Code Reference

The metrics are calculated in the `val()` function in `train_val.py`:
- **Lines 217-239**: Class-specific metrics calculation
- **Lines 243-246**: Weighted average metrics
- **Lines 248-251**: Unweighted average metrics

