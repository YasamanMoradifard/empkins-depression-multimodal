# Data Exploration

Exploratory notebooks and quick scripts to understand the dataset **before** modeling.

**Goal:** verify data quality, understand distributions, spot leakage risks, and document assumptions.

---

## What’s inside
```
- `BioSig_Data_Exploration_ECG.ipynb`
- `BioSig_Data_Exploration_EMG_Inc.ipynb`
- `BioSig_Data_Exploration_EMG_Exc.ipynb`
- `BioSig_Data_Exploration_RSP.ipynb`
- `figures/`  ← exported plots (PNG/SVG)
- `tables/`   ← exported CSV summaries (class balance, missingness, outliers)
- `html/`     ← HTML exports of notebooks for quick viewing
```

---

## Inputs (read-only)

Use **phase-assigned, feature-level** CSVs created in preprocessing:

```
feature_extracted_data/
ECG/ECG_{1|3|5}_Minute**phase_assigned.csv
EMG/Including_Masseter/EMG*{1|3|5}_Minute_Inc_Masseter**phase_assigned.csv
EMG/Excluding_Masseter/EMG*{1|3|5}_Minute_Exc_Masseter**phase_assigned.csv
RSP/RSP*{1|3|5}_Minute*_phase_assigned.csv
```

## What to cover in each notebook (checklist)

- **Dataset overview**
```
  - Rows, columns, participants (`ID`), phases, conditions, window sizes
  - Class balance: **Depressed vs Healthy** (overall and per phase/condition)
  ```
- **Missingness & data quality**
```
  - % missing per feature; drop/flag rules; zero-variance features
  - Basic **outlier** markers (IQR or z-score) per modality/phase
  ```
- **Distributions**
```
  - Histograms / KDE for top features
  - Phase- and condition-wise comparisons (e.g., training vs coping)
  ```
- **Correlation & redundancy**
```
  - Feature–feature correlation heatmap; flag highly collinear groups
  ```
- **Univariate signal**
```
  - Quick tests per feature (e.g., Mann–Whitney U, effect size) for D vs H
  - Rank top-K candidates (save as CSV)
  ```
- **Low-dim embeddings (optional)**
```
  - PCA/UMAP colored by label, phase, condition (for intuition only)
  ```
- **Leakage checks**
```
  - Ensure no participant ID appears in both splits (if you split in EDA)
  - Exclude label proxies (ID-level meta) from features
  ```
- **Notes & decisions**
```
  - Document exclusions, thresholds, quirks found here
```

## Roadmap

1. **Class balance & missingness**
2. **Distributions** and **outliers**
3. **Correlations** / redundancy
4. **Univariate** signal (rank top-K)
5. Optional **PCA/UMAP**
6. **Notes/decisions** summary at the end