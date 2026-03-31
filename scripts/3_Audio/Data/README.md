## Audio Data Preparation Overview

This folder contains the metadata and scripts used to prepare OpenSMILE audio features.

### Snapshot of the dataset (from `merged_RCT_info.csv`)

- **Number of participants**: 253
- **Diagnoses**:
  - **Healthy**: 126
  - **Depressed**: 127
- **Conditions**:
  - **CR**: 64 participants
  - **CRADK**: 63 participants
  - **ADK**: 63 participants
  - **SHAM**: 63 participants
- **Condition × Diagnose breakdown**:
  - **CR**: 32 Healthy, 32 Depressed
  - **CRADK**: 32 Healthy, 31 Depressed
  - **ADK**: 31 Healthy, 32 Depressed
  - **SHAM**: 31 Healthy, 32 Depressed
- **Class balance**:
  - Overall and within each condition the classes are **almost perfectly balanced** (Healthy ≈ Depressed).

---

### Files in this folder

- **`merged_RCT_info.csv`**  
  - Contains one row per participant with:
    - `ID`, `condition`, `Diagnose`
    - `training1_type`, `training2_type` (positive/negative)
    - Number of files per phase (`training*_num_files`, `induction*_num_files`, etc.).

- **`preparing_audio_data.py`**  
  - Main script that prepares per‑participant CSVs under `processed_audio_opensmile/`.

- **`processed_audio_opensmile/`**  
  - Output folder with one subfolder per participant ID (zero‑padded, e.g. `004`, `075`, `1101`).
  - Inside each participant folder you get CSVs like:
    - `4_healthy_CR_induction1_1.csv`
    - `4_healthy_CR_training_neg_5.csv`
    - `4_healthy_CR_training_pos_12.csv`

- **`logs/`**  
  - Contains `runlog_YYYYMMDD_HHMMSS.txt` with detailed information about each run:
    - Which files were found
    - How many features and rows each file had
    - Which columns were removed as duplicates.

---

### Output CSV naming and structure

Each output CSV in `processed_audio_opensmile/{ID}/timeseries_opensmile_features_vad` has the name:

- **`{ID}_{diagnose}_{condition}_{phase}_{aufgabe}.csv`**
  - Example: `4_healthy_CR_training_pos_5.csv`

Each CSV has:

- **Metadata columns** (always first):
  - `ID` – numeric participant ID from `merged_RCT_info.csv`
  - `diagnose` – `Healthy` or `Depressed`
  - `condition` – `CR`, `CRADK`, `ADK`, or `SHAM`
  - `phase` – one of:
    - `induction1`, `induction2`
    - `training_pos`, `training_neg`
  - `aufgabe` – integer from 1–20 (for `training_*`) or `1` for induction phases

- **Feature columns** (all remaining columns):
  - Time‑series OpenSMILE features from the five methods:
    - `compare16`, `egemaps`, `gemaps`, `is09`, `is13` (and optionally `is02` if present)
  - Columns are named like:
    - `F0final_sma_compare16`, `F0final_sma_egemaps`, `mfcc_sma[1]_gemaps`, …

---

### How phases and tasks (Aufgaben) are detected

- **Participant folders**:
  - For each participant, the script looks in:
    - `<BASE_PATH>/{padded_ID}/timeseries_opensmile_features_vad/`
  - `padded_ID` is the ID with zero padding (e.g. `4 → 004`).

- **Induction files (`Belastungsphase`)**:
  - Filenames contain `Belastungsphase` and two timestamps:
    - `YYYY-MM-DD_HH-MM_Belastungsphase_YYYY-MM-DD_HH-MM_<method>.csv`
  - The script:
    - Parses the first timestamp.
    - Groups Belastungsphase files by start time.
    - The earliest timestamp group → **`induction1`**.
    - The later timestamp group → **`induction2`**.
  - All induction files are treated as **`aufgabe = 1`**.

- **Training files (`Training_1` / `Training_2`)**:
  - Filenames look like:
    - `YYYY-MM-DD_HH-MM_Training_1_Aufgabe_5_YYYY-MM-DD_HH-MM_<method>.csv`
  - The script:
    - Reads `Training_{N}` and `Aufgabe_{M}` from the name.
    - Uses `training1_type` / `training2_type` from `merged_RCT_info.csv`:
      - `positive` → phase = **`training_pos`**
      - `negative` → phase = **`training_neg`**
    - Sets `aufgabe = M` (1–20).

---

### How feature files are combined for each phase/Aufgabe

For each participant and each `(phase, aufgabe)` combination:

- **Step 1 – group by method**:
  - The script collects at most one CSV per method:
    - e.g. `{compare16, egemaps, gemaps, is09, is13}`.

- **Step 2 – read and clean each CSV**:
  - Tries `sep=';'` first; if only 1 column is detected, retries with comma.
  - Removes index/meta columns such as:
    - `name`, `frameTime`, or any compound columns where `name` / `frameTime` appear at the start.
  - Logs how many features and rows each file had before and after cleaning.

- **Step 3 – rename columns with method suffix**:
  - After cleaning, every column is renamed to make its method explicit:
    - e.g. `F0final_sma` from `compare16` → `F0final_sma_compare16`
    - `Loudness_sma3` from `egemaps` → `Loudness_sma3_egemaps`

- **Step 4 – align number of rows**:
  - Different methods may produce different numbers of frames (rows).
  - The script:
    - Computes the maximum number of rows across methods.
    - **Pads shorter dataframes with rows of zeros** so that all methods have the same row count:
      - Existing rows = real feature values.
      - Extra rows at the bottom = zeros (meaning “no data” for that method at those frames).

- **Step 5 – horizontal concatenation**:
  - Once all method dataframes have the same number of rows, they are concatenated **horizontally** (`axis=1`):
    - This gives one big time‑series where each row is a frame.
    - Columns are all features from all methods side‑by‑side.

---

### How duplicated feature columns are detected and removed

After concatenation, there can be duplicated or redundant columns. The script runs `remove_duplicate_columns()` on the big dataframe; this function works in several passes:

- **1. Same column name (exact name duplicates)**:
  - The script groups columns by their **exact name**.
  - For each group with the same name:
    - If **all** columns have exactly the same values:
      - Keep the first one, drop the rest.
    - If columns have the same name but **different values**:
      - It creates a **merged** column:
        - Start from the first column.
        - Wherever the first column has a missing value (NaN),
          fill it using values from the other columns.
      - Drops the other columns after merging.
      - Result: one best “combined” column for that feature name.

- **2. Suffix pattern duplicates**:
  - Some features may look like:
    - `MMFCC1`, `MMFCC1_1`, `MMFCC1_2`, …
  - If a base column (e.g. `MMFCC1`) exists:
    - The script drops the suffixed versions (`MMFCC1_1`, `MMFCC1_2`, …).

- **3. Identical values but different names**:
  - The script compares remaining columns pairwise:
    - If two columns have **identical values**:
      - Keeps the one with **fewer NaNs** (more complete data).
      - Drops the other as redundant.

- **4. Final cleanup and logging**:
  - All columns marked as duplicates are dropped at once.
  - The function returns:
    - The cleaned dataframe.
    - A dictionary (`duplicate_info`) with:
      - Which columns were dropped.
      - Whether they had the same name, same values, suffix pattern, etc.
  - The main script logs:
    - How many duplicate columns were processed.
    - How many were:
      - Same name / different values (merged)
      - Same name / same values
      - Suffix pattern duplicates
      - Identical‑value duplicates.

---

### Summary of the final prepared dataset

For each participant and each `(phase, aufgabe)`:

- **one CSV file** named `ID_diagnose_condition_phase_aufgabe.csv`.
- Each CSV:
  - Keeps **all frames** from the **longest** method by padding shorter methods with zeros.
  - Contains **only one version** of each feature (duplicated columns are merged/removed).
  - Clearly indicates:
    - Which participant
    - Which condition
    - Which diagnosis
    - Which phase and Aufgabe


