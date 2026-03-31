
# BioSignal — Preprocessing & Data Preparation

Step-by-step pipeline to turn **raw participant TXT** files into **phase-assigned, feature tables** ready for ML.

---

## 🚀 Roadmap: Raw → Features

1) **Raw → PreProcessed**
````markdown
   - **Input:** `participant_XXX.txt`
   - **Tool:** `Raw_Data_To_PreProcessed_CSV.ipynb`
   - **What happens:** Standardize column names; identify **ECG**, **RSP**, **EMG** channels (incl./excl. **Masseter**)
   - **Output:** one **`Data.csv` per participant** (clean columns, no features yet)
````
2) **PreProcessed → Windowed/Filtered**
````
   - **Input:** per-participant `Data.csv`
   - **Tool:** `Data_Processing.ipynb`
   - **What happens:** Filtering/denoising; **windowing** (1/3/5 min); optional basic stats
   - **Output:** clean, windowed signals per participant (intermediate)
````

3) **Windowed Signals → Modality Features**
````
   - **Input:** windowed per-participant data
   - **Tools:** `ECG_Signal_Processing.py`, `EMG_Signal_Processing_.py`, `RSP_Signal_Processing.py`
   - **What happens:** Feature extraction per modality (ECG HRV; EMG time/frequency; RSP RRV + stats)
   - **Output:** **one CSV per modality & window size** (e.g., `ECG_1_Minute.csv`), consolidated across participants
````

4) **Features → Phase-Assigned Features**
````
   - **Input:** modality feature CSVs (e.g., `ECG_1/3/5_Minute.csv`, EMG Inc/Exc Masseter, `RSP_*`)
   - **Tool:** `Phase_Assignment_Final.py`
   - **What happens:** Add **time** per window (from journal), align with `RCT_Phase_Info.csv`, assign **phase**, merge meta
   - **Output:** `*_phase_assigned.csv` (final feature tables for ML)
````

### Visual flow
```
flowchart TD
    A[Raw TXT per participant] --> B[1) Raw → PreProcessed (ipynb)]
    B --> C[2) Filter + Window (ipynb)]
    C --> D[3) Feature Extraction per Modality (py)]
    D --> E[4) Phase Assignment + Meta Merge (py)]
    E --> F[Phase-assigned feature CSVs (1/3/5 min)]
````

> 🔒 **No raw data in Git.** Keep raw and intermediate files in secure storage; scripts/notebooks read and write there.

---

## 1) Raw → PreProcessed CSV

**Notebook:** `Raw_Data_To_PreProcessed_CSV.ipynb`

**Does**

* Load each participant’s raw TXT.
* Normalize column names; detect **ECG**, **RSP**, **EMG** channels (and whether **Masseter** exists).
* Save **per-participant `Data.csv`** with a consistent schema (e.g., `ECG`, `RSP`, `Corrugator`, `Oculi`, `Zygomaticus`, `Masseter` if present).

**Features after this step:** *None (signal-level only)* — standardized columns.

---

## 2) Data Processing (filtering, windowing, basic stats)

**Notebook:** `Data_Processing.ipynb`

**Does**

* Apply **denoising/filtering** as needed.
* **Window** signals into **1 / 3 / 5-minute** non-overlapping chunks.
* Optionally compute basic window stats (e.g., mean, std, min, max, median).

**Features after this step:** optional **basic stats** per window; otherwise clean **windowed signals** passed to step 3.

---

## 3) Signal Processing per Modality (feature extraction)

### ECG — `ECG_Signal_Processing.py`

**Input:** per-participant `Data.csv` with `ECG`, participant meta.
**Does**

* ECG cleaning and **peak detection**; derive **HRV** series.
* Extract **HRV features** (time, frequency, nonlinear).
* Concatenate **all participants** per window size.
* **Output:** `ECG_{1|3|5}_Minute.csv`

**ECG feature examples**

* Time: **RMSSD**, **SDNN**, pNN50
* Frequency: **LF**, **HF**, **LF/HF**
* Nonlinear: **SD1**, **SD2**, **ApEn**, **SampEn**
  (*Exact columns depend on library and script settings.*)

---

### EMG — `EMG_Signal_Processing_.py`

**Input:** per-participant `Data.csv` with `Corrugator`, `Oculi`, `Zygomaticus`, and optionally `Masseter`; meta to detect Masseter.
**Does**

* EMG cleaning per channel.
* Windowing; compute **time** and **spectral** features per muscle.
* Saves **separate outputs** for **Including** vs **Excluding Masseter**.

**Output**

* `EMG_{1|3|5}_Minute_Inc_Masseter__.csv`
* `EMG_{1|3|5}_Minute_Exc_Masseter__.csv`
* Optional error log: `EMG_{Window}_Minute_errors__.csv`

**EMG feature examples (per muscle)**

* Time: mean, std, min, max, range, **MAV**, **variance**, **energy**, **RMSE**
* Frequency: **MNF** (mean freq), **MDF** (median freq), **FR** (peak/mean ratio)

---

### RSP — `RSP_Signal_Processing.py`

**Input:** per-participant `Data.csv` with `RSP`, participant meta.
**Does**

* RSP cleaning; respiration peaks/fixes.
* Respiration rate series; **RRV features** when feasible; **basic stats** for all windows.
* Concatenates across participants.

**Output:** `RSP_{1|3|5}_Minute.csv`

**RSP feature examples**

* Basic stats: mean, std, min, max, median
* RRV: **RRV_RMSSD**, **RRV_SDBB**, **SD1**, **SD2**, **LF**, **HF**, **LF/HF**, **ApEn**, **SampEn**
  (*RRV may be NaN for short/flat windows; basic stats remain available.*)

---

## 4) Phase Assignment & Meta Merge — `Phase_Assignment_Final.py`

**Inputs**

* Modality feature CSVs from step 3 (e.g., `ECG_5_Minute.csv`, `EMG_5_Minute_Inc_Masseter__.csv`, `RSP_5_Minute.csv`)
* Per-participant **journal `.xls`** (to read true **start time**)
* `RCT_Phase_Info.csv` (phase intervals), `RCT_info_.csv` (ID-level meta)

**Does**

1. Group rows **by participant ID**.
2. From journal `.xls`, read earliest **“Date Created”** → set **start time** per participant.
3. Create a **time** column per row (advance by window size).
4. **Assign `phase`** by aligning each row’s time with intervals in `RCT_Phase_Info.csv`.
5. Merge **RCT meta**; normalize fields (e.g., `condition`, gender).
6. Save **final, phase-assigned** CSV(s).

**Columns added/enriched**

* `time` (window start), `phase` (`training`, `coping`, `emotion_induction_*`, …), normalized meta (`condition`, etc.)

---

## Where files land (example)

```
feature_extracted_data/
├─ ECG/
│  ├─ ECG_1_Minute.csv
│  ├─ ECG_3_Minute.csv
│  └─ ECG_5_Minute.csv
├─ EMG/
│  ├─ Including_Masseter/EMG_1_Minute_Inc_Masseter__.csv
│  ├─ Including_Masseter/EMG_3_Minute_Inc_Masseter__.csv
│  ├─ Including_Masseter/EMG_5_Minute_Inc_Masseter__.csv
│  ├─ Excluding_Masseter/EMG_1_Minute_Exc_Masseter__.csv
│  ├─ Excluding_Masseter/EMG_3_Minute_Exc_Masseter__.csv
│  └─ Excluding_Masseter/EMG_5_Minute_Exc_Masseter__.csv
└─ RSP/
   ├─ RSP_1_Minute.csv
   ├─ RSP_3_Minute.csv
   └─ RSP_5_Minute.csv

# After phase assignment:
.../{MOD}/.../{MOD}_{1|3|5}_Minute[_Inc|_Exc]_..._phase_assigned.csv
```

---

## Minimal “How to Run”

1. **Standardize raw files (per participant)**

   * Open `Raw_Data_To_PreProcessed_CSV.ipynb`
   * Set input/output folders; run all → produces per-participant `Data.csv`

2. **Filter & window**

   * Open `Data_Processing.ipynb`
   * Choose window sizes (1/3/5 min); run → produces windowed/cleaned data

3. **Extract features (by modality)**

```bash
python ECG_Signal_Processing.py
python EMG_Signal_Processing_.py
python RSP_Signal_Processing.py
```

* Check `feature_extracted_data/{ECG|EMG|RSP}` for new CSVs

4. **Assign phases**

```bash
python Phase_Assignment_Final.py
```

* Provide paths to journal `.xls`, `RCT_Phase_Info.csv`, modality CSVs