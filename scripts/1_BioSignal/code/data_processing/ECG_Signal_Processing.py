import pandas as pd
import neurokit2 as nk
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parameters
RCT_path = '/home/vault/empkins/tpD/D02/Students/Yasaman/RCT_Data_Info/result/RCT_info_.csv'
RCT_df = pd.read_csv(RCT_path)
IDs = RCT_df['ID'].tolist()

SAMPLING_RATE = 2000
Window_Time = 5  # minutes [1, 3, 5]
WINDOW_SIZE = Window_Time * 60 * SAMPLING_RATE

def apply_windowing(df, window_size):
    """
    Splits DataFrame into non-overlapping windows.
    """
    num_windows = len(df) // window_size
    windows = [df.iloc[i * window_size:(i + 1) * window_size] for i in range(num_windows)]
    if len(df) % window_size != 0:
        windows.append(df.iloc[num_windows * window_size:])
    return windows

def extract_ecg_features(signal, sampling_rate):
    """
    Extract HRV features from ECG signal.
    """
    cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
    peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, correct_artifacts=True)
    hrv = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
    return hrv

Dataframes = []
cnt = 1

for ID in IDs:
    ID = f'{ID:03d}'  # zero-padded ID like 001, 002...
    try:
        path1 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/processed_CRT_data/{ID}/Data.csv'
        path2 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/{ID}/Data.csv'

        if os.path.exists(path1):
            df = pd.read_csv(path1)
            print(f"{cnt}. Loaded ECG for {ID}")
        elif os.path.exists(path2):
            df = pd.read_csv(path2)
            print(f"{cnt}. Loaded ECG for {ID}")
        else:
            print(f"[ERROR] No path found for {ID}")
            continue
    except Exception as e:
        print(f"[ERROR] Could not load data for ID={ID} -> {type(e).__name__}: {e}")
        continue

    if "ECG" not in df.columns:
        print(f"[ERROR] ECG column missing for {ID}")
        continue

    windows = apply_windowing(df[["ECG"]], WINDOW_SIZE)
    print(f"Applied Windowing for {ID} - {len(windows)} windows")

    ecg_features = []
    for w_idx, window in enumerate(windows[:-1]):  # skip last incomplete window
        try:
            features = extract_ecg_features(window["ECG"].values, sampling_rate=SAMPLING_RATE)
            features["ID"] = ID
            ecg_features.append(features)
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for ID={ID}, window={w_idx} -> {type(e).__name__}: {e}")
            continue

    if ecg_features:
        df_ecg = pd.concat(ecg_features)
        Dataframes.append(df_ecg)
        print(f"Features extracted for {ID}")

    cnt += 1
if Dataframes:
    df_ECG = pd.concat(Dataframes, ignore_index=True)

    # --- Normalize IDs on both sides (zero-padded strings like 001) ---
    df_ECG['ID'] = df_ECG['ID'].astype(str).str.zfill(3)
    RCT_df['ID'] = pd.to_numeric(RCT_df['ID'], errors='coerce').dropna().astype(int).astype(str).str.zfill(3)

    # (Optional) if RCT has multiple rows per ID, pick the first to avoid cartesian merges
    RCT_df_unique = RCT_df.drop_duplicates(subset=['ID'], keep='first')

    # --- Merge RCT features onto ECG features by ID ---
    df_final = df_ECG.merge(RCT_df_unique, on='ID', how='left', validate='m:1')

    # (Optional) quick sanity checks
    unmatched = df_final['ID'][df_final.filter(regex='^(?!ID$).*', axis=1).isna().all(axis=1)].unique()
    if len(unmatched) > 0:
        print(f"[WARN] {len(unmatched)} IDs had no RCT match (showing up to 10): {list(unmatched)[:10]}")

    # --- Save ---
    output_path = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/ECG/ECG_{Window_Time}_Minute.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Saved ECG+RCT features to {output_path}")
else:
    print("[WARNING] No ECG features extracted.")

