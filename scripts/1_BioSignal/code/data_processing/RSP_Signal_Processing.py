#################### Library ########################
import pandas as pd
import neurokit2 as nk
import numpy as np
from scipy.signal import welch
import os
import warnings
import traceback
warnings.filterwarnings("ignore", category=RuntimeWarning)

####################################################
###################### RSP #########################
####################################################

# ================== CONFIG ==================
RCT_path = '/home/vault/empkins/tpD/D02/Students/Yasaman/RCT_Data_Info/result/RCT_info_.csv'
df_RCT = pd.read_csv(RCT_path)
IDs = df_RCT['ID'].tolist()

SAMPLING_RATE = 2000
Window_Time = 5  # minutes
WINDOW_SIZE = Window_Time * 60 * SAMPLING_RATE
overlap = 0

# ================== FUNCTIONS ==================

def extract_features(signal, window_size, overlap, ID):
    features = []
    step_size = int(window_size * (1 - overlap))

    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]

        # Always compute basic stats
        stats = {
            "mean": np.mean(window),
            "std": np.std(window),
            "min": np.min(window),
            "max": np.max(window),
            "median": np.median(window),
            "ID": ID
        }

        try:
            # Signal cleaning
            window_cleaned = nk.rsp_clean(window, sampling_rate=SAMPLING_RATE)

            # Peak detection and correction
            df_peaks, peaks_dict = nk.rsp_peaks(window_cleaned)
            info = nk.rsp_fixpeaks(peaks_dict)

            # Respiration rate and variability
            rsp_rate = nk.rsp_rate(window_cleaned, peaks_dict, sampling_rate=SAMPLING_RATE)

            # Try extracting RRV features
            try:
                rrv_features = nk.rsp_rrv(rsp_rate, info, sampling_rate=SAMPLING_RATE, show=False)
            except ValueError as e:
                if "dimension * delay" in str(e):
                    warning_msg = f"RRV metrics skipped (too short BBI series)"
                    print(f"[WARNING] {warning_msg} for ID={ID}, start={start}")
                    error_log.append({"ID": ID, "Error": warning_msg, "Window_Start": start})
                    rrv_features = {}  # no RRV features
                else:
                    raise

            # Merge stats + rrv features
            row = {**rrv_features, **stats}
            features.append(row)

        except Exception as e:
            print(f"[ERROR] Feature extraction failed for ID={ID}, window starting at {start}")
            print(f"        Exception: {type(e).__name__}: {e}")
            traceback.print_exc()

            # Append row with only basic stats and NaNs for RRV features
            nan_rrv = {k: np.nan for k in [
                "RRV_RMSSD", "RRV_MeanBB", "RRV_SDBB", "RRV_SDSD",
                "RRV_CVBB", "RRV_CVSD", "RRV_MedianBB", "RRV_MadBB",
                "RRV_MCVBB", "RRV_VLF", "RRV_LF", "RRV_HF",
                "RRV_LFHF", "RRV_LFn", "RRV_HFn", "RRV_SD1",
                "RRV_SD2", "RRV_SD2SD1", "RRV_ApEn", "RRV_SampEn"
            ]}
            row = {**nan_rrv, **stats}
            features.append(row)

    return features

# ================== MAIN LOOP ==================
Dataframes = []
error_log = []
cnt = 1

for ID in IDs:
    print(f"{cnt}. Start Processing {ID}")
    ID_str = f'{ID:03d}'

    try:
        # File paths
        path1 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/processed_CRT_data/{ID_str}/Data.csv'
        path2 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/{ID_str}/Data.csv'

        # Load CSV
        if os.path.exists(path1):
            df = pd.read_csv(path1)
            print(f"{cnt}. Loaded Signal for {ID_str}\n")
        elif os.path.exists(path2):
            df = pd.read_csv(path2)
            print(f"{cnt}. Loaded Signal for {ID_str}\n")
        else:
            print(f"[WARNING] No data found for {ID_str}")
            error_log.append({"ID": ID_str, "Error": "File not found"})
            continue

        # Feature extraction
        try:
            signal = df.RSP
            features = extract_features(signal, WINDOW_SIZE, overlap, ID_str)
            features_df = pd.DataFrame(features)  # convert list of dicts to DataFrame
            Dataframes.append(features_df)
        except Exception as e:
            print(f"[ERROR] Processing failed for ID={ID_str}")
            print(f"        Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            error_log.append({"ID": ID_str, "Error": str(e)})
            continue

    except Exception as e:
        print(f"[ERROR] Could not load or process data for ID={ID_str}")
        print(f"        Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        continue

    print(f"Features Extracted for ID {ID_str}\n")
    cnt += 1

# ================== SAVE RESULTS ==================
if Dataframes:
    df_RSP = pd.concat(Dataframes).reset_index(drop=True)
    df_RSP["ID"] = df_RSP["ID"].astype(str).str.zfill(3)
    df_RCT["ID"] = df_RCT["ID"].astype(str).str.zfill(3)
    df_RSP = pd.merge(df_RSP, df_RCT, on="ID", how="left")

    output_path = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/RSP/RSP_{Window_Time}_Minute.csv"
    df_RSP.to_csv(output_path, index=False)
    print(f"[INFO] Feature file saved: {output_path}")

# Save error log
if error_log:
    error_df = pd.DataFrame(error_log)
    error_log_path = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/RSP/RSP_{Window_Time}_Minute_errors.csv"
    error_df.to_csv(error_log_path, index=False)
    print(f"[INFO] Error log saved: {error_log_path}")