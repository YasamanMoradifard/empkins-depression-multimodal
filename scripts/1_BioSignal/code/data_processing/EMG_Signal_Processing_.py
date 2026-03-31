#################### Library ########################
import pandas as pd
import os
import neurokit2 as nk
import numpy as np
from biosppy.signals import emg
from scipy.signal import welch
import warnings
import traceback
warnings.filterwarnings("ignore", category=RuntimeWarning)

####################################################
###################### EMG ########################
####################################################

print(f'\n#######################\nExcluding Masseter\n##########################\n')

# ================== CONFIG ==================
RCT_path = '/home/vault/empkins/tpD/D02/Students/Yasaman/RCT_Data_Info/result/RCT_info_.csv'
RCT_df = pd.read_csv(RCT_path)
IDs = RCT_df['ID'].tolist()

SAMPLING_RATE = 2000
Window_Time = 5  # in minutes
WINDOW_SIZE = Window_Time * 60 * SAMPLING_RATE


# ================== WINDOWING & FEATURE UTILS (NEW) ==================
def apply_windowing(df, window_size):
    """Return only full, non-overlapping windows of exact size."""
    n = len(df)
    num_full = n // window_size
    return [df.iloc[i*window_size:(i+1)*window_size] for i in range(num_full)]

def _rmse(x): 
    x = np.asarray(x); return float(np.sqrt(np.mean(x**2))) if x.size else np.nan
def _mav(x):  
    x = np.asarray(x); return float(np.mean(np.abs(x))) if x.size else np.nan
def _var(x):  
    x = np.asarray(x); return float(np.var(x)) if x.size else np.nan
def _energy(x):
    x = np.asarray(x); return float(np.sum(x**2)) if x.size else np.nan

def _mnf_mdf_fr(x, fs):
    x = np.asarray(x)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    from scipy.signal import welch
    nperseg = min(max(256, len(x)//8), len(x))
    f, psd = welch(x, fs=fs, nperseg=nperseg, detrend='constant')
    total = np.sum(psd); eps = 1e-12
    if total < eps:  # silent/flat window
        return np.nan, np.nan, np.nan
    mnf = float(np.sum(f * psd) / max(total, eps))
    csum = np.cumsum(psd)
    mdf_idx = int(np.searchsorted(csum, total/2.0))
    mdf_idx = min(max(mdf_idx, 0), len(f)-1)
    mdf = float(f[mdf_idx])
    peak_idx = int(np.argmax(psd))
    fr = float(f[peak_idx] / mnf) if mnf > 0 else np.nan
    return mnf, mdf, fr

def extract_emg_features_from_clean(clean_df, muscles, fs):
    """One-row DataFrame with features from cleaned signals."""
    feats = {}
    present = [m for m in muscles if m in clean_df.columns]
    missing = [m for m in muscles if m not in clean_df.columns]

    # Fill NaNs for any missing channels
    for m in missing:
        p = m[0].lower()
        for name in ["mean","std","min","max","range","rmse","mav","var","energy","mnf","mdf","fr"]:
            feats[f"emg_{p}_{name}"] = np.nan

    for m in present:
        p = m[0].lower()
        x = clean_df[m].to_numpy()

        # simple stats
        if len(x):
            feats[f'emg_{p}_mean']  = float(np.mean(x))
            feats[f'emg_{p}_std']   = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            feats[f'emg_{p}_min']   = float(np.min(x))
            feats[f'emg_{p}_max']   = float(np.max(x))
            feats[f'emg_{p}_range'] = feats[f'emg_{p}_max'] - feats[f'emg_{p}_min']
        else:
            for k in ['mean','std','min','max','range']:
                feats[f'emg_{p}_{k}'] = np.nan

        # amplitude/variance/energy
        feats[f'emg_{p}_rmse']   = _rmse(x)
        feats[f'emg_{p}_mav']    = _mav(x)
        feats[f'emg_{p}_var']    = _var(x)
        feats[f'emg_{p}_energy'] = _energy(x)

        # spectral
        mnf, mdf, fr = _mnf_mdf_fr(x, fs)
        feats[f'emg_{p}_mnf'] = mnf
        feats[f'emg_{p}_mdf'] = mdf
        feats[f'emg_{p}_fr']  = fr

    return pd.DataFrame([feats])

# ================== TWO PIPELINES (NEW) ==================
def process_emg_excluding_masseter(df_raw: pd.DataFrame,
                                   sampling_rate: int,
                                   window_size: int,
                                   id_str: str,
                                   error_log: list):
    """Corrugator, Oculi, Zygomaticus only."""
    muscles = ['Corrugator', 'Oculi', 'Zygomaticus']
    have_cols = [m for m in muscles if m in df_raw.columns]
    if not have_cols:
        raise RuntimeError(f"No EMG channels found for {id_str} (excluding Masseter).")

    # Clean once
    cleaned = {}
    for m in have_cols:
        try:
            cleaned[m] = nk.emg_clean(df_raw[m].values, sampling_rate=sampling_rate)
        except Exception as e:
            raise RuntimeError(f"nk.emg_clean failed for {m}: {e}")
    clean_df = pd.DataFrame(cleaned)

    # Windowing
    windows = apply_windowing(clean_df, window_size)
    if not windows:
        raise ValueError("No full windows (exact size).")

    # Features
    rows = []
    for w_idx, w in enumerate(windows):
        try:
            fdf = extract_emg_features_from_clean(w, muscles, sampling_rate)
            fdf["Window_Index"] = w_idx
            rows.append(fdf)
        except Exception as e:
            error_log.append({
                "ID": id_str, "Window_Index": w_idx,
                "Error_Type": "Feature Extraction (EXCL)",
                "Error_Message": str(e),
                "Traceback": traceback.format_exc()
            })
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out["ID"] = id_str
    out["masseter_included"] = 0
    return out


def process_emg_including_masseter(df_raw: pd.DataFrame,
                                   sampling_rate: int,
                                   window_size: int,
                                   id_str: str,
                                   error_log: list):
    """Corrugator, Oculi, Zygomaticus, Masseter."""
    muscles = ['Corrugator', 'Oculi', 'Zygomaticus', 'Masseter']
    have_cols = [m for m in muscles if m in df_raw.columns]
    if not have_cols:
        raise RuntimeError(f"No EMG channels found for {id_str} (including Masseter).")

    # Clean once
    cleaned = {}
    for m in have_cols:
        try:
            cleaned[m] = nk.emg_clean(df_raw[m].values, sampling_rate=sampling_rate)
        except Exception as e:
            raise RuntimeError(f"nk.emg_clean failed for {m}: {e}")
    clean_df = pd.DataFrame(cleaned)

    # Windowing
    windows = apply_windowing(clean_df, window_size)
    if not windows:
        raise ValueError("No full windows (exact size).")

    # Features
    rows = []
    for w_idx, w in enumerate(windows):
        try:
            fdf = extract_emg_features_from_clean(w, muscles, sampling_rate)
            fdf["Window_Index"] = w_idx
            rows.append(fdf)
        except Exception as e:
            error_log.append({
                "ID": id_str, "Window_Index": w_idx,
                "Error_Type": "Feature Extraction (INCL)",
                "Error_Message": str(e),
                "Traceback": traceback.format_exc()
            })
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out["ID"] = id_str
    out["masseter_included"] = 1
    return out

# ================== AUTO-DISPATCHER (NEW) ==================
def auto_process_emg_for_id(id_int: int,
                            sampling_rate: int,
                            window_minutes: int,
                            error_log: list):
    """
    Loads Data.csv for an ID, detects if Masseter is present,
    and routes to the correct pipeline automatically.
    """
    id_str = f"{id_int:03d}"  # change to :04d if your folders are 4-digit
    window_size = window_minutes * 60 * sampling_rate

    # Locate file
    path1 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/processed_CRT_data/{id_str}/Data.csv'
    path2 = f'/home/vault/empkins/tpD/D02/Students/Yasaman/{id_str}/Data.csv'
    if os.path.exists(path1):
        df_raw = pd.read_csv(path1)
    elif os.path.exists(path2):
        df_raw = pd.read_csv(path2)
    else:
        raise FileNotFoundError(f"No Data.csv found for {id_str}")

    # Ground-truth decision from RCT_info_.csv
    has_masseter = bool(MASSETER_MAP.get(int(id_int), False))

    if has_masseter:
        return process_emg_including_masseter(df_raw, sampling_rate, window_size, id_str, error_log)
    else:
        return process_emg_excluding_masseter(df_raw, sampling_rate, window_size, id_str, error_log)

# ---- Build ground-truth map: ID -> has_masseter (True/False)
def build_masseter_map(df):
    df = df.copy()
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df = df.dropna(subset=['ID'])
    df['ID'] = df['ID'].astype(int)

    col = 'Including_masseter'  # exact column name in RCT_info_.csv
    # normalize values like Ja/Nein/yes/no/1/0
    v = df[col].astype(str).str.strip().str.lower()
    has = v.map({'ja': True, 'nein': False, 'yes': True, 'no': False, '1': True, '0': False}).fillna(False)

    return {int(i): bool(h) for i, h in zip(df['ID'], has)}

MASSETER_MAP = build_masseter_map(RCT_df)




# ================== MAIN (SEPARATE OUTPUTS) ==================
dfs_excl = []   # excluding Masseter
dfs_incl = []   # including Masseter
error_log = []
cnt = 1
print(f"[INFO] Total IDs: {len(IDs)} | Include masseter: {sum(MASSETER_MAP.get(i, False) for i in IDs)} | Exclude: {sum(not MASSETER_MAP.get(i, False) for i in IDs)}")

for ID in IDs:
    print(f"{cnt}. Start Processing {ID:03d}")
    try:
        df_emg = auto_process_emg_for_id(
            id_int=int(ID),
            sampling_rate=SAMPLING_RATE,
            window_minutes=Window_Time,
            error_log=error_log
        )
        if df_emg is not None:
            # Route to the correct bucket based on the flag the pipeline sets
            is_incl = int(df_emg.get("masseter_included", 0).iloc[0]) == 1
            if is_incl:
                dfs_incl.append(df_emg)
                print(f"   ✔ Features (INCLUDING Masseter) for {ID:03d}")
            else:
                dfs_excl.append(df_emg)
                print(f"   ✔ Features (EXCLUDING Masseter) for {ID:03d}")
        else:
            print(f"   ⚠ No features for {ID:03d}")
    except Exception as e:
        error_log.append({
            "ID": f"{ID:03d}",
            "Window_Index": None,
            "Error_Type": "General",
            "Error_Message": str(e),
            "Traceback": traceback.format_exc()
        })
        print(f"   ✖ Error for {ID:03d}: {e}")
    cnt += 1

# ================== SAVE RESULTS (SEPARATE FILES) ==================
# Excluding Masseter
if dfs_excl:
    df_excl_all = pd.concat(dfs_excl, ignore_index=True)
    out_excl = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Excluding_Masseter/EMG_{Window_Time}_Minute_Exc_Masseter__.csv"
    df_excl_all.to_csv(out_excl, index=False)
    print(f"[INFO] Excluding-Masseter features saved: {out_excl}")
else:
    print("[INFO] No Excluding-Masseter features to save.")

# Including Masseter
if dfs_incl:
    df_incl_all = pd.concat(dfs_incl, ignore_index=True)
    out_incl = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Including_Masseter/EMG_{Window_Time}_Minute_Inc_Masseter__.csv"
    df_incl_all.to_csv(out_incl, index=False)
    print(f"[INFO] Including-Masseter features saved: {out_incl}")
else:
    print("[INFO] No Including-Masseter features to save.")

# Error log (combined; or split by type if you prefer)
if error_log:
    err_df = pd.DataFrame(error_log)
    err_out = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/EMG_{Window_Time}_Minute_errors__.csv"
    err_df.to_csv(err_out, index=False)
    print(f"[INFO] Error log saved: {err_out}")