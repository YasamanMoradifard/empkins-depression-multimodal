import os, sys, traceback, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")



###############################################################
# 1. Read main Data
###############################################################

def read_and_group_by_id(file_path):
    """
    Read a CSV file into a dictionary of DataFrames, grouped by 'ID'.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file containing the data. 
        Must include an 'ID' column for grouping.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are unique ID values from the 'ID' column.
        - Values are pandas DataFrames containing rows for that ID.
        Returns an empty dictionary if the file cannot be read.

    Notes
    -----
    - If the 'ID' column is missing, an empty dictionary is returned.
    - All exceptions are caught and printed as error messages, but not raised.
    """
    try:
        df = pd.read_csv(file_path)
        if 'ID' not in df.columns:
            print("[ERROR] read_and_group_by_id: 'ID' column missing")
            return {}

        # Normalize ID -> int (strip leading zeros, drop non-numeric)
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        df = df.dropna(subset=['ID'])
        df['ID'] = df['ID'].astype(int)

        grouped = {int(pid): g.drop(columns=[], errors='ignore').reset_index(drop=True)
                   for pid, g in df.groupby('ID', sort=False)}
        return grouped
    except Exception as e:
        print(f"[ERROR] in read_and_group_by_id: {e}")
        return {}



###############################################################
# 2. Read Journal (xls file)
###############################################################


def read_xlsx_files_from_subfolders(folder_path):
    """
    Read the first `.xls` file from each subfolder in a given directory.

    Parameters
    ----------
    folder_path : str or Path
        Path to the parent folder containing subfolders.
        Each subfolder should contain at least one `.xls` file.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are integer folder names with leading zeros removed (e.g., "005" → 5).
        - Values are pandas DataFrames read from the first `.xls` file in each subfolder.

    Notes
    -----
    - If no `.xls` file is found in a subfolder, a message is printed and that subfolder is skipped.
    - If a file fails to read, an error is printed and processing continues.
    - Only the first `.xls` file found in each subfolder is read.
    """
    xls_dict = {}
    try:
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue
            try:
                xls_file = next(f for f in os.listdir(subfolder_path) if f.lower().endswith(".xls"))
                file_path = os.path.join(subfolder_path, xls_file)
                df = pd.read_excel(file_path, engine='xlrd', header=0)

                stripped = subfolder_name.lstrip("0")
                key = int(stripped) if stripped.isdigit() else None
                if key is None:
                    print(f"[WARNING] Skip non-numeric journal folder: {subfolder_name}")
                    continue

                xls_dict[key] = df
            except StopIteration:
                print(f"[WARNING] No .xls file in folder: {subfolder_name}")
            except Exception as e:
                print(f"[WARNING] Failed to read journal in {subfolder_name}: {e}")
    except Exception as e:
        print(f"[ERROR] in read_xlsx_files_from_subfolders: {e}")
    return xls_dict



###############################################################
# 3. Concatenate CSV dataframes
###############################################################

def concat_dataframes_from_dict(df_dict, add_key_as_column=True, key_column_name="ID"):
    """
    Concatenate multiple DataFrames stored in a dictionary into a single DataFrame.

    Parameters
    ----------
    df_dict : dict
        Dictionary where:
        - Keys are identifiers (e.g., participant IDs).
        - Values are pandas DataFrames.
    add_key_as_column : bool, default=True
        Whether to add the dictionary key as a new column in each DataFrame.
    key_column_name : str, default="ID"
        Name of the column to store dictionary keys when `add_key_as_column=True`.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame containing all non-empty DataFrames from `df_dict`.
        Returns an empty DataFrame if input is empty or all DataFrames are empty.

    Notes
    -----
    - Ignores `None` or empty DataFrames in the dictionary.
    - If `key_column_name` already exists, the key will not overwrite it.
    - Exception-safe: if an error occurs, returns an empty DataFrame.
    """
    try:
        if not df_dict:
            # No data to concatenate
            return pd.DataFrame()

        concatenation_list = []

        for key, df in df_dict.items():
            # Skip None or empty DataFrames
            if df is None or getattr(df, "empty", False):
                continue

            df_copy = df.copy()

            # Add key as a new column if requested and it doesn't already exist
            if add_key_as_column and key_column_name not in df_copy.columns:
                df_copy[key_column_name] = key

            concatenation_list.append(df_copy)

        if not concatenation_list:
            # All DataFrames were empty
            return pd.DataFrame()

        # Concatenate all DataFrames into one
        return pd.concat(concatenation_list, ignore_index=True)

    except Exception as e:
        print(f"[ERROR] in concat_dataframes_from_dict: {e}")
        return pd.DataFrame()


###############################################################
# 5. Debug Information
###############################################################
def log_debug_error(exception, participant_id, step_name):
    """
    Log detailed debug information when an exception occurs in processing.

    Parameters
    ----------
    exception : Exception
        The caught exception object to log.
    participant_id : str or int
        Identifier for the participant or record that caused the error.
    step_name : str
        Descriptive name of the step or phase where the error occurred.

    Returns
    -------
    None
        This function prints debug information to standard output.
    
    Notes
    -----
    - Displays both a concise error message and a full stack trace.
    - Intended for debugging during development or data processing pipelines.
    """

    # Print a clear header with participant ID and processing step
    print(f"\n[DEBUG] Error for ID={participant_id} at step: {step_name}")

    # Print the exception type and message for quick identification
    print(f"[DEBUG] Exception: {repr(exception)}")

    # Print the full traceback to locate the error in the code
    traceback.print_exc()


###############################################################
# 6. Time column to BioSig data
###############################################################
# === Helper: earliest time-of-day from journal ===
def earliest_time_string_from_journal(journal_df):
    """
    Return the time-of-day from the earliest 'Date Created' value
    in the XLS journal as a string 'HH:MM:SS.microsecond'.
    """
    if journal_df is None or journal_df.empty:
        return None

    # Detect header row containing 'Date Created' (case-insensitive)
    def _has_date_created(row_vals):
        return any(isinstance(v, str) and "date created" in v.lower().strip() for v in row_vals)

    header_row_idx = next((i for i, row in journal_df.iterrows() if _has_date_created(row.values)), None)
    if header_row_idx is not None:
        journal_df = journal_df.copy()
        journal_df.columns = [str(c).strip() for c in journal_df.iloc[header_row_idx].tolist()]
        journal_df = journal_df.iloc[header_row_idx + 1:].reset_index(drop=True)

    # Find the exact column name (case-insensitive)
    col_map = {str(c).lower().strip(): c for c in journal_df.columns}
    dc_col = col_map.get("date created")
    if dc_col is None:
        return None

    # Parse to datetime and pick earliest
    created = pd.to_datetime(journal_df[dc_col].astype(str).str.strip(), errors="coerce")
    if created.notna().sum() == 0:
        return None

    earliest = created.min()
    return earliest.time().strftime("%H:%M:%S.%f")  # e.g., '14:39:31.560000'


def add_time_column_to_biosignal_data(
    biosignal_data_dict: dict,
    journal_data_dict: dict,
    rct_info_df: pd.DataFrame,
    window_size_minutes: float = 1,
) -> dict:
    """
    Add a 'time' column to each participant's biosignal DataFrame based on the
    TRUE experiment start datetime parsed from the journal file, then increase by a
    fixed window size per row (e.g., 1 minute).

    Parameters
    ----------
    biosignal_data_dict : dict[int, pd.DataFrame]
        Participant -> biosignal DataFrame (no 'time' yet).
    journal_data_dict : dict[int, pd.DataFrame]
        Participant -> journal DataFrame (contains a 'Date Created' column somewhere).
    rct_info_df : pd.DataFrame
        Must contain an 'ID' column listing the participants to process.
    window_size_minutes : float, default 1
        Interval between rows.

    Returns
    -------
    dict[int, pd.DataFrame]
        Same mapping as input, but DataFrames now include a 'time' column.

    Notes
    -----
    - Uses the full datetime from journal ('Date Created'); no artificial anchor date.
    - If a participant has no valid start datetime in their journal, that ID is skipped.
    """
    processed_ids = set()
    participant_ids = (
        pd.to_numeric(rct_info_df["ID"], errors="coerce")
        .dropna().astype(int).tolist()
    )

    for participant_id in participant_ids:
        try:
            # ---- Must have biosignal rows ----
            if participant_id not in biosignal_data_dict or biosignal_data_dict[participant_id].empty:
                print(f"[WARNING] Skipping ID {participant_id} — no biosignal rows found")
                continue

            # ---- Parse earliest time-of-day from journal (or special-case) ----
            if participant_id == 1104:
                time_str = "11:27:02.010000"
            else:
                if participant_id not in journal_data_dict or journal_data_dict[participant_id].empty:
                    print(f"[WARNING] Skipping ID {participant_id} — missing or empty journal file")
                    continue
                time_str = earliest_time_string_from_journal(journal_data_dict[participant_id])
                if time_str is None:
                    print(f"[WARNING] Could not parse earliest 'Date Created' for ID {participant_id}, skipping...")
                    continue

            # ---- Make a safe Timestamp (anchor date is arbitrary; time-of-day is what matters) ----
            t = datetime.strptime(time_str, "%H:%M:%S.%f").time()
            start_ts = pd.Timestamp(datetime(2000, 1, 1, t.hour, t.minute, t.second, t.microsecond))

            # ---- Build the time column ----
            df_biosignal = biosignal_data_dict[participant_id].copy()
            periods = len(df_biosignal)
            if periods == 0:
                print(f"[WARNING] Skipping ID {participant_id} — empty biosignal DataFrame")
                continue

            freq = pd.to_timedelta(window_size_minutes, unit="m")
            df_biosignal["time"] = pd.date_range(start=start_ts, periods=periods, freq=freq)
            df_biosignal["start_time_of_day"] = time_str  # optional, per-row

            biosignal_data_dict[participant_id] = df_biosignal
            processed_ids.add(participant_id)

        except Exception as e:
            log_debug_error(e, participant_id, "Add time to biosignal data")
            continue

    # ---- Summary ----
    skipped_ids = sorted(set(participant_ids) - processed_ids)
    print(f"[INFO] Added times for {len(processed_ids)} IDs. Skipped {len(skipped_ids)} "
          f"IDs due to missing/invalid journals or dates: "
          f"{skipped_ids[:10]}{' ...' if len(skipped_ids) > 10 else ''}")

    return biosignal_data_dict

def assign_phases_to_biosignal(
    biosignal_data_dict: dict,
    rct_phase_df: pd.DataFrame,
    window_size_minutes: float
) -> dict:
    """
    Assign a 'phase' column to each participant's biosignal DataFrame based on
    start_time/end_time intervals from rct_phase_df. Compares only time-of-day, ignoring date.

    Parameters
    ----------
    biosignal_data_dict : dict[int, pd.DataFrame]
        ID -> biosignal DataFrame containing a 'time' column (datetime64[ns]).
    rct_phase_df : pd.DataFrame
        Must contain 'ID', 'Phase', 'start_time', 'end_time' columns.
        start_time/end_time are strings or datetime.time values like '17:21:19.096056'.
    window_size_minutes : float
        Size of the biosig row's window, used to compute row center times.

    Returns
    -------
    dict[int, pd.DataFrame]
        Same as input, but with a new 'phase' column.
    """
    processed_ids = 0

    # Ensure start_time/end_time are parsed to datetime.time
    rct_phase_df = rct_phase_df.copy()
    rct_phase_df['start_time'] = pd.to_datetime(rct_phase_df['start_time'], errors='coerce').dt.time
    rct_phase_df['end_time'] = pd.to_datetime(rct_phase_df['end_time'], errors='coerce').dt.time

    # Convert time-of-day to seconds helper
    def tod_to_seconds(t):
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

    for pid, df_bio in biosignal_data_dict.items():
        if df_bio is None or df_bio.empty or 'time' not in df_bio.columns:
            continue

        # Get this ID's phase rows
        phase_rows = rct_phase_df[rct_phase_df['ID'] == pid]
        if phase_rows.empty:
            continue

        df_bio = df_bio.copy()
        df_bio['phase'] = np.nan  # default NaN

        # Biosig row centers in seconds since midnight
        bio_secs = df_bio['time'].dt.hour * 3600 \
                 + df_bio['time'].dt.minute * 60 \
                 + df_bio['time'].dt.second \
                 + df_bio['time'].dt.microsecond / 1e6
        window_sec = window_size_minutes * 60.0
        bio_centers = (bio_secs + (window_sec / 2.0)) % 86400  # wrap around 24h

        # Assign phases
        for _, prow in phase_rows.iterrows():
            phase_name = prow['phase']
            if pd.isna(prow['start_time']) or pd.isna(prow['end_time']):
                continue

            start_sec = tod_to_seconds(prow['start_time'])
            end_sec   = tod_to_seconds(prow['end_time'])

            if end_sec >= start_sec:
                # Normal case (no midnight wrap)
                mask = (bio_centers >= start_sec) & (bio_centers < end_sec)
            else:
                # Wraps past midnight
                mask = (bio_centers >= start_sec) | (bio_centers < end_sec)

            df_bio.loc[mask, 'phase'] = phase_name

        biosignal_data_dict[pid] = df_bio
        processed_ids += 1

    print(f"[INFO] Assigned phases for {processed_ids} participants.")
    return biosignal_data_dict


def main():
    try:
       
        file_path = {
            'ecg_1min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/ECG/ECG_1_Minute.csv',
            'ecg_3min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/ECG/ECG_3_Minute.csv',
            'ecg_5min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/ECG/ECG_5_Minute.csv',

            'rsp_1min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/RSP/RSP_1_Minute.csv',
            'rsp_3min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/RSP/RSP_3_Minute.csv',
            'rsp_5min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/RSP/RSP_5_Minute.csv',

            'emg_1min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Excluding_Masseter/EMG_1_Minute_Exc_Masseter__.csv',
            'emg_3min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Excluding_Masseter/EMG_3_Minute_Exc_Masseter__.csv',
            'emg_5min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Excluding_Masseter/EMG_5_Minute_Exc_Masseter__.csv',
            
            'emg_M_1min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Including_Masseter/EMG_1_Minute_Inc_Masseter__.csv',
            'emg_M_3min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Including_Masseter/EMG_3_Minute_Inc_Masseter__.csv',
            'emg_M_5min' : '/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/EMG/Including_Masseter/EMG_5_Minute_Inc_Masseter__.csv'
        }
        
        # Data information
        data_type = 'EMG'
        time = 5 #[5, 3, 1]
        data_name = f'{data_type.lower()}_M_{str(time)}min'
        print(data_name)

        print(f'\n##############################\n################# time: {time}\n############################')

        # ------------------ 1. Read Main Data
        print("-------------------\nReading BioSig Data: \n")
        df = read_and_group_by_id(file_path[data_name])
        #print(f"Data shape at the begining: {df[4].shape}\n")
        #print(f"Number of participant IDs:{len(df.keys())}\n")

        # ------------------ 2. Read Journal
        print("-------------------\nReading journals: \n")
        RCT_folder_path = '/home/vault/empkins/tpD/D02/RCT/raw_data'
        df_journal = read_xlsx_files_from_subfolders(RCT_folder_path)
        print(f"Number of participant IDs:{len(df_journal.keys())}\n")

        if len(df_journal) == 0:
            print("[ERROR] No journal files found. Aborting.")
            return

        # ------------------ 3. Read RCT Data Info
        print("-------------------\nReading RCT Info: \n")
        RCT_path = '/home/vault/empkins/tpD/D02/Students/Yasaman/RCT_Data_Info/result/RCT_info_.csv'
        RCT_df = pd.read_csv(RCT_path)

        # ---- Prepare RCT meta for merge (ID-normalized, deduped, tidy names)
        RCT_meta = RCT_df.copy()
        RCT_meta['ID'] = pd.to_numeric(RCT_meta['ID'], errors='coerce')
        RCT_meta = RCT_meta.dropna(subset=['ID'])
        RCT_meta['ID'] = RCT_meta['ID'].astype(int)
        RCT_meta = RCT_meta.drop_duplicates(subset='ID', keep='first')

        # ------------------ 4. Read Phase Info
        print("-------------------\nReading RCT_Phase_Info.csv")
        rct_phase_path = '/home/vault/empkins/tpD/D02/Students/Yasaman/RCT_Data_Info/result/RCT_Phase_Info.csv'
        rct_phase_df = pd.read_csv(rct_phase_path)

        # ------------------ 5. make time column in BioSig
        print("-------------------\nMake Time Column")
        df = add_time_column_to_biosignal_data(df, df_journal, RCT_df, window_size_minutes=time)

        # ------------------ 6. Assign Phase column
        print("-------------------\nMake Phase Column")
        # Assign phases using rct_phase_df
        df = assign_phases_to_biosignal(df, rct_phase_df, window_size_minutes=time)

        # ------------------ 7. Concatenate dataframes
        df_concat = concat_dataframes_from_dict(df, add_key_as_column=True, key_column_name="ID")
        print(f"Data shape after concatenation: {df_concat.shape}\n")
        print(f"Final Number of Participants: {len(df_concat['ID'].unique())}\n")

        # ---- Merge ALL RCT columns by ID (left join)
        df_concat['ID'] = pd.to_numeric(df_concat['ID'], errors='coerce').astype('Int64')
        df_concat = df_concat.merge(RCT_meta, on='ID', how='left', suffixes=('', '_rct'))
        print(f"[INFO] RCT meta merged. Rows: {df_concat.shape[0]}")

        # ------------------ 8. Clean/enrich final concatenated dataframe
        # Make drops safe (won't crash if columns are missing)
        df_concat = df_concat.drop(columns=["Unnamed: 0", "start_time_of_day", "Excluding_masseter"])

        # --- Normalize columns used below (so code works even if capitalized upstream)
        #lower_map = {c: c.lower() for c in df_concat.columns}
        #df_concat.rename(columns=lower_map, inplace=True)

        # 1) condition: remove everything starting with space + '('
        if "condition" in df_concat.columns:
            df_concat["condition"] = df_concat["condition"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)

        # 2) previous_depression_diagnosis: map to 'Ja'/'Nein' (keep 'Nein' as Nein, everything else -> Ja)
        if "previous_depression_diagnosis" in df_concat.columns:
            df_concat["previous_depression_diagnosis"] = (
                df_concat["previous_depression_diagnosis"]
                .astype(str).str.strip().str.lower()
                .apply(lambda x: "Nein" if x == "nein" else "Ja")
            )

        # 3) gender normalization
        def _normalize_gender(val):
            s = str(val).strip()
            sl = s.lower()
            if sl == "m (trans)":
                return "m (trans)"
            elif "w" in sl:
                return "W"
            elif "m" in sl:
                return "M"
            elif "d" in sl:
                return "D"
            else:
                return s

        if "gender" in df_concat.columns:
            df_concat["gender"] = df_concat["gender"].apply(_normalize_gender)

        # (Optional) sanity checks in logs
        print("Unique condition values:", sorted(df_concat["condition"].dropna().unique()) if "condition" in df_concat.columns else "n/a")
        print("Unique prev. depression diagnosis:", sorted(df_concat["previous_depression_diagnosis"].dropna().unique()) if "previous_depression_diagnosis" in df_concat.columns else "n/a")
        print("Unique gender values:", sorted(df_concat["gender"].dropna().unique()) if "gender" in df_concat.columns else "n/a")

        # ------------------ 9. Save the Result
        output_path = f"/home/vault/empkins/tpD/D02/Students/Yasaman/BioSig_data/feature_extracted_data/{data_type}/Including_Masseter/{data_type}_{time}_Minute_Inc_Masseter_phase_assigned.csv"
        df_concat.to_csv(output_path, index=False)
        print("Final Data saved successfully!")

    except Exception as e:
        print(f"[ERROR] in read_main_data: {e}")




if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] Script crashed: {e}")