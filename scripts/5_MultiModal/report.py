"""
Aggregate all final_results.csv from Early/Late Fusion Classification/Regression
into one Excel file per main folder, with 4 subsheets:
  by_ID_inc_text, by_ID_exc_text, by_phase_inc_text, by_phase_exc_text.
Each row has original final_results.csv columns plus condition, phase, aggregation_method.

Requires: pandas, openpyxl (pip install openpyxl)
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent

MAIN_FOLDERS = [
    "Early Fusion Classification Results",
    "Early Fusion Regression Results",
    "Late Fusion Classification Results",
    "Late Fusion Regression Results",
]

MODALITIES = {"audio", "ecg", "emg", "rsp", "text", "video"}

# Subsheet names (aggregation + text inclusion)
SHEET_NAMES = [
    "by_ID_inc_text",
    "by_ID_exc_text",
    "by_phase_inc_text",
    "by_phase_exc_text",
]


def parse_config_folder_name(config_folder_name: str) -> Tuple[str, str, str, str]:
    """
    Parse config folder name to get condition, phase, aggregation_method, and subsheet key.
    Returns (condition, phase, aggregation_method, subsheet_key).
    subsheet_key is one of: by_ID_inc_text, by_ID_exc_text, by_phase_inc_text, by_phase_exc_text.
    """
    # Aggregation
    if config_folder_name.endswith("_by_ID"):
        aggregation_method = "by_ID"
        base = config_folder_name[:-6]
    elif config_folder_name.endswith("_byPhase"):
        aggregation_method = "by_phase"
        base = config_folder_name[:-8]
    else:
        return "", "", "", ""

    # Text included?
    inc_text = "_text_" in config_folder_name or config_folder_name.endswith("_text")
    if aggregation_method == "by_ID":
        subsheet_key = "by_ID_inc_text" if inc_text else "by_ID_exc_text"
    else:
        subsheet_key = "by_phase_inc_text" if inc_text else "by_phase_exc_text"

    # Strip Early_Fusion_ or Late_Fusion_
    if base.startswith("Early_Fusion_"):
        base = base[12:]
    elif base.startswith("Late_Fusion_"):
        base = base[11:]
    else:
        return "", "", aggregation_method, subsheet_key

    tokens = base.split("_")
    condition_phase_tokens = []
    for t in reversed(tokens):
        if t.lower() in MODALITIES:
            break
        condition_phase_tokens.append(t)
    condition_phase_tokens.reverse()

    if len(condition_phase_tokens) == 0:
        condition, phase = "", ""
    elif len(condition_phase_tokens) == 1:
        condition, phase = condition_phase_tokens[0], ""
    else:
        condition = condition_phase_tokens[0]
        phase = "_".join(condition_phase_tokens[1:])

    return condition, phase, aggregation_method, subsheet_key


def collect_final_results_in_dir(main_dir: Path) -> List[Tuple[Path, str, str, str, str]]:
    """
    Find all final_results.csv under main_dir.
    Returns list of (csv_path, condition, phase, aggregation_method, subsheet_key).
    """
    results = []
    if not main_dir.is_dir():
        return results
    for config_dir in main_dir.iterdir():
        if not config_dir.is_dir():
            continue
        condition, phase, aggregation_method, subsheet_key = parse_config_folder_name(
            config_dir.name
        )
        for model_dir in config_dir.iterdir():
            if not model_dir.is_dir():
                continue
            csv_path = model_dir / "final_results.csv"
            if csv_path.is_file():
                results.append(
                    (csv_path, condition, phase, aggregation_method, subsheet_key)
                )
    return results


def aggregate_folder(main_folder_name: str) -> Optional[Path]:
    """
    Aggregate all final_results.csv in main_folder_name into one Excel file
    with 4 sheets: by_ID_inc_text, by_ID_exc_text, by_phase_inc_text, by_phase_exc_text.
    """
    main_dir = BASE_DIR / main_folder_name
    if not main_dir.is_dir():
        print(f"Skipping (not a directory): {main_dir}")
        return None

    rows = collect_final_results_in_dir(main_dir)
    if not rows:
        print(f"No final_results.csv found under {main_dir}")
        return None

    # Group by subsheet_key
    by_sheet: dict[str, List[pd.DataFrame]] = {k: [] for k in SHEET_NAMES}
    for csv_path, condition, phase, aggregation_method, subsheet_key in rows:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not read {csv_path}: {e}")
            continue
        df["condition"] = condition
        df["phase"] = phase
        df["aggregation_method"] = aggregation_method
        if subsheet_key in by_sheet:
            by_sheet[subsheet_key].append(df)

    # Build one DataFrame per sheet (concatenate), then write Excel
    out_path = main_dir / "aggregated_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name in SHEET_NAMES:
            dfs = by_sheet[sheet_name]
            if not dfs:
                pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                continue
            out = pd.concat(dfs, ignore_index=True)
            extra = ["condition", "phase", "aggregation_method"]
            cols = [c for c in out.columns if c not in extra] + [
                c for c in extra if c in out.columns
            ]
            out = out[cols]
            out.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Written {out_path} (sheets: {SHEET_NAMES})")
    return out_path


def main():
    for folder_name in MAIN_FOLDERS:
        aggregate_folder(folder_name)


if __name__ == "__main__":
    main()
