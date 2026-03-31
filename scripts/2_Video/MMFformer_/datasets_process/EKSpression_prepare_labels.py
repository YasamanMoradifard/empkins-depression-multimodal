"""
Builds participant-level label CSVs for the EKSpression (d02) dataset.

For each condition subfolder inside `d02_manifests` (e.g. `CR`, `ADK`,
`CRADK`, `SHAM`, `All`), this script reads the corresponding split CSVs
(`*_train.csv`, `*_validation.csv`, `*_test.csv`) and produces:

    - One labels CSV per condition, e.g. `EKSpression_labels_CR.csv`
    - One combined labels CSV: `EKSpression_labels_ALL.csv`

Each row in the output has the columns:

    ID, label, Condition, fold, phase, Aufgabe

Where:
  - `ID`       : zero-padded participant ID (3 digits for <1000, 4 digits otherwise)
  - `label`    : either "Depressed" or "Healthy" derived from `Diagnose`
  - `Condition`: one of {CR, CRADK, ADK, SHAM, ALL}
  - `fold`     : "train", "validation", or "test" from the split filename
  - `phase`    : copied from the `phase` column in the manifest
  - `Aufgabe`  : copied from the `Aufgabe` column in the manifest (task index)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def format_id(raw_id: str) -> str:
    """
    Zero-pad numeric IDs to 3 digits (<1000) or 4 digits (>=1000).
    Examples:
        "4"   -> "004"
        "70"  -> "070"
        "1101"-> "1101"
    """
    raw_id = str(raw_id).strip()
    if not raw_id:
        return ""
    try:
        val = int(raw_id)
    except ValueError:
        return raw_id
    if val < 1000:
        return f"{val:03d}"
    return f"{val:04d}"


def normalize_label(raw: str) -> Optional[str]:
    """
    Map `Diagnose` values to "Depressed" or "Healthy".
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if "healthy" in s or "control" in s:
        return "Healthy"
    # default to Depressed for any non-healthy diagnosis
    return "Depressed"


def infer_fold_from_filename(name: str) -> Optional[str]:
    """
    Infer fold ("train", "validation", "test") from a CSV filename.
    """
    s = name.lower()
    if "train" in s:
        return "train"
    if "val" in s or "validation" in s:
        return "validation"
    if "test" in s:
        return "test"
    return None


def build_labels_for_condition(cond_dir: Path) -> List[Dict[str, str]]:
    """
    Read all split CSVs in a condition folder and build label records.
    """
    records: List[Dict[str, str]] = []
    if not cond_dir.is_dir():
        return records

    for csv_path in sorted(cond_dir.glob("*.csv")):
        fold = infer_fold_from_filename(csv_path.name)
        if fold is None:
            continue

        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = row.get("ID") or row.get("pid")
                condition = row.get("condition") or cond_dir.name
                diagnose = row.get("Diagnose") or row.get("diagnose")
                phase = row.get("phase", "")
                aufgabe = row.get("Aufgabe", "")

                pid_fmt = format_id(pid)
                label = normalize_label(diagnose)
                if not pid_fmt or label is None:
                    continue

                record = {
                    "ID": pid_fmt,
                    "label": label,
                    "Condition": condition.upper(),
                    "fold": fold,
                    "phase": phase,
                    "Aufgabe": aufgabe,
                }
                records.append(record)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EKSpression_labels CSVs from d02_manifests."
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        required=True,
        help="Root folder containing condition subfolders with split CSVs (e.g., d02_manifests).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where label CSVs will be written (e.g., d02_npy).",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional list of condition subfolders to process (e.g., CR ADK CRADK SHAM All). "
             "Default: all immediate subdirectories under manifests-root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifests_root: Path = args.manifests_root
    output_root: Path = args.output_root

    if not manifests_root.is_dir():
        raise SystemExit(f"manifests-root does not exist or is not a directory: {manifests_root}")

    if args.conditions:
        cond_names = args.conditions
    else:
        cond_names = sorted([p.name for p in manifests_root.iterdir() if p.is_dir()])

    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, str]] = []

    for cond_name in cond_names:
        cond_dir = manifests_root / cond_name
        if not cond_dir.is_dir():
            print(f"[WARN] Condition directory not found, skipping: {cond_dir}")
            continue

        records = build_labels_for_condition(cond_dir)
        if not records:
            print(f"[WARN] No records found for condition {cond_name}")
            continue

        all_records.extend(records)

        # Write per-condition labels CSV
        df_cond = pd.DataFrame(records)
        df_cond = df_cond.sort_values(["ID", "fold"]).reset_index(drop=True)
        cond_out = output_root / f"EKSpression_labels_{cond_name}.csv"
        df_cond.to_csv(cond_out, index=False)
        print(f"Wrote {len(df_cond)} rows to {cond_out}")


if __name__ == "__main__":
    main()

"""
python datasets_process/EKSpression_prepare_labels.py \
--manifests-root ./d02_manifests \
--output-root ./data
"""