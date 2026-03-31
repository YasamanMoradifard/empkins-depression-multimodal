"""
Script to check NPY files in d02_npy folder against merged_RCT_info.csv.

Checks:
1. Visual folder: Counts files per participant ID and compares with total_files
2. Reports any discrepancies (missing or extra files)
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def extract_participant_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract participant ID from filename.
    Expected format: {ID}_{...}_visual.npy or {ID}_{...}.npy
    ID can be zero-padded (e.g., 004, 005) or not (e.g., 4, 5)
    """
    # Remove extension
    stem = Path(filename).stem
    
    # Try to match ID at the start (with or without leading zeros)
    match = re.match(r'^(\d+)', stem)
    if match:
        return match.group(1)
    return None


def normalize_pid(pid: str) -> str:
    """Normalize participant ID by removing leading zeros for comparison."""
    return str(int(pid)) if pid.isdigit() else pid


def load_rct_info(csv_path: Path) -> Dict[str, Dict]:
    """
    Load merged_RCT_info.csv and return a dictionary mapping normalized ID to info.
    
    Returns:
        Dict mapping normalized ID (e.g., "4") to {"ID": "004", "total_files": 42, ...}
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"RCT info file not found: {csv_path}")
    
    # Try different encodings
    df = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"[INFO] Loaded RCT info with encoding: {encoding}")
            break
        except Exception as e:
            continue
    
    if df is None:
        raise ValueError(f"Failed to read RCT info CSV: {csv_path}")
    
    # Normalize column names (case-insensitive)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Find ID and total_files columns
    id_col = None
    num_folders_col = None
    
    for col in df.columns:
        if col == "id" or col.startswith("id"):
            id_col = col
        if "total_files" in col or "numfolders" in col or "folders" in col:
            num_folders_col = col
    
    if not id_col:
        raise ValueError("Could not find 'ID' column in RCT info CSV")
    if not num_folders_col:
        raise ValueError("Could not find 'num_folders' column in RCT info CSV")
    
    rct_info = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        num_folders = row[num_folders_col]
        
        # Try to convert num_folders to int
        try:
            num_folders = int(float(num_folders)) if pd.notna(num_folders) else 0
        except (ValueError, TypeError):
            num_folders = 0
        
        if not raw_id or raw_id.lower() in ("nan", "none", ""):
            continue
        
        # Normalize ID for comparison
        normalized_id = normalize_pid(raw_id)
        
        rct_info[normalized_id] = {
            "ID": raw_id,  # Keep original format
            "total_files": num_folders,
            "row_data": row.to_dict()
        }
    
    print(f"[INFO] Loaded info for {len(rct_info)} participants from RCT CSV")
    return rct_info


def count_files_by_participant(folder: Path, pattern: str = "*.npy") -> Dict[str, List[str]]:
    """
    Count files in folder grouped by participant ID.
    
    Returns:
        Dict mapping normalized participant ID to list of filenames
    """
    if not folder.exists():
        return {}
    
    files_by_pid = defaultdict(list)
    
    for file_path in folder.glob(pattern):
        pid = extract_participant_id_from_filename(file_path.name)
        if pid:
            normalized_pid = normalize_pid(pid)
            files_by_pid[normalized_pid].append(file_path.name)
    
    return dict(files_by_pid)


def check_folder(
    folder: Path,
    folder_name: str,
    rct_info: Dict[str, Dict],
    log_file,
    pattern: str = "*.npy"
) -> Dict[str, Dict]:
    """
    Check a folder against RCT info.
    
    Returns:
        Dict with statistics and issues found
    """
    print(f"\n[INFO] Checking {folder_name} folder: {folder}")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Checking: {folder_name} folder\n")
    log_file.write(f"Path: {folder}\n")
    log_file.write(f"{'='*80}\n\n")
    
    files_by_pid = count_files_by_participant(folder, pattern)
    
    stats = {
        "total_participants_found": len(files_by_pid),
        "total_files": sum(len(files) for files in files_by_pid.values()),
        "participants_in_rct": 0,
        "participants_not_in_rct": 0,
        "matches": 0,
        "mismatches": 0,
        "missing_participants": 0,
        "extra_participants": 0,
    }
    
    issues = []
    
    # Check participants found in folder
    for normalized_pid, files in sorted(files_by_pid.items()):
        file_count = len(files)
        
        if normalized_pid in rct_info:
            stats["participants_in_rct"] += 1
            expected_count = rct_info[normalized_pid]["total_files"]
            original_id = rct_info[normalized_pid]["ID"]
            
            if file_count == expected_count:
                stats["matches"] += 1
                log_file.write(f"✓ ID {original_id} (normalized: {normalized_pid}): {file_count} files (expected: {expected_count}) - OK\n")
            else:
                stats["mismatches"] += 1
                issue = {
                    "type": "count_mismatch",
                    "pid": original_id,
                    "normalized_pid": normalized_pid,
                    "expected": expected_count,
                    "found": file_count,
                    "difference": file_count - expected_count
                }
                issues.append(issue)
                log_file.write(f"✗ ID {original_id} (normalized: {normalized_pid}): {file_count} files (expected: {expected_count}) - MISMATCH\n")
                if file_count < expected_count:
                    log_file.write(f"  → Missing {expected_count - file_count} file(s)\n")
                else:
                    log_file.write(f"  → Extra {file_count - expected_count} file(s)\n")
        else:
            stats["participants_not_in_rct"] += 1
            stats["extra_participants"] += 1
            issue = {
                "type": "not_in_rct",
                "pid": normalized_pid,
                "file_count": file_count
            }
            issues.append(issue)
            log_file.write(f"⚠ ID {normalized_pid}: {file_count} files - NOT FOUND IN RCT CSV\n")
    
    # Check participants in RCT but not in folder
    for normalized_pid, info in rct_info.items():
        if normalized_pid not in files_by_pid:
            stats["missing_participants"] += 1
            issue = {
                "type": "missing_participant",
                "pid": info["ID"],
                "normalized_pid": normalized_pid,
                "expected_files": info["total_files"]
            }
            issues.append(issue)
            log_file.write(f"✗ ID {info['ID']} (normalized: {normalized_pid}): MISSING - Expected {info['total_files']} files\n")
    
    # Write summary
    log_file.write(f"\n--- Summary for {folder_name} ---\n")
    log_file.write(f"Total participants found: {stats['total_participants_found']}\n")
    log_file.write(f"Total files: {stats['total_files']}\n")
    log_file.write(f"Participants in RCT: {stats['participants_in_rct']}\n")
    log_file.write(f"Participants not in RCT: {stats['participants_not_in_rct']}\n")
    log_file.write(f"Count matches: {stats['matches']}\n")
    log_file.write(f"Count mismatches: {stats['mismatches']}\n")
    log_file.write(f"Missing participants: {stats['missing_participants']}\n")
    log_file.write(f"Extra participants: {stats['extra_participants']}\n")
    log_file.write(f"Total issues: {len(issues)}\n")
    
    return {"stats": stats, "issues": issues}


def main():
    parser = argparse.ArgumentParser(
        description="Check NPY files in d02_npy folder against merged_RCT_info.csv"
    )
    parser.add_argument(
        "--d02-npy-root",
        type=Path,
        default=Path("/data/d02_npy"),
        help="Root directory of d02_npy folder (default: /data/d02_npy)",
    )
    parser.add_argument(
        "--rct-info-csv",
        type=Path,
        default=Path("merged_RCT_info.csv"),
        help="Path to merged_RCT_info.csv file (default: merged_RCT_info.csv)",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=Path("EKSpression_checking.txt"),
        help="Output log file (default: EKSpression_checking.txt)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV file for mismatched IDs (default: mismatched_ids.csv in same directory as log file)",
    )
    args = parser.parse_args()
    
    # Load RCT info
    print(f"[INFO] Loading RCT info from: {args.rct_info_csv}")
    try:
        rct_info = load_rct_info(args.rct_info_csv)
    except Exception as e:
        print(f"[ERROR] Failed to load RCT info: {e}")
        return
    
    # Open log file
    log_file = args.output_log.open("w", encoding="utf-8")
    log_file.write("EKSpression NPY Files Checking Report\n")
    log_file.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"d02_npy root: {args.d02_npy_root}\n")
    log_file.write(f"RCT info CSV: {args.rct_info_csv}\n")
    log_file.write(f"\n")
    
    all_issues = []
    all_stats = {}
    
    # Check visual folder
    visual_folder = args.d02_npy_root / "visual"
    if visual_folder.exists():
        result = check_folder(visual_folder, "visual", rct_info, log_file, "*.npy")
        all_stats["visual"] = result["stats"]
        all_issues.extend([("visual", issue) for issue in result["issues"]])
    else:
        log_file.write(f"\n[WARN] Visual folder not found: {visual_folder}\n")
    
    # Overall summary
    log_file.write(f"\n\n{'='*80}\n")
    log_file.write("OVERALL SUMMARY\n")
    log_file.write(f"{'='*80}\n\n")
    
    total_issues = len(all_issues)
    log_file.write(f"Total issues found across all folders: {total_issues}\n\n")
    
    if total_issues == 0:
        log_file.write("✓ All checks passed! No issues found.\n")
    else:
        log_file.write("Issues by type:\n")
        issue_types = defaultdict(int)
        for folder_name, issue in all_issues:
            issue_types[issue["type"]] += 1
        
        for issue_type, count in sorted(issue_types.items()):
            log_file.write(f"  - {issue_type}: {count}\n")
    
    log_file.write(f"\nReport saved to: {args.output_log}\n")
    log_file.close()
    
    # Export mismatched IDs to CSV
    if total_issues > 0:
        # Determine CSV output path
        if args.output_csv:
            csv_output_path = args.output_csv
        else:
            # Default to same directory as log file with name "mismatched_ids.csv"
            csv_output_path = args.output_log.parent / "mismatched_ids.csv"
        
        # Collect all mismatched IDs (one row per ID-folder combination)
        mismatch_records = []
        
        for folder_name, issue in all_issues:
            # Only include count_mismatch and missing_participant issues
            if issue["type"] in ("count_mismatch", "missing_participant"):
                pid = issue.get("pid", "")
                normalized_pid = issue.get("normalized_pid", "")
                
                record = {
                    "ID": pid,
                    "Normalized_ID": normalized_pid,
                    "Folder": folder_name,
                    "Issue_Type": issue["type"],
                    "Expected_Files": issue.get("expected", issue.get("expected_files", "")),
                    "Found_Files": issue.get("found", ""),
                    "Difference": issue.get("difference", "")
                }
                mismatch_records.append(record)
        
        # Write CSV file
        if mismatch_records:
            # Sort by ID and folder for better readability
            mismatch_records.sort(key=lambda x: (x["Normalized_ID"], x["Folder"]))
            
            with csv_output_path.open("w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["ID", "Normalized_ID", "Folder", "Issue_Type", "Expected_Files", "Found_Files", "Difference"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mismatch_records)
            
            # Count unique IDs
            unique_ids = set(r["Normalized_ID"] for r in mismatch_records)
            print(f"[INFO] Mismatched IDs exported to: {csv_output_path}")
            print(f"[INFO] Total mismatched ID-folder combinations: {len(mismatch_records)}")
            print(f"[INFO] Unique mismatched IDs: {len(unique_ids)}")
        else:
            print(f"[INFO] No mismatched IDs to export (only non-RCT participants found)")
    
    print(f"\n[INFO] Checking complete. Report saved to: {args.output_log}")
    print(f"[INFO] Total issues found: {total_issues}")


if __name__ == "__main__":
    main()


# runing samples:

"""
python datasets_process/EKSpresseion_checking.py \
--d02-npy-root /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/d02_video_npy \
--rct-info-csv /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_manifests/merged_RCT_info.csv \
--output-log /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/EKSpression_checking.txt \
--output-csv /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data/mismatched_ids.csv
"""