#!/usr/bin/env python3
"""
Simplified Fine-tuning Script for debugging
"""

import os
import sys
import subprocess
import argparse
import json
import re
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

ENTRYPOINT = "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/main.py"
RESULTS_DIR = Path("/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/FineTune_Results")
BEST_HYPERPARAMS_FILE = RESULTS_DIR / "best_hyperparameters.json"

# Base arguments
BASE_ARGS = {
    "dataset": "d02",
    "model": "MultiModalDepDet",
    "train": True,
    "fusion": "video",
    "num_heads": 8,
    "condition": "CRADK",
    "phase": "training_pos",
    "modalities": "video",
    "device": "cuda",
    "tqdm_able": False,
    "if_wandb": False,
    "train_gender": "both",
    "test_gender": "both",
    "resume_path": "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth",
    "data_dir": "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data",  # ADD THIS!
}

# Default values
# All hyperparameters that can be tuned (matching main.py arguments)
DEFAULT_VALUES = {
    # Training hyperparameters
    "learning_rate": 5e-4,
    "batch_size": 16,
    "epochs": 15,
    "optimizer": "Adam",
    "weight_decay": 1e-3,
    "lr_scheduler": "cos",
    "lr_patience": 10,
    "lr_steps": [100, 200, 300, 400, 550, 700],  # For StepLR scheduler
    "begin_epoch": 1,  # Starting epoch (for resuming)
    
    # Optimizer-specific hyperparameters
    "amsgrad": 0,  # For Adam/AdamW
    "momentum": 0.9,  # For SGD
    "dampening": 0.9,  # For SGD
    
    # Dropout hyperparameters (regularization)
    "fusion_dropout": 0.5,
    "audio_dropout": 0.5,
    "visual_dropout": 0.5,
    "classifier_dropout": 0.0,
    "attention_dropout": 0.1,
    "transformer_dropout": 0.1,
    
    # Loss function hyperparameters (regularization)
    "lambda_reg": 1e-5,
    "focal_weight": 0.5,
    "l2_weight": 0.5,
    "label_smoothing": 0.1,
    
    # Early stopping hyperparameters
    "early_stopping_patience": 5,
    "early_stopping_delta": 0.0,
}

# Hyperparameter search spaces
# Define which hyperparameters to test and their possible values
# All hyperparameters from main.py (lines 81-157) are supported

HYPERPARAMETER_SEARCH_SPACES = {
    "learning_rate": [5e-4],
    "batch_size": [16],
    "epochs": [50],
    "optimizer": ["AdamW"],
    "weight_decay": [1e-3],
    "lr_scheduler": ["cos"],
    "visual_dropout": [0.3],
    "classifier_dropout": [0.0],
    "attention_dropout": [0.2],
    "transformer_dropout": [0.2],
    "lambda_reg": [1e-4],
    "focal_weight": [0.5],
    "l2_weight": [0.7],
    "label_smoothing": [0.2],
    "early_stopping_patience": [3],
    "early_stopping_delta": [0.001],
}
# HYPERPARAMETER_SEARCH_SPACES = {
#     # Training hyperparameters
#     "learning_rate": [5e-4],
#     "batch_size": [16],
#     "epochs": [50],
#     "optimizer": ["AdamW"],
#     "weight_decay": [0.1],
#     "lr_scheduler": ["cos"],
#     #"lr_patience": [5, 10, 15],  # For Plateau scheduler / not important
#     # "lr_steps": [[100, 200], [200, 400]],  # For StepLR scheduler (list of lists) / not important
#     # "begin_epoch": [1],  # Usually 1, but can be set for resuming / not important
    
#     # Optimizer-specific hyperparameters
#     #"amsgrad": [0, 1],  # For Adam/AdamW (0 or 1)
#     # "momentum": [0.8, 0.9, 0.95],  # For SGD / not important
#     # "dampening": [0.8, 0.9, 0.95],  # For SGD / not important
    
#     # Dropout hyperparameters (regularization)
#     #"fusion_dropout": [0.3, 0.5, 0.7], / not important
#     # "audio_dropout": [0.3, 0.5, 0.7], / not important
#     "visual_dropout": [0.1],
#     "classifier_dropout": [0.0],
#     # "attention_dropout": [0.0, 0.1, 0.2], / not important
#     # "transformer_dropout": [0.0, 0.1, 0.2], / not important
    
#     # Loss function hyperparameters (regularization)
#     "lambda_reg": [1e-5],
#     "focal_weight": [0.5],
#     "l2_weight": [0.5],
#     # "label_smoothing": [0.0, 0.1, 0.2],
    
#     # Early stopping hyperparameters
#     # "early_stopping_patience": [3, 5, 7], / not important
#     # "early_stopping_delta": [0.0, 0.001, 0.01], /not important
# }

# # Learning rates to test (kept for backward compatibility)
# LEARNING_RATES = [5e-4]



def parse_test_metrics(stdout: str, stderr: str) -> Optional[Dict[str, float]]:
    """Parse test accuracy and F1 score from stdout/stderr"""
    full_output = stdout + stderr
    
    # Try to find test results in various formats
    # Pattern 1: "Test Accuracy: 0.xxxx" and "Test F1: 0.xxxx"
    acc_pattern = r'Test\s+Accuracy[:\s]+([0-9]+\.[0-9]+)'
    f1_pattern = r'Test\s+F1[:\s]+([0-9]+\.[0-9]+)'
    
    # Pattern 2: "Accuracy:0.xxxx" (from results file format)
    acc_pattern2 = r'Accuracy[:\s]+([0-9]+\.[0-9]+)'
    f1_pattern2 = r'F1[:\s]+([0-9]+\.[0-9]+)'
    
    test_acc = None
    test_f1 = None
    
    # Try pattern 1 first
    acc_match = re.search(acc_pattern, full_output, re.IGNORECASE)
    f1_match = re.search(f1_pattern, full_output, re.IGNORECASE)
    
    if acc_match:
        test_acc = float(acc_match.group(1))
    if f1_match:
        test_f1 = float(f1_match.group(1))
    
    # If pattern 1 didn't work, try pattern 2 (but look for test context)
    if test_acc is None or test_f1 is None:
        # Look for test results section
        test_section = re.search(r'TEST\s+RESULTS?.*?(?=\n\n|\Z)', full_output, re.IGNORECASE | re.DOTALL)
        if test_section:
            section_text = test_section.group(0)
            acc_match2 = re.search(acc_pattern2, section_text, re.IGNORECASE)
            f1_match2 = re.search(f1_pattern2, section_text, re.IGNORECASE)
            
            if acc_match2 and test_acc is None:
                test_acc = float(acc_match2.group(1))
            if f1_match2 and test_f1 is None:
                test_f1 = float(f1_match2.group(1))
    
    if test_acc is not None and test_f1 is not None:
        return {"test_acc": test_acc, "test_f1": test_f1}
    
    return None


def load_best_hyperparameters() -> Optional[Dict[str, Any]]:
    """Load best hyperparameters from file"""
    if BEST_HYPERPARAMS_FILE.exists():
        try:
            with open(BEST_HYPERPARAMS_FILE, 'r') as f:
                data = json.load(f)
                return data.get("hyperparameters")
        except Exception as e:
            print(f"Warning: Could not load best hyperparameters: {e}")
    return None


def save_best_hyperparameters(hyperparams: Dict[str, Any], metrics: Dict[str, float]):
    """Save best hyperparameters to file"""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    data = {
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(BEST_HYPERPARAMS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Best hyperparameters saved to: {BEST_HYPERPARAMS_FILE}")
    print(f"Test Accuracy: {metrics['test_acc']:.4f}, Test F1: {metrics['test_f1']:.4f}")
    print(f"{'='*80}\n")


def generate_hyperparameter_combinations(search_spaces: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters (grid search)
    
    Args:
        search_spaces: Dictionary mapping hyperparameter names to lists of values
        
    Returns:
        List of dictionaries, each containing one combination of hyperparameters
    """
    if not search_spaces:
        return [{}]
    
    # Get all keys and their value lists
    keys = list(search_spaces.keys())
    value_lists = [search_spaces[key] for key in keys]
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*value_lists):
        hyperparams = {keys[i]: combination[i] for i in range(len(keys))}
        combinations.append(hyperparams)
    
    return combinations


def load_hyperparameter_search_spaces(file_path: Optional[Path] = None) -> Dict[str, List[Any]]:
    """
    Load hyperparameter search spaces from a JSON file
    
    Args:
        file_path: Path to JSON file. If None, uses default HYPERPARAMETER_SEARCH_SPACES
        
    Returns:
        Dictionary mapping hyperparameter names to lists of values
    """
    if file_path and file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load hyperparameter search spaces from {file_path}: {e}")
            print("Using default search spaces.")
    
    return HYPERPARAMETER_SEARCH_SPACES


def build_command(hyperparams: Dict[str, Any]) -> List[str]:
    """Build command-line arguments - SIMPLIFIED VERSION"""
    # Use sys.executable to use the same Python that's running this script
    cmd = [sys.executable, ENTRYPOINT]  # CHANGED FROM "python"
    
    # Merge all arguments
    all_args = {**BASE_ARGS, **DEFAULT_VALUES, **hyperparams}
    
    # Simple mapping - just convert to strings
    # Maps hyperparameter names to command-line flags (matching main.py)
    arg_mapping = {
        # Dataset and model configuration
        "dataset": "-ds",
        "model": "-m",
        "train": "-tr",
        "fusion": "-fus",
        "num_heads": "-n_h",
        "condition": "--condition",
        "phase": "--phase",
        "modalities": "--modalities",
        "device": "-d",
        "tqdm_able": "-tqdm",
        "if_wandb": "-wdb",
        "train_gender": "--train_gender",
        "test_gender": "--test_gender",
        "data_dir": "--data_dir",
        "resume_path": "--resume_path",
        
        # Training hyperparameters
        "learning_rate": "-lr",
        "batch_size": "-bs",
        "epochs": "-e",
        "optimizer": "--optimizer",
        "weight_decay": "--weight_decay",
        "lr_scheduler": "--lr_scheduler",
        "lr_patience": "--lr_patience",
        "lr_steps": "--lr_steps",  # For StepLR scheduler
        "begin_epoch": "--begin_epoch",
        
        # Optimizer-specific hyperparameters
        "amsgrad": "--amsgrad",
        "momentum": "--momentum",
        "dampening": "--dampening",
        
        # Dropout hyperparameters
        "fusion_dropout": "--fusion_dropout",
        "audio_dropout": "--audio_dropout",
        "visual_dropout": "--visual_dropout",
        "classifier_dropout": "--classifier_dropout",
        "attention_dropout": "--attention_dropout",
        "transformer_dropout": "--transformer_dropout",
        
        # Loss function hyperparameters
        "lambda_reg": "--lambda_reg",
        "focal_weight": "--focal_weight",
        "l2_weight": "--l2_weight",
        "label_smoothing": "--label_smoothing",
        
        # Early stopping hyperparameters
        "early_stopping_patience": "--early_stopping_patience",
        "early_stopping_delta": "--early_stopping_delta",
    }
    
    for key, flag in arg_mapping.items():
        if key in all_args:
            value = all_args[key]
            
            # Skip None and empty strings
            if value is None or (isinstance(value, str) and value == ""):
                continue
            
            # Handle lists (e.g., lr_steps)
            if isinstance(value, list):
                cmd.append(flag)
                for item in value:
                    cmd.append(str(item))
                continue
            
            # Convert boolean to lowercase string
            if isinstance(value, bool):
                value = "true" if value else "false"
            else:
                value = str(value)
            
            cmd.extend([flag, value])
    
    return cmd


def run_experiment(hyperparams: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Run a single experiment"""
    cmd = build_command(hyperparams)
    
    print(f"\n{'='*80}")
    print("COMMAND TO RUN:")
    print(' '.join(cmd))
    print(f"{'='*80}\n")
    
    if dry_run:
        return {"status": "dry_run", **hyperparams}
    
    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Get full output
        full_output = result.stdout + result.stderr
        
        print("STDOUT:")
        print(result.stdout[:2000])  # First 2000 chars
        print("\nSTDERR:")
        print(result.stderr[:2000])  # First 2000 chars
        
        # Parse test metrics
        metrics = parse_test_metrics(result.stdout, result.stderr)
        
        result_dict = {
            "status": "success",
            **hyperparams,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        
        if metrics:
            result_dict["metrics"] = metrics
            print(f"\n📊 Parsed Metrics:")
            print(f"   Test Accuracy: {metrics['test_acc']:.4f}")
            print(f"   Test F1: {metrics['test_f1']:.4f}")
        
        return result_dict
    
    except subprocess.CalledProcessError as e:
        # Get FULL error output
        full_stderr = e.stderr if e.stderr else ""
        full_stdout = e.stdout if e.stdout else ""
        
        print(f"\n{'='*80}")
        print("ERROR OCCURRED!")
        print(f"Return code: {e.returncode}")
        print(f"\nFULL STDERR ({len(full_stderr)} chars):")
        print("="*80)
        print(full_stderr)
        print("="*80)
        print(f"\nFULL STDOUT ({len(full_stdout)} chars):")
        print("="*80)
        print(full_stdout)
        print("="*80)
        
        return {
            "status": "failed",
            "error_code": e.returncode,
            "error_stderr": full_stderr,
            "error_stdout": full_stdout,
            **hyperparams
        }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            **hyperparams
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_msg": str(e),
            **hyperparams
        }


def main():
    parser = argparse.ArgumentParser(description="Simplified fine-tuning script")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--test_one", action="store_true", help="Test with just one hyperparameter combination")
    parser.add_argument("--use_best", action="store_true", help="Use best hyperparameters from previous runs")
    parser.add_argument("--ignore_best", action="store_true", help="Ignore saved best hyperparameters and run all experiments")
    parser.add_argument("--search_spaces_file", type=str, help="Path to JSON file with hyperparameter search spaces")
    parser.add_argument("--param", nargs=2, metavar=("NAME", "VALUES"), action="append",
                       help="Override a hyperparameter search space. Can be used multiple times. "
                            "Example: --param learning_rate '[1e-5, 5e-4]' --param batch_size '[16, 32]'")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("SIMPLIFIED FINE-TUNING SCRIPT")
    print(f"{'='*80}\n")
    
    # Load best hyperparameters if requested
    best_hyperparams = None
    if args.use_best and not args.ignore_best:
        best_hyperparams = load_best_hyperparameters()
        if best_hyperparams:
            print(f"\n{'='*80}")
            print("LOADED BEST HYPERPARAMETERS FROM PREVIOUS RUNS")
            print(f"{'='*80}")
            for key, value in best_hyperparams.items():
                print(f"  {key}: {value}")
            print(f"{'='*80}\n")
        else:
            print("\n⚠️  No saved best hyperparameters found. Will run all experiments.\n")
    
    # If using best hyperparameters, only run with those
    if best_hyperparams and args.use_best and not args.ignore_best:
        print(f"\n{'#'*80}")
        print("RUNNING WITH BEST HYPERPARAMETERS")
        print(f"{'#'*80}\n")
        
        result = run_experiment(best_hyperparams, dry_run=args.dry_run)
        
        if result["status"] == "success":
            print(f"\n✅ SUCCESS with best hyperparameters")
        else:
            print(f"\n❌ FAILED with best hyperparameters")
        
        return
    
    # Load hyperparameter search spaces
    search_spaces_file = Path(args.search_spaces_file) if args.search_spaces_file else None
    search_spaces = load_hyperparameter_search_spaces(search_spaces_file)
    
    # Override with command-line parameters if provided
    if args.param:
        for param_name, param_values_str in args.param:
            try:
                # Parse the values string (expecting JSON-like list format)
                param_values = json.loads(param_values_str)
                if isinstance(param_values, list):
                    search_spaces[param_name] = param_values
                    print(f"Overriding {param_name} search space: {param_values}")
                else:
                    print(f"Warning: {param_name} values must be a list, got {type(param_values)}")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse values for {param_name}: {param_values_str}")
    
    # Generate all hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations(search_spaces)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Search spaces:")
    for param, values in search_spaces.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations to test: {len(hyperparameter_combinations)}")
    print(f"{'='*80}\n")
    
    if args.test_one:
        # Just test the first combination
        hyperparameter_combinations = hyperparameter_combinations[:1]
        print("⚠️  Testing only the first combination (--test_one flag)\n")
    
    results = []
    best_metrics = None
    best_result = None
    
    for idx, hyperparams in enumerate(hyperparameter_combinations, 1):
        print(f"\n{'#'*80}")
        print(f"Experiment {idx}/{len(hyperparameter_combinations)}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'#'*80}")
        
        result = run_experiment(hyperparams, dry_run=args.dry_run)
        results.append(result)
        
        if result["status"] == "failed":
            print(f"\n❌ FAILED with hyperparameters: {hyperparams}")
            print("Check the error output above for details.")
            # Stop on first failure for debugging
            if args.test_one:
                break
        elif result["status"] == "success" and "metrics" in result:
            print(f"\n✅ SUCCESS with hyperparameters: {hyperparams}")
            
            # Check if this is the best result
            metrics = result["metrics"]
            if best_metrics is None:
                best_metrics = metrics
                best_result = result
                print(f"🏆 FIRST RESULT (Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f})")
            else:
                # Compare: prefer higher accuracy, then higher F1
                if (metrics["test_acc"] > best_metrics["test_acc"]) or \
                   (metrics["test_acc"] == best_metrics["test_acc"] and 
                    metrics["test_f1"] > best_metrics["test_f1"]):
                    best_metrics = metrics
                    best_result = result
                    print(f"🏆 NEW BEST RESULT! (Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f})")
        else:
            print(f"\n✅ SUCCESS with hyperparameters: {hyperparams} (metrics not parsed)")
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Save best hyperparameters if we found them
    if best_result and best_metrics:
        best_hyperparams_to_save = {k: v for k, v in best_result.items() 
                                   if k not in ["status", "stdout", "stderr", "metrics"]}
        save_best_hyperparameters(best_hyperparams_to_save, best_metrics)
    else:
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()