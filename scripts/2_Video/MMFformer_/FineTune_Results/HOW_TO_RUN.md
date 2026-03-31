# How to Run MMFformer_finetuning.py

## Quick Start

### 1. Basic Run (Test Current Configuration)
```bash
cd /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts
python MMFformer_finetuning.py
```

This will test the hyperparameters defined in `HYPERPARAMETER_SEARCH_SPACES` (currently 2 learning rates).

### 2. Use Best Hyperparameters from Previous Run
```bash
python MMFformer_finetuning.py --use_best
```

This loads and runs with the best hyperparameters saved from previous runs.

### 3. Test with Custom Search Spaces (JSON file)
```bash
python MMFformer_finetuning.py --search_spaces_file ../FineTune_Results/example_search_spaces.json
```

### 4. Test with Command-Line Parameters
```bash
python MMFformer_finetuning.py --param learning_rate '[1e-5, 5e-4]' --param batch_size '[16, 32]'
```

### 5. Dry Run (See Commands Without Executing)
```bash
python MMFformer_finetuning.py --dry_run
```

### 6. Test Only First Combination (Quick Test)
```bash
python MMFformer_finetuning.py --test_one
```

## Time Estimation

### Per Experiment Time
- **Default configuration**: 35 epochs
- **Estimated time per experiment**: 
  - With GPU (CUDA): ~30-60 minutes per experiment
  - Without GPU (CPU): ~3-6 hours per experiment (not recommended)
- **Timeout per experiment**: 1 hour (3600 seconds)

### Total Time Calculation

**Current configuration** (2 learning rates):
- **Total experiments**: 2
- **Estimated total time**: 1-2 hours (with GPU)

**Example with more hyperparameters**:
```json
{
  "learning_rate": [1e-5, 5e-4],
  "batch_size": [16, 32],
  "weight_decay": [1e-4, 1e-3]
}
```
- **Total combinations**: 2 × 2 × 2 = **8 experiments**
- **Estimated total time**: 4-8 hours (with GPU)

**Large search space example**:
```json
{
  "learning_rate": [1e-5, 5e-4, 1e-3],
  "batch_size": [8, 16, 32],
  "weight_decay": [1e-4, 1e-3, 1e-2],
  "fusion_dropout": [0.3, 0.5, 0.7]
}
```
- **Total combinations**: 3 × 3 × 3 × 3 = **81 experiments**
- **Estimated total time**: 40-80 hours (with GPU) ⚠️ **Very long!**

## Factors Affecting Runtime

1. **Number of epochs**: More epochs = longer training
   - Default: 35 epochs
   - Can reduce to 10-20 for quick tests

2. **Batch size**: Smaller batch = slower per epoch
   - Default: 16
   - Larger batches (32, 64) train faster per epoch

3. **Dataset size**: Larger datasets take longer
   - Your dataset: d02

4. **Early stopping**: Can stop early if not improving
   - Default patience: 5 epochs
   - Saves time if model stops improving

5. **GPU availability**: GPU is **much faster** than CPU
   - Script uses CUDA by default

## Recommended Workflow

### Step 1: Quick Test (5-10 minutes)
```bash
# Test with just one combination to verify everything works
python MMFformer_finetuning.py --test_one
```

### Step 2: Small Search (1-2 hours)
```bash
# Test 2-4 combinations
python MMFformer_finetuning.py --param learning_rate '[1e-5, 5e-4]'
```

### Step 3: Medium Search (4-8 hours)
```bash
# Test 8-16 combinations
python MMFformer_finetuning.py --search_spaces_file example_search_spaces.json
```

### Step 4: Use Best Results
```bash
# Run final training with best hyperparameters
python MMFformer_finetuning.py --use_best
```

## Running in Background (Long Runs)

For long-running experiments, use `nohup` or `screen`:

### Using nohup:
```bash
nohup python MMFformer_finetuning.py > ../FineTune_Results/training.log 2>&1 &
```

### Using screen:
```bash
screen -S hyperparameter_tuning
python MMFformer_finetuning.py
# Press Ctrl+A then D to detach
# Reattach with: screen -r hyperparameter_tuning
```

## Monitoring Progress

The script will:
1. Print progress for each experiment
2. Show parsed metrics (accuracy, F1) after each run
3. Save results to JSON files in `FineTune_Results/`
4. Save best hyperparameters to `best_hyperparameters.json`

### Check Results
```bash
# View latest results
ls -lt ../FineTune_Results/debug_results_*.json | head -1

# View best hyperparameters
cat ../FineTune_Results/best_hyperparameters.json
```

## Tips to Reduce Runtime

1. **Start small**: Test with `--test_one` first
2. **Reduce epochs**: Set `"epochs": [10, 20]` for quick tests
3. **Use early stopping**: Already enabled (patience=5)
4. **Test fewer hyperparameters**: Focus on most important ones
5. **Use best hyperparameters**: After finding good ones, use `--use_best` for final training

## Troubleshooting

### If experiments timeout:
- Increase timeout in script (line 352) or reduce epochs
- Check GPU memory usage
- Reduce batch size

### If out of memory:
- Reduce batch size
- Reduce number of parallel experiments (run sequentially)

### To stop running experiments:
- Press `Ctrl+C` (will stop after current experiment)
- Kill process: `pkill -f MMFformer_finetuning.py`

