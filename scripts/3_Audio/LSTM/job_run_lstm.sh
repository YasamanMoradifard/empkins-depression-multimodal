#!/bin/bash -l
#SBATCH --partition=a100               # GPU partition
#SBATCH --gres=gpu:a100:1              # 1 A100 GPU
#SBATCH -C a100_80                     # specifically the 80 GB GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=audio_lstm
#SBATCH --mail-user=yasaman.moradi.fard@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# LSTM DEPRESSION DETECTION - UNIFIED TRAINING SCRIPT
# =============================================================================
# This script supports two modes:
#   1. OPTIMIZED MODE (recommended for small datasets ~50-100 subjects)
#      - Simplified model architecture
#      - Strong regularization (high dropout, weight decay)
#      - Feature reduction (PCA)
#      - LR scheduler and label smoothing
#
#   2. STANDARD MODE (for larger datasets or hyperparameter exploration)
#      - Full model architecture
#      - Nested CV with hyperparameter tuning
#      - Default regularization
# =============================================================================

# Load necessary modules
module load cuda/11.8.0
module load python
source /home/hpc/iwso/iwso170h/MT/venv/bin/activate

# Ensure Python output is not buffered
export PYTHONUNBUFFERED=1

# =============================================================================
# MAIN CONFIGURATION - EDIT THESE SETTINGS
# =============================================================================

# ---- MODE SELECTION ----
# Set to "optimized" for small datasets, "standard" for larger datasets/HP tuning
MODE="optimized"  # Options: "optimized", "standard"

# ---- DATA SELECTION ----
CONDITIONS=("All")                       # Options: CR, CRADK, ADK, SHAM, All
PHASES=("all")                           # Options: training_pos, training_neg, induction1, induction2, all

# ---- AGGREGATION METHOD ----
# For small datasets: "mean" recommended (fewest parameters)
# For larger datasets: "attention" or "lstm" can capture more patterns
AGGREGATION="mean"                       # Options: attention, lstm, mean, max, mean_max

# =============================================================================
# MODE-SPECIFIC DEFAULTS (you can override these below)
# =============================================================================

if [ "$MODE" == "optimized" ]; then
    # ----- OPTIMIZED MODE SETTINGS (for small datasets) -----
    CV_TYPE="simple"
    N_FOLDS=3
    
    # Training
    NUM_EPOCHS=150
    EARLY_STOPPING_PATIENCE=20
    BATCH_SIZE=2
    LEARNING_RATE="1e-5"
    
    # Model architecture - SIMPLIFIED
    LSTM_HIDDEN_DIM=32
    FILE_EMBEDDING_DIM=16
    SUBJECT_EMBEDDING_DIM=16
    DROPOUT_LSTM=0.7
    DROPOUT_CLASSIFIER=0.7
    
    # Anti-overfitting features
    USE_SIMPLE_LSTM=true
    USE_FEATURE_REDUCTION=true
    FEATURE_REDUCTION_METHOD="pca"
    N_FEATURES_REDUCED=30
    USE_LR_SCHEDULER=true
    LR_SCHEDULER_PATIENCE=7
    LABEL_SMOOTHING=0.1
    
    # No hyperparameter tuning in optimized mode (use fixed params)
    TUNE_HP=false
    
    RESULTS_SUBDIR="results_optimized"
    
else
    # ----- STANDARD MODE SETTINGS (for HP exploration) -----
    CV_TYPE="nested"
    N_FOLDS=3
    N_OUTER_FOLDS=5
    N_INNER_FOLDS=3
    
    # Training
    NUM_EPOCHS=100
    EARLY_STOPPING_PATIENCE=15
    BATCH_SIZE=4
    LEARNING_RATE="1e-4"
    
    # Model architecture - FULL
    LSTM_HIDDEN_DIM=128
    FILE_EMBEDDING_DIM=64
    SUBJECT_EMBEDDING_DIM=64
    DROPOUT_LSTM=0.3
    DROPOUT_CLASSIFIER=0.5
    
    # Standard features (no special anti-overfitting)
    USE_SIMPLE_LSTM=false
    USE_FEATURE_REDUCTION=false
    FEATURE_REDUCTION_METHOD="pca"
    N_FEATURES_REDUCED=30
    USE_LR_SCHEDULER=false
    LR_SCHEDULER_PATIENCE=7
    LABEL_SMOOTHING=0.0
    
    # Enable hyperparameter tuning for nested CV
    TUNE_HP=true
    
    RESULTS_SUBDIR="results"
fi

# =============================================================================
# OPTIONAL: OVERRIDE SPECIFIC SETTINGS HERE
# =============================================================================
# Uncomment and modify any settings you want to override:
#
# CV_TYPE="simple"                       # Force simple CV
# NUM_EPOCHS=200                         # More epochs
# BATCH_SIZE=4                           # Larger batch
# USE_SIMPLE_LSTM=true                   # Force simple LSTM
# USE_FEATURE_REDUCTION=true             # Force PCA
# N_FEATURES_REDUCED=50                  # More features
# LABEL_SMOOTHING=0.2                    # More smoothing
# TUNE_HP=true                           # Enable HP tuning in nested CV

# =============================================================================
# GPU VERIFICATION (DO NOT EDIT BELOW THIS LINE)
# =============================================================================

echo "=========================================="
echo "GPU Verification"
echo "=========================================="
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

# Check if nvidia-smi is available and GPU is accessible
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"
else
    echo "WARNING: nvidia-smi not found - cannot verify GPU"
fi

# Verify CUDA is available via Python
echo ""
echo "Verifying CUDA availability via Python..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    echo "The script will fail when it tries to use GPU."
    exit 1
}

echo "✓ GPU verification passed"
echo "=========================================="
echo ""

# Change to the LSTM directory
cd "/home/vault/empkins/tpD/D02/Students/Yasaman/3_Audio_data/LSTM"

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

echo "=========================================="
echo "LSTM Training Configuration"
echo "=========================================="
echo "Mode: ${MODE^^}"
echo ""
echo "Data Settings:"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Phases: ${PHASES[*]}"
echo ""
echo "Model Settings:"
echo "  Aggregation: ${AGGREGATION}"
echo "  Simple LSTM: ${USE_SIMPLE_LSTM}"
echo "  LSTM Hidden: ${LSTM_HIDDEN_DIM}"
echo "  File Embedding: ${FILE_EMBEDDING_DIM}"
echo "  Dropout LSTM: ${DROPOUT_LSTM}"
echo ""
echo "Training Settings:"
echo "  CV Type: ${CV_TYPE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Early Stop Patience: ${EARLY_STOPPING_PATIENCE}"
echo ""
echo "Anti-Overfitting Features:"
echo "  Feature Reduction: ${USE_FEATURE_REDUCTION} (${FEATURE_REDUCTION_METHOD} -> ${N_FEATURES_REDUCED})"
echo "  LR Scheduler: ${USE_LR_SCHEDULER}"
echo "  Label Smoothing: ${LABEL_SMOOTHING}"
echo ""
echo "Results Directory: ${RESULTS_SUBDIR}/"
echo "=========================================="
echo ""

# =============================================================================
# BUILD COMMAND ARGUMENTS
# =============================================================================

# CV arguments
if [ "$CV_TYPE" == "simple" ]; then
    CV_ARGS="--n_folds ${N_FOLDS}"
else
    CV_ARGS="--n_outer_folds ${N_OUTER_FOLDS} --n_inner_folds ${N_INNER_FOLDS}"
fi

# Optional flags
OPTIONAL_ARGS=""

if [ "$USE_SIMPLE_LSTM" == true ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --use_simple_lstm"
fi

if [ "$USE_FEATURE_REDUCTION" == true ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --use_feature_reduction --feature_reduction_method ${FEATURE_REDUCTION_METHOD} --n_features_reduced ${N_FEATURES_REDUCED}"
fi

if [ "$USE_LR_SCHEDULER" == true ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --use_lr_scheduler --lr_scheduler_patience ${LR_SCHEDULER_PATIENCE}"
fi

# HP tuning flags (only for nested CV)
HP_ARGS=""
if [ "$TUNE_HP" == true ] && [ "$CV_TYPE" == "nested" ]; then
    HP_ARGS="--tune_learning_rate --tune_lstm_hidden_dim --tune_dropout_lstm"
    echo "Hyperparameter tuning enabled: ${HP_ARGS}"
    echo ""
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

for CONDITION in "${CONDITIONS[@]}"; do
    for PHASE in "${PHASES[@]}"; do
        echo "=========================================="
        echo "Training: ${CONDITION} / ${PHASE}"
        echo "=========================================="
        
        # Create results directory name
        RESULTS_DIR="${RESULTS_SUBDIR}/${CV_TYPE}_${CONDITION}_${PHASE}_agg_${AGGREGATION}"
        
        echo "Results: ${RESULTS_DIR}"
        echo ""
        
        python3 main.py \
            --cv_type ${CV_TYPE} \
            --condition ${CONDITION} \
            --phase ${PHASE} \
            --aggregation ${AGGREGATION} \
            ${CV_ARGS} \
            --num_epochs ${NUM_EPOCHS} \
            --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --lstm_hidden_dim ${LSTM_HIDDEN_DIM} \
            --file_embedding_dim ${FILE_EMBEDDING_DIM} \
            --subject_embedding_dim ${SUBJECT_EMBEDDING_DIM} \
            --dropout_lstm ${DROPOUT_LSTM} \
            --dropout_classifier ${DROPOUT_CLASSIFIER} \
            --label_smoothing ${LABEL_SMOOTHING} \
            ${OPTIONAL_ARGS} \
            ${HP_ARGS} \
            --device cuda \
            --results_dir ${RESULTS_DIR} || {
            echo "ERROR: Training failed for ${CONDITION}/${PHASE}!"
            exit 1
        }
        
        echo ""
        echo "✓ Completed: ${CONDITION} / ${PHASE}"
        echo ""
    done
done

echo "=========================================="
echo "✓ All training runs completed successfully!"
echo "Results saved to ${RESULTS_SUBDIR}/ directory"
echo "=========================================="
