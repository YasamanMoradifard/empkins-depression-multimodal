#!/bin/bash
#SBATCH --job-name=mmf_cv
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19                 # 20 tasks: 5 conditions * 4 phases
#SBATCH --partition=tinyfat          # change if cluster uses a different partition name
#SBATCH --gres=gpu:1                 # request 1 GPU (must be gpu:NUMBER)
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G                   # adjust as needed
#SBATCH --time=12:00:00              # HH:MM:SS
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yasaman.moradi.fard@fau.de

# --- setup env (change to your environment)
module purge
module load anaconda/2021.11       # example, change per your cluster
source activate myenv              # or: conda activate mmf_env

# --- parameter lists (must match your desired order)
conditions=(SHAM CRADK ADK CR All)
phases=(training_pos training_neg induction1 induction2)

# compute indices from SLURM_ARRAY_TASK_ID
task_id=${SLURM_ARRAY_TASK_ID}
num_phases=${#phases[@]}           # 4
cond_idx=$(( task_id / num_phases ))
phase_idx=$(( task_id % num_phases ))

condition=${conditions[$cond_idx]}
phase=${phases[$phase_idx]}

echo "Task $task_id -> condition=$condition phase=$phase"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# optional: set CUDA device explicitly if needed (let scheduler decide usually)
/usr/bin/time -v python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
    --model MultiModalDepDet \
    --dataset d02_npy_downsampled \
    --condition "$condition" \
    --phase "$phase" \
    --modalities video \
    --train true \
    --fusion video \
    --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
    --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth \
    -lr 0.0005 -bs 16 -e 15 \
    --optimizer AdamW --weight_decay 0.001 --lr_scheduler cos \
    --visual_dropout 0.3 --classifier_dropout 0.0 \
    --attention_dropout 0.2 --transformer_dropout 0.2 \
    --lambda_reg 0.0001 --focal_weight 0.5 --l2_weight 0.7 --label_smoothing 0.2 \
    --early_stopping_patience 3 --early_stopping_delta 0.001 \
    --device cuda \
    --num_folds 5 --start_fold 0
