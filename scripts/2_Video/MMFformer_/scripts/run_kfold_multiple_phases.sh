#!/bin/bash

# Configuration
CONDITION="CRADK"  # Change this to your desired condition
PHASES=("training_pos" "training_neg" "induction1" "induction2")
NUM_FOLDS=10
START_FOLD=0  # Start from fold 0 (or change if resuming)

# Base command arguments (customize these)
BASE_ARGS=(
    --model MultiModalDepDet
    --dataset d02_npy_downsampled
    --condition "$CONDITION"
    --modalities video
    --train true
    --fusion video
    --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data
    --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth
    -lr 0.0005
    -bs 16
    -e 50
    --optimizer AdamW
    --weight_decay 0.001
    --lr_scheduler cos
    --visual_dropout 0.3
    --classifier_dropout 0.0
    --attention_dropout 0.2
    --transformer_dropout 0.2
    --lambda_reg 0.0001
    --focal_weight 0.5
    --l2_weight 0.7
    --label_smoothing 0.2
    --early_stopping_patience 3
    --early_stopping_delta 0.001
    --device cuda
    --num_folds "$NUM_FOLDS"
    --start_fold "$START_FOLD"
)

# Loop through each phase
for PHASE in "${PHASES[@]}"; do
    echo "=========================================="
    echo "Running K-Fold Cross-Validation"
    echo "Condition: $CONDITION"
    echo "Phase: $PHASE"
    echo "Folds: $NUM_FOLDS (starting from fold $START_FOLD)"
    echo "=========================================="
    
    python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
        "${BASE_ARGS[@]}" \
        --phase "$PHASE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed K-Fold for phase: $PHASE"
    else
        echo "✗ Failed to complete K-Fold for phase: $PHASE"
        # Uncomment the next line if you want to stop on error:
        # exit 1
    fi
    
    echo ""
    echo "Waiting 10 seconds before next phase..."
    sleep 10
    echo ""
done

echo "=========================================="
echo "All phases completed!"
echo "=========================================="


# ============================================================================
# AUDIO FINETUNING WITH CONDITION AND PHASE FILTERING
# ============================================================================
# Uncomment the section below to run audio finetuning with specific conditions/phases

# Audio Configuration
# AUDIO_CONDITION="CRADK"  # Change this to your desired condition (CRADK, ADK, SHAM, CR, all)
# AUDIO_PHASES=("training_pos" "training_neg" "induction1" "induction2")
# AUDIO_NUM_FOLDS=5
# AUDIO_START_FOLD=0

# Base audio command arguments
# AUDIO_BASE_ARGS=(
#     --model MultiModalDepDet
#     --dataset d02_npy_downsampled
#     --condition "$AUDIO_CONDITION"
#     --modalities audio
#     --train true
#     --fusion audio
#     --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data
#     --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/audioset_10_10_0.4593.pth
#     -lr 0.00001
#     -bs 16
#     -e 15
#     --optimizer AdamW
#     --weight_decay 0.001
#     --lr_scheduler cos
#     --audio_dropout 0.3
#     --classifier_dropout 0.0
#     --attention_dropout 0.2
#     --transformer_dropout 0.2
#     --lambda_reg 0.0001
#     --focal_weight 0.5
#     --l2_weight 0.7
#     --label_smoothing 0.2
#     --early_stopping_patience 3
#     --early_stopping_delta 0.001
#     --device cuda
#     --num_folds "$AUDIO_NUM_FOLDS"
#     --start_fold "$AUDIO_START_FOLD"
# )

# Loop through each phase for audio
# for PHASE in "${AUDIO_PHASES[@]}"; do
#     echo "=========================================="
#     echo "Running Audio K-Fold Cross-Validation"
#     echo "Condition: $AUDIO_CONDITION"
#     echo "Phase: $PHASE"
#     echo "Folds: $AUDIO_NUM_FOLDS (starting from fold $AUDIO_START_FOLD)"
#     echo "=========================================="
#     
#     python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
#         "${AUDIO_BASE_ARGS[@]}" \
#         --phase "$PHASE"
#     
#     # Check if the command was successful
#     if [ $? -eq 0 ]; then
#         echo "✓ Successfully completed Audio K-Fold for phase: $PHASE"
#     else
#         echo "✗ Failed to complete Audio K-Fold for phase: $PHASE"
#         # Uncomment the next line if you want to stop on error:
#         # exit 1
#     fi
#     
#     echo ""
#     echo "Waiting 10 seconds before next phase..."
#     sleep 10
#     echo ""
# done

# echo "=========================================="
# echo "All audio phases completed!"
# echo "=========================================="

# Previous week
for condition in CRADK; do
    for phase in training_pos training_neg; do
        python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
            --model MultiModalDepDet \
            --dataset d02_video_npy_downsampled  --condition "$condition" --phase "$phase" \
            --modalities video --train true --fusion video \
            --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
            --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth \
            -lr 0.00001 -bs 16 -e 20 --optimizer AdamW --weight_decay 0.001 --lr_scheduler cos \
            --visual_dropout 0.3 --classifier_dropout 0.0 --attention_dropout 0.2 --transformer_dropout 0.2 \
            --lambda_reg 0.0001 --focal_weight 0.5 --l2_weight 0.7 --label_smoothing 0.2 \
            --early_stopping_patience 3 --early_stopping_delta 0.001 --device cuda \
            --num_folds 5 --start_fold 0
    done
done

# chmod +x run_kfold_multiple_phases.sh
# ./run_kfold_multiple_phases.sh


# training_pos training_neg induction1 induction2
# Or run the following command:
# ADK CR All SHAM CRADK
for condition in SHAM; do
    for phase in induction2; do
        python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
            --model MultiModalDepDet \
            --dataset d02_video_npy_downsampled_30_avg --condition "$condition" --phase "$phase" \
            --modalities video --train true --fusion video \
            --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
            --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth \
            -lr 0.0005 -bs 16 -e 100 --optimizer AdamW --weight_decay 0.1 --lr_scheduler cos \
            --visual_dropout 0.3 --classifier_dropout 0.2 --attention_dropout 0.2 --transformer_dropout 0.2 \
            --lambda_reg 0.001 --focal_weight 0.25 --l2_weight 0.7 --label_smoothing 0.05 \
            --early_stopping_patience 15 --early_stopping_delta 0.001 --device cuda \
            --num_folds 5 --start_fold 0
    done
done






# Audio finetuning examples with condition and phase filtering
# Example 1: Single condition and phase
# python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
#     --model MultiModalDepDet \
#     --dataset d02_npy_downsampled \
#     --condition CRADK \
#     --phase training_pos \
#     --modalities audio \
#     --train true \
#     --fusion audio \
#     --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
#     --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/audioset_10_10_0.4593.pth \
#     -lr 0.00001 \
#     -bs 16 \
#     -e 15 \
#     --optimizer AdamW \
#     --weight_decay 0.001 \
#     --lr_scheduler cos \
#     --audio_dropout 0.3 \
#     --classifier_dropout 0.0 \
#     --attention_dropout 0.2 \
#     --transformer_dropout 0.2 \
#     --lambda_reg 0.0001 \
#     --focal_weight 0.5 \
#     --l2_weight 0.7 \
#     --label_smoothing 0.2 \
#     --early_stopping_patience 3 \
#     --early_stopping_delta 0.001 \
#     --device cuda \
#     --num_folds 5 \
#     --start_fold 0

# CRADK, ADK, SHAM, All, CR
# training_pos training_neg induction1 induction2
#Example 2: Loop through multiple conditions and phases (similar to video)
for condition in All; do
    for phase in training_pos training_neg induction1 induction2; do
        python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
            --model MultiModalDepDet \
            --dataset d02_npy_downsampled \
            --condition "$condition" \
            --phase "$phase" \
            --modalities audio \
            --train true \
            --fusion audio \
            --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
            --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/audioset_10_10_0.4593.pth \
            -lr 0.00001 \
            -bs 16 \
            -e 15 \
            --optimizer AdamW \
            --weight_decay 0.001 \
            --lr_scheduler cos \
            --audio_dropout 0.3 \
            --classifier_dropout 0.0 \
            --attention_dropout 0.2 \
            --transformer_dropout 0.2 \
            --lambda_reg 0.0001 \
            --focal_weight 0.5 \
            --l2_weight 0.7 \
            --label_smoothing 0.2 \
            --early_stopping_patience 3 \
            --early_stopping_delta 0.001 \
            --device cuda \
            --num_folds 5 \
            --start_fold 0
    done
done

# Example 3: All conditions and phases (use "all" or omit the flags)
# python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/mainkfold.py \
#     --model MultiModalDepDet \
#     --dataset d02_npy_downsampled \
#     --condition all \
#     --phase all \
#     --modalities audio \
#     --train true \
#     --fusion audio \
#     --condition CRADK \
#     --phase training_pos \
#     --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
#     --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/audioset_10_10_0.4593.pth \
#     -lr 0.00001 \
#     -bs 16 \
#     -e 15 \
#     --optimizer AdamW \
#     --weight_decay 0.001 \
#     --lr_scheduler cos \
#     --audio_dropout 0.3 \
#     --classifier_dropout 0.0 \
#     --attention_dropout 0.2 \
#     --transformer_dropout 0.2 \
#     --lambda_reg 0.0001 \
#     --focal_weight 0.5 \
#     --l2_weight 0.7 \
#     --label_smoothing 0.2 \
#     --early_stopping_patience 3 \
#     --early_stopping_delta 0.001 \
#     --device cuda \
#     --num_folds 5 \
#     --start_fold 0