#!/bin/bash
# conditions: CRADK, CR, ADK, SHAM, All
# Define the phases you want to run
PHASES=("training_pos" "training_neg" "induction1" "induction2")

# Loop through each phase
for PHASE in "${PHASES[@]}"; do
    echo "=========================================="
    echo "Running phase: $PHASE"
    echo "=========================================="
    
    python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/main.py \
        --dataset d02_npy_downsampled \
        --condition ADK \
        --phase "$PHASE" \
        --modalities video \
        --train true \
        --fusion video \
        --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
        --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth \
        -lr 0.0005 \
        -bs 16 \
        -e 50 \
        --optimizer AdamW \
        --weight_decay 0.001 \
        --lr_scheduler cos \
        --visual_dropout 0.3 \
        --classifier_dropout 0.0 \
        --attention_dropout 0.2 \
        --transformer_dropout 0.2 \
        --lambda_reg 0.0001 \
        --focal_weight 0.5 \
        --l2_weight 0.7 \
        --label_smoothing 0.2 \
        --early_stopping_patience 3 \
        --early_stopping_delta 0.001 \
        --device cuda
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed phase: $PHASE"
    else
        echo "✗ Failed to complete phase: $PHASE"
        # Uncomment the next line if you want to stop on error:
        # exit 1
    fi
    
    echo ""
    echo "Waiting 5 seconds before next phase..."
    sleep 5
    echo ""
done

echo "=========================================="
echo "All phases completed!"
echo "=========================================="



# # """ Run the script:
# chmod +x scripts/run_multiple_phases.sh
# ./scripts/run_multiple_phases.sh
# # """



# Or use the following command:
# for phase in training_pos training_neg induction1 induction2; do
#     python /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/scripts/main.py \
#         --dataset d02_npy_downsampled --condition All --phase "$phase" \
#         --modalities video --train true --fusion video \
#         --data_dir /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/data \
#         --resume_path /home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/pretrained_models/visualmae_pretrained.pth \
#         -lr 0.0005 -bs 16 -e 50 --optimizer AdamW --weight_decay 0.001 --lr_scheduler cos \
#         --visual_dropout 0.3 --classifier_dropout 0.0 --attention_dropout 0.2 --transformer_dropout 0.2 \
#         --lambda_reg 0.0001 --focal_weight 0.5 --l2_weight 0.7 --label_smoothing 0.2 \
#         --early_stopping_patience 3 --early_stopping_delta 0.001 --device cuda
# done
