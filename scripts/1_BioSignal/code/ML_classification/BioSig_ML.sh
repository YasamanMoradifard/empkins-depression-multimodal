#!/bin/bash -l
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name=BioSig_ML
#SBATCH --mail-user=yasaman.moradi.fard@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL


WORKDIR="/home/vault/empkins/tpD/D02/Students/Yasaman/5_MultiModal_ML"
VENV_DIR="/home/hpc/iwso/iwso170h/MT/venv"
source "${VENV_DIR}/bin/activate"

# Change to the directory containing the script
cd "${WORKDIR}/Single_modality" || exit 1

CONDITIONS=("CRADK" "CR" "all" "SHAM" "ADK")
PHASES=("training_pos" "training_neg" "latency" "induction2" "induction1")
DataType=("RSP" "ECG" "EMG")
for data_type in "${DataType[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        for PHASE in "${PHASES[@]}"; do
            python BioSig_ML.py \
            --data_type "$data_type" \
            --minutes 3 \
            --condition "$CONDITION" \
            --phases "$PHASE" \
            --aggregation_method by_phase \
            --mannwhitney_prefilter 50
        done
    done
done


CONDITIONS=("CRADK" "CR" "all" "SHAM" "ADK")
PHASES=("training_pos" "training_neg" "induction2" "induction1")

for CONDITION in "${CONDITIONS[@]}"; do
    for PHASE in "${PHASES[@]}"; do
        python Audio_ML.py \
        --condition "$CONDITION" \
        --phases "$PHASE" \
        --aggregation_method by_phase \
        --mannwhitney_prefilter 50
    done
done


CONDITIONS=("CRADK" "CR" "all" "SHAM" "ADK")
PHASES=("training_pos" "training_neg" "latency" "induction2" "induction1")

for CONDITION in "${CONDITIONS[@]}"; do
    for PHASE in "${PHASES[@]}"; do
        python Video_ML.py \
        --condition "$CONDITION" \
        --phases "$PHASE" \
        --aggregation_method by_phase \
        --mannwhitney_prefilter 50
    done
done