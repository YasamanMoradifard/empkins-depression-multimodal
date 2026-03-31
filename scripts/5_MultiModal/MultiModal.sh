#!/bin/bash -l
#SBATCH --partition=work               # Woody's default partition
#SBATCH --constraint=icx               # Ice Lake nodes (32 cores, 256GB RAM)
#SBATCH --nodes=1                      # Single node (required on Woody)
#SBATCH --ntasks=1                     # Single task for Python
#SBATCH --cpus-per-task=8             # 16 cores (adjust based on your needs)
#SBATCH --mem-per-cpu=16G               # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00                # Max walltime on Woody
#SBATCH --job-name=multimodal_ml
#SBATCH --mail-user=yasaman.moradi.fard@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE                  # Clean environment (recommended for Woody)

# ----------------------------
# Configuration
# ----------------------------
WORKDIR="/home/vault/empkins/tpD/D02/Students/Yasaman/5_MultiModal_ML"
VENV_DIR="/home/hpc/iwso/iwso170h/MT/venv"
SCRIPT="MultiModal_EarlyFusion.py"

# Change to working directory
cd "${WORKDIR}"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
#-----------------------------
# Current run configuration
# ----------------------------
Aggregation_method="by_phase"          # Options: by_ID or by_phase
MODALITIES="ecg rsp emg video audio"               # Space-separated list: ecg rsp emg video audio
CONDITION="ADK"                      # Options: CRADK, ADK, CR, SHAM, all
PHASES="training_pos"                 # Space-separated list or single phase
MINUTES=3                              # Options: 1, 3, 5
MANNWHITNEY_PER_MODALITY=20            # Early fusion only: MW selects this many features per modality, then concat; RFE on result

CLASSIFIERS=("LogisticRegression" "SVC_RBF" "RandomForest" "AdaBoost" "DecisionTree" "KNN" "XGBoost")


# ----------------------------
# Command
# ----------------------------
echo "Running ${SCRIPT} with the following parameters:
=======================
aggregation method: ${Aggregation_method}
=======================
modalities: ${MODALITIES}
=======================
condition: ${CONDITION}
=======================
phases: ${PHASES}
=======================
minutes: ${MINUTES}
=======================
mannwhitney_per_modality: ${MANNWHITNEY_PER_MODALITY}
=======================



CONDITION="all"
CLASSIFIERS=("LogisticRegression" "SVC_RBF" "RandomForest" "AdaBoost" "DecisionTree" "KNN" "XGBoost")
MODALITIES="ecg rsp emg video audio text"
MINUTES=3
MANNWHITNEY_PER_MODALITY=20 
Aggregation_method="by_ID"

for classifier in "${CLASSIFIERS[@]}"; do
        python MultiModal_EarlyFusion.py \
        --aggregation_method "${Aggregation_method}" \
        --fusion_modalities ${MODALITIES} \
        --condition "${CONDITION}" \
        --minutes ${MINUTES} \
        --mannwhitney_per_modality ${MANNWHITNEY_PER_MODALITY} \
        --classifier "${classifier}"
done




############################# Run Classification #############################

## Early: 
  # By Phase:
    # No Text: 20 runs, 4 phases, 5 conditions 1 ############### DONE
    # With Text: 4 runs, 4 phases, 1 condition (All) 4 ############### DONE

  # By ID:
    # No Text: 5 runs, No phases, 5 conditions 3 ############### DONE
    # With Text: 1 runs, No phases, 1 condition (All) 4 ############### DONE

## Late: 
  # By Phase:
    # No Text: 20 runs, 4 phases, 5 conditions 2
    # With Text: 4 runs, 4 phases, 1 condition (All) 4 ############### DONE

  # By ID:
    # No Text: 5 runs, No phases, 5 conditions 3 ############### DONE
    # With Text: 1 runs, No phases, 1 condition (All) 4 ############### DONE


SCRIPT=(MultiModal_Late_Classification.py)
MODALITIES="ecg rsp emg video audio"
CONDITION=("CRADK" "CR" "all")
PHASES=("induction1")
CLASSIFIERS=("KNN" "XGBoost" "RandomForest" "AdaBoost" "DecisionTree" "LogisticRegression" "SVC_RBF")
Aggregation_method=("by_phase")

for script in "${SCRIPT[@]}"; do
    for aggregation_method in "${Aggregation_method[@]}"; do
        for condition in "${CONDITION[@]}"; do
            for phase in "${PHASES[@]}"; do
                for classifier in "${CLASSIFIERS[@]}"; do
                    python ${script} \
                        --aggregation_method "${aggregation_method}" \
                        --fusion_modalities ${MODALITIES} \
                        --condition "${condition}" \
                        --phases "${phase}" \
                        --classifier "${classifier}"
                done
            done
        done
    done
done







############################# Run Regression #############################

## Early: 
  # By Phase:
    # No Text: 20 runs, 4 phases, 5 conditions ######################## DONE
    # With Text: 4 runs, 4 phases, 1 condition (All) ######################## DONE

  # By ID:
    # No Text: 5 runs, No phases, 5 conditions
    # With Text: 4 runs, 4 phases, 1 condition (All) ######################## DONE

## Late: 
  # By Phase:
    # No Text: 20 runs, 4 phases, 5 conditions ######################## DONE
    # With Text: 4 runs, 4 phases, 1 condition (All) ######################## DONE

  # By ID:
    # No Text: 5 runs, No phases, 5 conditions ######################## DONE
    # With Text: 4 runs, 4 phases, 1 condition (All) ######################## DONE



SCRIPT=(MultiModal_Late_Regression.py MultiModal_Early_Regression.py)
MODALITIES="ecg rsp emg video audio"
CONDITION=("ADK" "CRADK" "CR" "SHAM" "all")
PHASES=("induction2")
REGRESSORS=("RandomForest" "AdaBoost" "DecisionTree" "KNN" "LinearRegression")
Aggregation_method=("by_ID")

for script in "${SCRIPT[@]}"; do
    for aggregation_method in "${Aggregation_method[@]}"; do
        for condition in "${CONDITION[@]}"; do
            for phase in "${PHASES[@]}"; do
                for regressor in "${REGRESSORS[@]}"; do
                    python ${script} \
                        --aggregation_method "${aggregation_method}" \
                        --fusion_modalities ${MODALITIES} \
                        --condition "${condition}" \
                        --phases "${phase}" \
                        --regressors "${regressor}"
                done
            done
        done
    done
done