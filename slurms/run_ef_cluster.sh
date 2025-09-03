#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --job-name=ef_bs8_ga2_224
#SBATCH --output=/users/spet4299/code/TEE/flow-matching/go-with-the-flow/cluster_outputs/ef_regression/%j.out
#SBATCH --gres=gpu:2

# Credit to Jong Kwon & John McGonigle
abort() { >&2 printf '█%.0s' {1..40}; (>&2 printf "\n[ERROR] $(basename $0) has exited early\n"); exit 1; }  # print error message
scriptdirpath=$(cd -P -- "$(dirname -- "$0")" && pwd -P);
IFS=$'\n\t'; set -eo pipefail; # exits if error, and set IFS, so no whitespace error

trap 'abort' 0; set -u;
# Sets abort trap defined in line 2, set -u exits when detects unset variables

# cd into the scriptdirpath so that relative paths work
pushd "${scriptdirpath}" > /dev/null

# _________ ACTUAL CODE THAT RUNS STUFF __________

echo -e "ml cuda: \n"
ml cuda

CONDA_ENV="flow_match"

# Activate conda env if in base env, or don't if already set.
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi


echo "Python: $(which python)"
# Count the number of GPUs assigned to this job
echo "GPU ID: $CUDA_VISIBLE_DEVICES"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Using $NUM_GPUS GPUs"

USER="spet4299"
PROJECT_PATH="/users/spet4299/code/TEE/flow-matching/go-with-the-flow"
DATA_SOURCE="/data/spet4299/TEE_Data/ML/datasets/echo_dataset"
OUTPUT_DIR=${PROJECT_PATH}/outputs/ef_regression

# Run the actual script
if [[ $NUM_GPUS -gt 1 ]]; then
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    python ${PROJECT_PATH}/ef_regression/train_reference.py \
      --config ${PROJECT_PATH}/ef_regression/config_reference/camus_112_32.yaml
    #accelerate launch --multi_gpu --debug --num_processes=$NUM_GPUS ${PROJECT_PATH}/train_base_model.py \
      #--config-name $CONFIG_NAME
else
    echo "Running single-GPU training..."

    python ${PROJECT_PATH}/ef_regression/train_reference.py \
      --config ${PROJECT_PATH}/ef_regression/config_reference/camus_112_32.yaml

 #\
      #--config-name $CONFIG_NAME
fi
conda deactivate

# ___________ MORE SAFE CRASH JARGON ____________

popd > /dev/null

trap : 0
(>&2 echo "✔")
exit 0