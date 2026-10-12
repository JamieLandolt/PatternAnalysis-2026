#!/bin/bash
# Train one model on a Rangpur GPU, e.g.:
#   sbatch --job-name=lob_timegan slurm_train.sh timegan
#   sbatch --job-name=lob_baseline slurm_train.sh baseline
# Override defaults with environment variables, e.g. EPOCHS=200 sbatch ...
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL=${1:?usage: sbatch slurm_train.sh <timegan|baseline>}
DATA_DIR=${DATA_DIR:-$HOME/COMP3710/LOB_project/data}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
source $HOME/miniconda3/bin/activate torch

nvidia-smi -L
python3 -u train.py --model "$MODEL" --data-dir "$DATA_DIR" --out-dir outputs \
    --ae-epochs "${AE_EPOCHS:-30}" --sup-epochs "${SUP_EPOCHS:-30}" \
    --epochs "${EPOCHS:-100}" --eval-every 5
