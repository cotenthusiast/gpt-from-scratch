#!/bin/bash
# Kelvin2 (Queen's University Belfast HPC) A100 training job for gpt-from-scratch.
#
# Assumes the repo is cloned to:
#   /mnt/scratch2/users/$USER/repos/gpt-from-scratch
# Assumes the project venv exists at:
#   /mnt/scratch2/users/$USER/venvs/gpt-from-scratch
#
# Usage (from the repo root on the Kelvin2 login node):
#   sbatch scripts/train_kelvin2.slurm
#
# Verify partition/GRES names before submitting, they can change:
#   sinfo -o "%P %D %G %m %l %N"

#SBATCH --job-name=gpt_shakespeare
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1

set -euo pipefail

# BASH_SOURCE/dirname resolves incorrectly in Kelvin2's SLURM execution
# environment (sbatch copies the script to a spool dir before running it).
REPO_ROOT="/mnt/scratch2/users/$USER/repos/gpt-from-scratch"
cd "$REPO_ROOT"

mkdir -p logs checkpoints outputs

VENV_DIR="/mnt/scratch2/users/$USER/venvs/gpt-from-scratch"

# `module` is not available in this cluster's batch shell; the venv above was
# created directly against /opt/apps/python3/3.10.5/gcc-9.3.0/bin/python3, so
# activating it is sufficient on its own.
source "$VENV_DIR/bin/activate"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"
echo "Venv:    $VENV_DIR"

# Fail loudly instead of silently training on CPU.
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available on this node'
print('CUDA OK:', torch.cuda.get_device_name(0))
"

python -u train.py --preset a100 --checkpoint checkpoints/shakespeare_gpt.pt

python -u generate.py \
    --checkpoint checkpoints/shakespeare_gpt.pt \
    --tokens 10000 \
    --output outputs/generated.txt

echo "Job complete."
