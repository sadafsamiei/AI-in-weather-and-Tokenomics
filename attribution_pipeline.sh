#!/bin/bash -l
#SBATCH --job-name=e2s
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=GPUMEM80GB
#SBATCH --output=gpu-%j.log

# restrict to H100 nodes
module load h100

# load GPU libraries matched to H100
module load cuda cudnn nccl

# confirm what GPU/node was allocated
echo "=== Hostname and allocated GPUs ==="
hostname
nvidia-smi

# run
uv run attribution_pipeline/main.py