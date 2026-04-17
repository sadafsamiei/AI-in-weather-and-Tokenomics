#!/bin/bash -l
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=gpu-%j.log

uv run analysis/main.py