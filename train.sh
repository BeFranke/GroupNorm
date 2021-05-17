#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benedikt.franke@uni-ulm.de
#SBATCH --error=GroupNorm_errors.txt
#SBATCH --partition=gpu_4,gpu_8
source ~/.bashrc
conda init
conda activate tf
python train.py --seeds 1 2 3 4 5

