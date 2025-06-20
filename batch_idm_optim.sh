#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --job-name=Counts
#SBATCH --output=/data/horse/ws/thdi929f-Waymo/Models/Log/output//%j.out
#SBATCH --error=/data/horse/ws/thdi929f-Waymo/Models/Log/errors//%j.out

srun python "horse/thdi929f-Waymo/Models/Model-Att-UNet-LSTM V1.py" 

