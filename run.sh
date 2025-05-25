#!/bin/bash


#SBATCH --account=digital_human_jobs
#SBATCH --output=./logs/bonn_balloon_gsAccel_sparseAdam16_fusedssim.out

python run.py  ./configs/Dynamic/Bonn/bonn_balloon.yaml
