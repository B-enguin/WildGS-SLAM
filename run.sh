#!/bin/bash


#SBATCH --account=3dv
#SBATCH --output=./logs/bonn_balloon_earlystop.out

python run.py  ./configs/Dynamic/Bonn/bonn_balloon.yaml
