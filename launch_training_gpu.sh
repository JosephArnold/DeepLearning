#!/bin/bash -x
#SBATCH --output=/p/project1/sdlrs/arnold6/fwdgrad/slurm_outputs/fwdgrad-out_gpu.%j
#SBATCH --error=/p/project1/sdlrs/arnold6/fwdgrad/slurm_outputs/fwdgrad-err_gpu.%j
#SBATCH --account=####
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:120:00
#SBATCH --partition=####


ml  Stages/2024  GCCcore/.12.3.0
ml Hydra/1.3.2
ml PyTorch/2.1.2
ml torchvision
ml tensorboard
ml matplotlib/3.7.2

srun --nodes=1 --ntasks=1 --gpus-per-task=1 python fwdgrad-nn.py
#srun --nodes=1 --ntasks-per-node=$OMP_NUM_THREADS /p/project1/sdlrs/arnold6/JUAN-Optimized/hpdbscan/build/hpdbscan -t $OMP_NUM_THREADS -i bremen.h5  -m 215 -e 25 --input-dataset DBSCAN
#start_time=$(date +%s)
#/p/project1/sdlrs/arnold6/dbscan-python/build-AVX-512/executable/dbscan eps 25 -minpts 215 -o clusters.txt bremen.csv 
#end_time=$(date +%s)
#elapsed=$((end_time - start_time))
