#!/bin/bash -e
#SBATCH --job-name=soft_BQ
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G

#SBATCH --cpus-per-gpu=64

#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io

srun --container-image="harbor.vinai-systems.com#research/tuannd42:softgroup" \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/SoftGroup:/home/ubuntu/SoftGroup \
--container-workdir=/home/ubuntu/SoftGroup/ \
python3 tools/train.py configs/softgroup_scannet_bbox_context_head_ballquery.yaml --exp_name thresh0.95