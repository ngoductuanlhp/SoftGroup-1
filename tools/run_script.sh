#!/bin/bash -e
#SBATCH --job-name=detr_c
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G

#SBATCH --cpus-per-gpu=64

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io


srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/softgroup.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/SoftGroup:/home/ubuntu/SoftGroup \
--container-workdir=/home/ubuntu/SoftGroup/ \
python3 tools/train.py configs/softgroup_scannet_bbox_context_detr.yaml --exp_name detr_dino_box_centroid