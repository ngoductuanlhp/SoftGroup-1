#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_only_semantic.yaml  

CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_only_semantic_points20.yaml 

CUDA_VISIBLE_DEVICES=4 python3 tools/train.py configs/softgroup_scannet_only_semantic_points_otoc.yaml

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_only_se
mantic_points_otoc.yaml

CUDA_VISIBLE_DEVICES=5 python3 tools/train.py configs/softgroup_scannet_only_semantic_points20_pointmatch.yaml


CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_only_semantic_points20_pointmatch.yaml 
