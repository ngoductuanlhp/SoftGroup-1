#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}

CUDA_VISIBLE_DEVICES=0,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet.yaml 

CUDA_VISIBLE_DEVICES=1 python3 tools/train.py configs/softgroup_scannet.yaml

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_dycohead.yaml 
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_dycohead.yaml 

CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_dycohead.yaml 

CUDA_VISIBLE_DEVICES=5,6 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_dycohead_embed_coord.yaml 