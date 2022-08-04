#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context.yaml 

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context.yaml --exp_name trainval

CUDA_VISIBLE_DEVICE=0 python3 tools/train.py configs/softgroup_scannet_bbox_context_head.yaml --resume work_dirs/softgroup_scannet_bbox_context_head/epoch_1.pth --exp_name debug

CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context.yaml --exp_name softgroup_scannet_bbox_context_extreme_points
CUDA_VISIBLE_DEVICES=7 python3 tools/train.py configs/softgroup_scannet_bbox_context.yaml --exp_name softgroup_scannet_bbox_context_extreme_points