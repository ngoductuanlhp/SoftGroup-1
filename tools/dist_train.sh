#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context.yaml 

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context_detr_simple.yaml --exp_name separate_matching_expcoef

CUDA_VISIBLE_DEVICE=0 python3 tools/train.py configs/softgroup_scannet_bbox_context_head.yaml --resume work_dirs/softgroup_scannet_bbox_context_head/epoch_1.pth --exp_name debug

CUDA_VISIBLE_DEVICES=4,7 OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/softgroup_scannet_bbox_context_detr.yaml --exp_name detr
CUDA_VISIBLE_DEVICES=4 python3 tools/train.py configs/softgroup_scannet_bbox_context_detr.yaml --exp_name no_transformer