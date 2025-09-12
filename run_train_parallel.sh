#!/bin/bash

source .venv/bin/activate
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)

python -u -m wbia_miew_id.train_parallel \
    --config wbia_miew_id/configs/config_zebra.yaml \
    --nodes $SLURM_JOB_NUM_NODES \
    --gpus-per-node $SLURM_NTASKS_PER_NODE