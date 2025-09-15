#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --job-name=miew-id-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up environment variables for distributed training
# Get the head node address (first node in the allocation)
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')

# Set up the rendezvous endpoint
export RDZV_PORT=$((29500 + ($SLURM_JOBID % 1000)))
export RDZV_ENDPOINT="${HEAD_NODE_IP}:${RDZV_PORT}"

# Number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=4

# Unique job ID for rendezvous
export RDZV_ID=$SLURM_JOBID

# Debug output
echo "==== SLURM Configuration ===="
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_NODE_IP: $HEAD_NODE_IP"
echo "RDZV_ENDPOINT: $RDZV_ENDPOINT"
echo "RDZV_ID: $RDZV_ID"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "============================="

# NCCL environment variables for better performance and debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# PyTorch environment variables
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Path to training script
TRAINING_SCRIPT="wbia_miew_id/train_parallel.py"
CONFIG_FILE="wbia_miew_id/configs/config_zebra.yaml"

# Launch torchrun on all nodes
# Using srun to launch torchrun on each node
srun --ntasks-per-node=1 \
    torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --max_restarts=3 \
    --monitor_interval=30 \
    $TRAINING_SCRIPT \
    --config $CONFIG_FILE