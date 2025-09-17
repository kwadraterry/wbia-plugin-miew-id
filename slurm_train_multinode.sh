#!/bin/bash -l
#SBATCH --job-name=miew-id-train
#SBATCH --account=PAS2136
#SBATCH -p nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=34              # 17 CPUs/GPU × 2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=wbia-plugin-miew-id/logs/train_%j.out
#SBATCH --error=wbia-plugin-miew-id/logs/train_%j.err

# Load virtual environment
source .venv/bin/activate 

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# single-node-friendly: avoid IB/rdma weirdness, and shm issues on some clusters
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

CONFIG="wbia_miew_id/configs/config_zebra.yaml"
test -f "$CONFIG" || { echo "Config not found: $CONFIG"; exit 1; }

# Master info (single node)
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=${MASTER_PORT:-29500}

# See both GPUs
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gpus-per-task=2 nvidia-smi -L

# torchrun spawns 2 ranks; your code now prefers LOCAL_RANK/RANK/WORLD_SIZE
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gpus-per-task=2 \
  torchrun --nproc_per_node=2 --nnodes=1 \
           --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  -m wbia_miew_id.train_parallel \
    --config "$CONFIG" \
    --nodes 1 \
    --gpus-per-node 2


# srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gpus-per-task=2 \
#   torchrun \
#     --nproc_per_node=$SLURM_GPUS_ON_NODE \
#     --nnodes=$SLURM_NNODES \
#     --master_addr "$MASTER_ADDR" \
#     --master_port ${MASTER_PORT:-29500} \
#     -m wbia_miew_id.train_parallel \
#     --config wbia_miew_id/configs/config_zebra.yaml