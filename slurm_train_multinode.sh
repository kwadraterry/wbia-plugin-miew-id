#!/bin/bash
#SBATCH --job-name=miew-id-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load necessary modules (adjust based on your cluster)
source .venv/bin/activate 
# module load python/3.9
# module load cuda/11.8
# module load nccl

# Set up environment
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

# Create logs directory if it doesn't exist
mkdir -p logs

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Get the master node hostname
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
export MASTER_ADDR

echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Run the training script
srun python -u wbia_miew_id/train_parallel.py \
    --config configs/config_zebra.yaml \
    --nodes $SLURM_JOB_NUM_NODES \
    --gpus-per-node $SLURM_NTASKS_PER_NODE