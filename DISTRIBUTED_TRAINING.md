# Distributed Training on SLURM

This document describes how to run distributed training for MiewID on SLURM clusters with multiple GPUs and nodes.

## Files Modified/Created

- `wbia_miew_id/train_parallel.py` - Updated to support multi-node distributed training
- `slurm_train_multinode.sh` - SLURM job script for multi-node training
- `slurm_train_single_node.sh` - SLURM job script for single-node multi-GPU training

## Key Features

### Distributed Training Support
- **Multi-node training**: Supports training across multiple compute nodes
- **Multi-GPU training**: Utilizes all available GPUs per node
- **SLURM integration**: Automatically detects SLURM environment variables
- **Fault tolerance**: Proper initialization and cleanup of distributed processes

### Performance Optimizations
- **Batch size scaling**: Automatically scales batch size per GPU
- **Distributed sampling**: Uses DistributedSampler for efficient data loading
- **Synchronized evaluation**: Only rank 0 performs evaluation and saves models
- **Wandb logging**: Only rank 0 logs to wandb to avoid conflicts

## Usage

### 1. Single Node Multi-GPU Training

For training on a single node with multiple GPUs:

```bash
sbatch slurm_train_single_node.sh
```

This script is configured for:
- 1 node
- 4 GPUs per node
- 8 CPU cores per task
- 64GB RAM per node

### 2. Multi-Node Training

For training across multiple nodes:

```bash
sbatch slurm_train_multinode.sh
```

This script is configured for:
- 2 nodes
- 4 GPUs per node (8 total GPUs)
- 8 CPU cores per task
- 64GB RAM per node

### 3. Custom Configuration

You can modify the SLURM scripts to adjust:
- Number of nodes (`--nodes`)
- GPUs per node (`--gres=gpu:N`)
- CPU cores per task (`--cpus-per-task`)
- Memory per node (`--mem`)
- Training time limit (`--time`)
- Partition (`--partition`)

## Environment Variables

The training script automatically handles SLURM environment variables:
- `SLURM_PROCID`: Global rank of the process
- `SLURM_NTASKS`: Total number of processes
- `SLURM_LOCALID`: Local rank within the node
- `SLURM_NODELIST`: List of allocated nodes

## Configuration

Make sure your training configuration file (`configs/default_config.yaml`) is properly set up with:
- Appropriate batch sizes (will be divided by world size)
- Model parameters
- Dataset paths
- Training parameters

## Monitoring

Training logs are saved to:
- `logs/train_<job_id>.out` - Standard output
- `logs/train_<job_id>.err` - Error output

You can monitor training progress with:
```bash
tail -f logs/train_<job_id>.out
```

## Troubleshooting

### Common Issues

1. **NCCL Initialization Errors**: Ensure NCCL is properly loaded and network connectivity exists between nodes
2. **CUDA Out of Memory**: Reduce batch size in configuration
3. **Port Conflicts**: Change MASTER_PORT in SLURM scripts if 29500 is occupied
4. **Module Loading**: Adjust module load commands based on your cluster setup

### Debug Mode

Enable detailed NCCL debugging by setting:
```bash
export NCCL_DEBUG=INFO
```

### Network Configuration

For multi-node training, ensure:
- Nodes can communicate over InfiniBand or high-speed network
- Firewall allows communication on the master port
- CUDA-aware MPI is available if needed

## Performance Tips

1. **Batch Size**: Use largest batch size that fits in GPU memory
2. **Data Loading**: Increase `num_workers` for faster data loading
3. **Network**: Use InfiniBand for best multi-node performance
4. **Storage**: Use fast shared storage (e.g., Lustre, parallel file system)

## Example Commands

### Interactive Testing
```bash
# Request interactive session
salloc --nodes=2 --ntasks-per-node=4 --gres=gpu:4 --time=1:00:00

# Run training
srun python -u wbia_miew_id/train_parallel.py --config configs/default_config.yaml
```

### Custom Configuration
```bash
# Run with custom config
srun python -u wbia_miew_id/train_parallel.py --config configs/my_config.yaml --nodes 4 --gpus-per-node 8
```