#!/bin/bash
#SBATCH --job-name=miew_id_train
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=150G
#SBATCH --account=PAS2136
#SBATCH --output=output/train_%j.out
#SBATCH --error=output/train_%j.err
#SBATCH --gpus=1


source .venv/bin/activate 
python -m wbia_miew_id.train --config wbia_miew_id/configs/config_zebra.yaml

# Run script for a single recorder
      # --output_dir /fs/ess/PAS2136/Hawaii-2025/bird-sounds/embeddings/koa/probabilities       # --output_separation_dir /fs/ess/PAS2136/Hawaii-2025/bird-sounds/embeddings/koa/source_separation       # --threshold 0.5       # --recorder_id submit_historic       # --species_dir None       # --noise_dir None


