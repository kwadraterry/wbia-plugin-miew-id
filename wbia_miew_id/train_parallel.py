from wbia_miew_id.logging_utils import WandbContext
from wbia_miew_id.etl import preprocess_data, print_intersect_stats, load_preprocessed_mapping, preprocess_dataset
from wbia_miew_id.losses import fetch_loss
from wbia_miew_id.schedulers import MiewIdScheduler
from wbia_miew_id.helpers import get_config, write_config, update_bn
from wbia_miew_id.metrics import AverageMeter, compute_calibration
from wbia_miew_id.datasets import MiewIdDataset, get_train_transforms, get_valid_transforms
from wbia_miew_id.models import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic, MiewIdNet
from wbia_miew_id.engine import train_fn, eval_fn, group_eval_fn


from torch.optim.swa_utils import AveragedModel, SWALR
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
import logging
import sys


class Trainer:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for distributed training"""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [Rank %(rank)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_info(self, message):
        """Log with rank information"""
        if hasattr(self, 'rank'):
            self.logger.info(message, extra={'rank': self.rank})
        else:
            self.logger.info(message, extra={'rank': 'N/A'})

    def set_seed_torch(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run_fn(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
        best_score = 0
        best_cmc = None
        
        for epoch in range(self.config.engine.epochs):
            # Set epoch for distributed sampler to ensure proper shuffling
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_loss = train_fn(
                train_loader, model, criterion, optimizer, device, 
                scheduler=scheduler, epoch=epoch, 
                use_wandb=use_wandb and self.rank == 0,
                swa_model=swa_model, swa_start=swa_start, 
                swa_scheduler=swa_scheduler
            )

            if self.rank == 0:
                self.log_info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                print("\nGetting metrics on validation set...")

            eval_groups = self.config.data.test.eval_groups

            if eval_groups:
                valid_score, valid_cmc = group_eval_fn(self.config, eval_groups, model)
                if self.rank == 0:
                    print('Group average score: ', valid_score)
            else:
                if self.rank == 0:
                    print('Evaluating on full test set')
                valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb and self.rank == 0)
                if self.rank == 0:
                    print('Valid score: ', valid_score)

            if valid_score > best_score:
                best_score = valid_score
                best_cmc = valid_cmc
                if self.rank == 0:
                    # Save the best model
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), f'{checkpoint_dir}/model_best.bin')
                    self.log_info(f'Best model found for epoch {epoch}')

        if swa_model and self.rank == 0:
            print("Updating SWA batchnorm statistics...")
            update_bn(train_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
            
        return best_score, best_cmc, model

    def setup_distributed(self):
        """Initialize distributed training using torchrun environment variables"""
        # torchrun sets these environment variables automatically
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        self.is_distributed = self.world_size > 1
        
        if self.is_distributed:
            # Initialize process group with NCCL backend for GPU training
            dist.init_process_group(backend='nccl')
            # Set the current device
            torch.cuda.set_device(self.local_rank)
            self.log_info(f"Initialized process group - Rank: {self.rank}/{self.world_size}, Local Rank: {self.local_rank}")
        else:
            self.log_info("Running in single GPU mode")
        
        return self.rank, self.world_size, self.local_rank
    
    def cleanup_distributed(self):
        """Clean up distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()

    def run(self, finetune=False):
        # Setup distributed training
        rank, world_size, local_rank = self.setup_distributed()
        
        try:
            config = self.config
            checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
            
            # Only rank 0 creates directories and writes config
            if rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.log_info(f'Checkpoints will be saved at: {checkpoint_dir}')
            
            # Synchronize all processes before proceeding
            if self.is_distributed:
                dist.barrier()

            config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
            config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

            # Set seed for reproducibility
            self.set_seed_torch(config.engine.seed + rank)

            # Data preprocessing
            df_train = preprocess_data(
                config.data.train.anno_path, 
                name_keys=config.data.name_keys,
                convert_names_to_ids=True, 
                viewpoint_list=config.data.viewpoint_list, 
                n_filter_min=config.data.train.n_filter_min, 
                n_subsample_max=config.data.train.n_subsample_max,
                use_full_image_path=config.data.use_full_image_path,
                images_dir=config.data.images_dir
            )

            df_val = preprocess_data(
                config.data.val.anno_path, 
                name_keys=config.data.name_keys,
                convert_names_to_ids=True, 
                viewpoint_list=config.data.viewpoint_list, 
                n_filter_min=config.data.val.n_filter_min, 
                n_subsample_max=config.data.val.n_subsample_max,
                use_full_image_path=config.data.use_full_image_path,
                images_dir=config.data.images_dir
            )

            if rank == 0:
                print_intersect_stats(df_train, df_val, individual_key='name_orig')
        
            n_train_classes = df_train['name'].nunique()

            crop_bbox = config.data.crop_bbox

            # Handle preprocessed images
            if config.data.preprocess_images.apply:
                if config.data.preprocess_images.preprocessed_dir is None:
                    preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
                else:
                    preprocess_dir_images = config.data.preprocess_images.preprocessed_dir

                if os.path.exists(preprocess_dir_images) and not config.data.preprocess_images.force_apply:
                    if rank == 0:
                        print('Preprocessed images directory found at: ', preprocess_dir_images)
                else:
                    if rank == 0:
                        preprocess_dataset(config, preprocess_dir_images)
                    if self.is_distributed:
                        dist.barrier()

                df_train = load_preprocessed_mapping(df_train, preprocess_dir_images)
                df_val = load_preprocessed_mapping(df_val, preprocess_dir_images)
                crop_bbox = False

            # Create datasets
            train_dataset = MiewIdDataset(
                csv=df_train,
                transforms=get_train_transforms((config.data.image_size[0], config.data.image_size[1])),
                fliplr=config.test.fliplr,
                fliplr_view=config.test.fliplr_view,
                crop_bbox=crop_bbox
            )
            
            valid_dataset = MiewIdDataset(
                csv=df_val,
                transforms=get_valid_transforms((config.data.image_size[0], config.data.image_size[1])),
                fliplr=config.test.fliplr,
                fliplr_view=config.test.fliplr_view,
                crop_bbox=crop_bbox
            )
            
            # Create distributed samplers for multi-GPU training
            if self.is_distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=config.engine.seed,
                    drop_last=True
                )
                valid_sampler = DistributedSampler(
                    valid_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False
                )
            else:
                train_sampler = None
                valid_sampler = None
            
            # Calculate per-GPU batch size
            train_batch_size_per_gpu = config.engine.train_batch_size // world_size
            valid_batch_size_per_gpu = config.engine.valid_batch_size // world_size
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size_per_gpu,
                num_workers=config.engine.num_workers,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if config.engine.num_workers > 0 else False
            )

            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=valid_batch_size_per_gpu,
                num_workers=config.engine.num_workers,
                sampler=valid_sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if config.engine.num_workers > 0 else False
            )

            # Set device
            device = torch.device(f'cuda:{local_rank}')

            # Update number of classes if needed
            if config.model_params.n_classes != n_train_classes:
                if rank == 0:
                    print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) with actual n_train_classes in dataset ({n_train_classes}).")
                config.model_params.n_classes = n_train_classes

            # Calculate margins for dynamic arcface if needed
            margins = None
            if config.model_params.loss_module == 'arcface_subcenter_dynamic':
                margin_min = 0.2
                margin_max = config.model_params.margin
                tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
                margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min

            # Initialize model
            if not self.model:
                self.model = MiewIdNet(**dict(config.model_params))
                if rank == 0:
                    self.log_info('Initialized model')

            model = self.model
            model = model.to(device)
            
            # Wrap model with DDP for distributed training
            if self.is_distributed:
                # Use SyncBatchNorm for better convergence
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(
                    model, 
                    device_ids=[local_rank], 
                    output_device=local_rank,
                    find_unused_parameters=False,  # Set to False for better performance
                    broadcast_buffers=True
                )

            # Initialize loss function
            loss_fn = fetch_loss()

            # Initialize criterion based on loss module type
            if config.model_params.loss_module == 'elastic_arcface':
                final_in_features = model.module.final_in_features if self.is_distributed else model.final_in_features
                criterion = ElasticArcFace(
                    loss_fn=loss_fn, 
                    in_features=final_in_features, 
                    out_features=config.model_params.n_classes
                )
                criterion = criterion.to(device)

            elif config.model_params.loss_module == 'arcface_subcenter_dynamic':
                if margins is None:
                    margins = [0.3] * n_train_classes
                final_in_features = model.module.final_in_features if self.is_distributed else model.final_in_features
                criterion = ArcFaceSubCenterDynamic(
                    loss_fn=loss_fn,
                    embedding_dim=final_in_features, 
                    output_classes=config.model_params.n_classes, 
                    margins=margins,
                    s=config.model_params.s,
                    k=config.model_params.k
                )
                criterion = criterion.to(device)
            else:
                raise NotImplementedError(f"Loss module {config.model_params.loss_module} not recognized")

            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(criterion.parameters()), 
                lr=config.scheduler_params.lr_start,
                weight_decay=config.scheduler_params.get('weight_decay', 0.0)
            )
            scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

            # Initialize SWA if enabled
            swa_model = None
            swa_scheduler = None
            swa_start = None
            if config.engine.use_swa:
                swa_model = AveragedModel(model)
                swa_model = swa_model.to(device)
                swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
                swa_start = config.swa_params.swa_start

            # Save config (only rank 0)
            if rank == 0:
                write_config(config, config_path_out)

            # Handle finetuning if needed
            if finetune:
                # Freeze model parameters for first stage
                for param in model.parameters():
                    param.requires_grad = False

                optimizer = torch.optim.Adam(
                    list(criterion.parameters()), 
                    lr=config.scheduler_params.lr_start
                )
                if rank == 0:
                    self.log_info('Frozen model parameters for finetuning stage 1')

                scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

                if config.engine.use_swa:
                    swa_model = AveragedModel(model)
                    swa_model = swa_model.to(device)
                    swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
                    swa_start = config.swa_params.swa_start

                # Stage 1: Train only the head
                epochs_orig = self.config.engine.epochs
                self.config.engine.epochs = 3
                if rank == 0:
                    print('Finetuning Stage 1: Training head only')
                
                use_wandb_context = config.engine.use_wandb and rank == 0
                if use_wandb_context:
                    with WandbContext(config):
                        best_score, best_cmc, model = self.run_fn(
                            model, train_loader, valid_loader, criterion, 
                            optimizer, scheduler, device, checkpoint_dir, 
                            use_wandb=True, swa_model=swa_model, 
                            swa_scheduler=swa_scheduler, swa_start=swa_start
                        )
                else:
                    best_score, best_cmc, model = self.run_fn(
                        model, train_loader, valid_loader, criterion, 
                        optimizer, scheduler, device, checkpoint_dir, 
                        use_wandb=False, swa_model=swa_model, 
                        swa_scheduler=swa_scheduler, swa_start=swa_start
                    )

                # Stage 2: Unfreeze and train everything
                if rank == 0:
                    print('Finetuning Stage 2: Training full model')
                
                for param in model.parameters():
                    param.requires_grad = True
            
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(criterion.parameters()), 
                    lr=config.scheduler_params.lr_start,
                    weight_decay=config.scheduler_params.get('weight_decay', 0.0)
                )
                scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))
            
                if config.engine.use_swa:
                    swa_model = AveragedModel(model)
                    swa_model = swa_model.to(device)
                    swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
                    swa_start = config.swa_params.swa_start
            
                self.config.engine.epochs = epochs_orig
            
                best_score, best_cmc, model = self.run_fn(
                    model, train_loader, valid_loader, criterion, 
                    optimizer, scheduler, device, checkpoint_dir, 
                    use_wandb=config.engine.use_wandb and rank == 0,
                    swa_model=swa_model, swa_scheduler=swa_scheduler, 
                    swa_start=swa_start
                )

                return best_score
            else:
                # Normal training (no finetuning)
                use_wandb_context = config.engine.use_wandb and rank == 0
                if use_wandb_context:
                    with WandbContext(config):
                        best_score, best_cmc, model = self.run_fn(
                            model, train_loader, valid_loader, criterion, 
                            optimizer, scheduler, device, checkpoint_dir, 
                            use_wandb=True, swa_model=swa_model, 
                            swa_scheduler=swa_scheduler, swa_start=swa_start
                        )
                else:
                    best_score, best_cmc, model = self.run_fn(
                        model, train_loader, valid_loader, criterion, 
                        optimizer, scheduler, device, checkpoint_dir, 
                        use_wandb=False, swa_model=swa_model, 
                        swa_scheduler=swa_scheduler, swa_start=swa_start
                    )
                
                return best_score
                
        finally:
            # Always cleanup distributed training
            self.cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training script for MiewId")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Enable finetuning mode (freeze backbone initially)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)
    
    # Create trainer and run training
    trainer = Trainer(config)
    best_score = trainer.run(finetune=args.finetune)
    
    # Log final results
    if trainer.rank == 0:
        print(f"Training completed. Best validation score: {best_score}")


if __name__ == '__main__':
    main()