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
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb


class Trainer:
    def __init__(self, config, model=None, rank=0, world_size=1):
        self.config = config
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

    def set_seed_torch(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def run_fn(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
        best_score = 0
        best_cmc = None
        for epoch in range(self.config.engine.epochs):
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)
            train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch, use_wandb=use_wandb and self.rank == 0, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

            print("\nGetting metrics on validation set...")

            eval_groups = self.config.data.test.eval_groups

            if eval_groups:
                valid_score, valid_cmc = group_eval_fn(self.config, eval_groups, model)
                print('Group average score: ', valid_score)
            else:
                print('Evaluating on full test set')
                valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb)
                print('Valid score: ', valid_score)

            if valid_score > best_score:
                best_score = valid_score
                best_cmc = valid_cmc
                if self.rank == 0:
                    if self.is_distributed:
                        torch.save(model.module.state_dict(), f'{checkpoint_dir}/model_best.bin')
                    else:
                        torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
                    print('best model found for epoch {}'.format(epoch))

        if swa_model and self.rank == 0:
            print("Updating SWA batchnorm statistics...")
            update_bn(train_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
            
        return best_score, best_cmc, model

    def setup_distributed(self):
        """Initialize distributed training"""
        if 'SLURM_PROCID' in os.environ:
            # SLURM environment
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
            
            # Get master address and port from SLURM
            node_list = os.environ['SLURM_NODELIST']
            master_addr = node_list.split(',')[0].strip('[')
            if '[' in master_addr:
                master_addr = master_addr.split('[')[0]
            
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            
        else:
            # Standard distributed setup
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        
        if self.is_distributed:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank

    def run(self, finetune=False):
        # Setup distributed training
        rank, world_size, local_rank = self.setup_distributed()
        
        config = self.config
        checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print('Checkpoints will be saved at: ', checkpoint_dir)
        
        # Synchronize all processes
        if self.is_distributed:
            dist.barrier()

        config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
        config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

        self.set_seed_torch(config.engine.seed)

        df_train = preprocess_data(config.data.train.anno_path, 
                                    name_keys=config.data.name_keys,
                                    convert_names_to_ids=True, 
                                    viewpoint_list=config.data.viewpoint_list, 
                                    n_filter_min=config.data.train.n_filter_min, 
                                    n_subsample_max=config.data.train.n_subsample_max,
                                    use_full_image_path=config.data.use_full_image_path,
                                    images_dir=config.data.images_dir)

        df_val = preprocess_data(config.data.val.anno_path, 
                                  name_keys=config.data.name_keys,
                                  convert_names_to_ids=True, 
                                  viewpoint_list=config.data.viewpoint_list, 
                                  n_filter_min=config.data.val.n_filter_min, 
                                  n_subsample_max=config.data.val.n_subsample_max,
                                  use_full_image_path=config.data.use_full_image_path,
                                  images_dir=config.data.images_dir)


    
        print_intersect_stats(df_train, df_val, individual_key='name_orig')
    
        n_train_classes = df_train['name'].nunique()

        crop_bbox = config.data.crop_bbox

        if config.data.preprocess_images.apply:
            if config.data.preprocess_images.preprocessed_dir is None:
                preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
            else:
                preprocess_dir_images = config.data.preprocess_images.preprocessed_dir

            if os.path.exists(preprocess_dir_images) and not config.data.preprocess_images.force_apply:
                print('Preprocessed images directory found at: ', preprocess_dir_images)
            else:
                preprocess_dataset(config, preprocess_dir_images)

            df_train = load_preprocessed_mapping(df_train, preprocess_dir_images)
            df_val = load_preprocessed_mapping(df_val, preprocess_dir_images)
            crop_bbox = False

        train_dataset = MiewIdDataset(
            csv=df_train,
            transforms=get_train_transforms((config.data.image_size[0], config.data.image_size[1])),
            fliplr=config.test.fliplr,
            fliplr_view=config.test.fliplr_view,
            crop_bbox=crop_bbox)
        
        valid_dataset = MiewIdDataset(
            csv=df_val,
            transforms=get_valid_transforms((config.data.image_size[0], config.data.image_size[1])),
            fliplr=config.test.fliplr,
            fliplr_view=config.test.fliplr_view,
            crop_bbox=crop_bbox)
        
        # Create distributed samplers
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            train_sampler = None
            valid_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.engine.train_batch_size // world_size,  # Scale batch size
            num_workers=config.engine.num_workers,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            pin_memory=True,
            drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.engine.valid_batch_size // world_size,  # Scale batch size
            num_workers=config.engine.num_workers,
            sampler=valid_sampler,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        if self.is_distributed:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device(config.engine.device)

        if config.model_params.n_classes != n_train_classes:
            print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the dataset - ({n_train_classes}).")
            config.model_params.n_classes = n_train_classes

        if config.model_params.loss_module == 'arcface_subcenter_dynamic':
            margin_min = 0.2
            margin_max = config.model_params.margin
            tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
            margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min
        else:
            margins = None

        if not self.model:
            self.model = MiewIdNet(**dict(config.model_params))
            if rank == 0:
                print('Initialized model')

        model = self.model
        model.to(device)
        
        # Wrap model with DDP for distributed training
        if self.is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        loss_fn = fetch_loss()

        if config.model_params.loss_module == 'elastic_arcface':
            final_in_features = model.module.final_in_features if self.is_distributed else model.final_in_features
            criterion = ElasticArcFace(loss_fn=loss_fn, in_features=final_in_features, out_features=config.model_params.n_classes)
            criterion.to(device)

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
                    k=config.model_params.k)
            criterion.to(device)
        else:
            raise NotImplementedError("Loss module not recognized")

        optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=config.scheduler_params.lr_start)
        scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

        if config.engine.use_swa:
            swa_model = AveragedModel(model)
            swa_model.to(device)
            swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
            swa_start = config.swa_params.swa_start
        else:
            swa_model = None
            swa_scheduler = None
            swa_start = None

        if rank == 0:
            write_config(config, config_path_out)

        if finetune:
            for param in model.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(list(criterion.parameters()), lr=config.scheduler_params.lr_start) 
            print('frozen parameters')

            scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

            if config.engine.use_swa:
                swa_model = AveragedModel(model)
                swa_model.to(device)
                swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
                swa_start = config.swa_params.swa_start
            else:
                swa_model = None
                swa_scheduler = None
                swa_start = None

            if rank == 0:
                write_config(config, config_path_out)

            epochs_orig = self.config.engine.epochs
            self.config.engine.epochs = 3
            print('Finetuning Stage 1')
            use_wandb_context = config.engine.use_wandb and rank == 0
            if use_wandb_context:
                with WandbContext(config):
                    best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
                                            swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
            else:
                best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=False,
                                        swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)

                print('Finetuning Stage 2')
                for param in model.parameters():
                    param.requires_grad = True
        
                optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=config.scheduler_params.lr_start)
                scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))
        
                if config.engine.use_swa:
                    swa_model = AveragedModel(model)
                    swa_model.to(device)
                    swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
                    swa_start = config.swa_params.swa_start
                else:
                    swa_model = None
                    swa_scheduler = None
                    swa_start = None
        
                if rank == 0:
                    write_config(config, config_path_out)
        
                self.config.engine.epochs = epochs_orig
        

                best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
                                        swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)

            return best_score
        else:
            use_wandb_context = config.engine.use_wandb and rank == 0
            if use_wandb_context:
                with WandbContext(config):
                    best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
                                            swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
            else:
                best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=False,
                                        swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
        
        # Cleanup distributed training
        if self.is_distributed:
            dist.destroy_process_group()

        return best_score

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    parser.add_argument(
        '--nodes',
        type=int,
        default=1,
        help='Number of nodes for distributed training'
    )
    parser.add_argument(
        '--gpus-per-node',
        type=int,
        default=1,
        help='Number of GPUs per node'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)
    trainer = Trainer(config)
    trainer.run()
