# from wbia_miew_id.logging_utils import WandbContext
# from wbia_miew_id.etl import preprocess_data, print_intersect_stats, load_preprocessed_mapping, preprocess_dataset
# from wbia_miew_id.losses import fetch_loss
# from wbia_miew_id.schedulers import MiewIdScheduler
# from wbia_miew_id.helpers import get_config, write_config, update_bn
# from wbia_miew_id.metrics import AverageMeter, compute_calibration
# from wbia_miew_id.datasets import MiewIdDataset, get_train_transforms, get_valid_transforms
# from wbia_miew_id.models import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic, MiewIdNet
# from wbia_miew_id.engine import train_fn, eval_fn, group_eval_fn


# from torch.optim.swa_utils import AveragedModel, SWALR
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# import random
# import numpy as np
# import os
# import argparse
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm
# import wandb


# class Trainer:
#     def __init__(self, config, model=None, rank=0, world_size=1):
#         self.config = config
#         self.model = model
#         self.rank = rank
#         self.world_size = world_size
#         self.is_distributed = world_size > 1

#     def set_seed_torch(self, seed):
#         random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True

#     def run_fn(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
#         best_score = 0
#         best_cmc = None
#         for epoch in range(self.config.engine.epochs):
#             if self.is_distributed:
#                 train_loader.sampler.set_epoch(epoch)
#             train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch, use_wandb=use_wandb and self.rank == 0, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

#             print("\nGetting metrics on validation set...")

#             eval_groups = self.config.data.test.eval_groups

#             if eval_groups:
#                 valid_score, valid_cmc = group_eval_fn(self.config, eval_groups, model)
#                 print('Group average score: ', valid_score)
#             else:
#                 print('Evaluating on full test set')
#                 valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb)
#                 print('Valid score: ', valid_score)

#             if valid_score > best_score:
#                 best_score = valid_score
#                 best_cmc = valid_cmc
#                 if self.rank == 0:
#                     if self.is_distributed:
#                         torch.save(model.module.state_dict(), f'{checkpoint_dir}/model_best.bin')
#                     else:
#                         torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
#                     print('best model found for epoch {}'.format(epoch))

#         if swa_model and self.rank == 0:
#             print("Updating SWA batchnorm statistics...")
#             update_bn(train_loader, swa_model, device=device)
#             torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
            
#         return best_score, best_cmc, model

#     def setup_distributed(self):
#         # Prefer torchrun env if present
#         if os.getenv("LOCAL_RANK") is not None:
#             rank = int(os.environ["RANK"])
#             world_size = int(os.environ["WORLD_SIZE"])
#             local_rank = int(os.environ["LOCAL_RANK"])
#         elif os.getenv("SLURM_PROCID") is not None:
#             rank = int(os.environ["SLURM_PROCID"])
#             world_size = int(os.environ["SLURM_NTASKS"])
#             local_rank = int(os.environ["SLURM_LOCALID"])
#             # Set master for SLURM case
#             node_list = os.environ["SLURM_NODELIST"]
#             master_addr = node_list.split(",")[0].split("[")[0]
#             os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", master_addr)
#             os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
#             os.environ["RANK"] = str(rank)
#             os.environ["WORLD_SIZE"] = str(world_size)
#             os.environ["LOCAL_RANK"] = str(local_rank)
#         else:
#             rank = int(os.environ.get("RANK", 0))
#             world_size = int(os.environ.get("WORLD_SIZE", 1))
#             local_rank = int(os.environ.get("LOCAL_RANK", 0))

#         self.rank = rank
#         self.world_size = world_size
#         self.local_rank = local_rank
#         self.is_distributed = world_size > 1

#         if self.is_distributed:
#             dist.init_process_group(backend="nccl", init_method="env://",
#                                 world_size=world_size, rank=rank)
#         # Avoid invalid device ordinal if visibility is restricted
#             dev_count = torch.cuda.device_count()
#             torch.cuda.set_device(local_rank if local_rank < dev_count else 0)

#         return rank, world_size, local_rank


#     def run(self, finetune=False):
#         # Setup distributed training
#         rank, world_size, local_rank = self.setup_distributed()
        
#         config = self.config
#         checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
#         if rank == 0:
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             print('Checkpoints will be saved at: ', checkpoint_dir)
        
#         # Synchronize all processes
#         if self.is_distributed:
#             dist.barrier()

#         config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
#         config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

#         self.set_seed_torch(config.engine.seed)

#         df_train = preprocess_data(config.data.train.anno_path, 
#                                     name_keys=config.data.name_keys,
#                                     convert_names_to_ids=True, 
#                                     viewpoint_list=config.data.viewpoint_list, 
#                                     n_filter_min=config.data.train.n_filter_min, 
#                                     n_subsample_max=config.data.train.n_subsample_max,
#                                     use_full_image_path=config.data.use_full_image_path,
#                                     images_dir=config.data.images_dir)

#         df_val = preprocess_data(config.data.val.anno_path, 
#                                   name_keys=config.data.name_keys,
#                                   convert_names_to_ids=True, 
#                                   viewpoint_list=config.data.viewpoint_list, 
#                                   n_filter_min=config.data.val.n_filter_min, 
#                                   n_subsample_max=config.data.val.n_subsample_max,
#                                   use_full_image_path=config.data.use_full_image_path,
#                                   images_dir=config.data.images_dir)


    
#         print_intersect_stats(df_train, df_val, individual_key='name_orig')
    
#         n_train_classes = df_train['name'].nunique()

#         crop_bbox = config.data.crop_bbox

#         if config.data.preprocess_images.apply:
#             if config.data.preprocess_images.preprocessed_dir is None:
#                 preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
#             else:
#                 preprocess_dir_images = config.data.preprocess_images.preprocessed_dir

#             if os.path.exists(preprocess_dir_images) and not config.data.preprocess_images.force_apply:
#                 print('Preprocessed images directory found at: ', preprocess_dir_images)
#             else:
#                 preprocess_dataset(config, preprocess_dir_images)

#             df_train = load_preprocessed_mapping(df_train, preprocess_dir_images)
#             df_val = load_preprocessed_mapping(df_val, preprocess_dir_images)
#             crop_bbox = False

#         train_dataset = MiewIdDataset(
#             csv=df_train,
#             transforms=get_train_transforms((config.data.image_size[0], config.data.image_size[1])),
#             fliplr=config.test.fliplr,
#             fliplr_view=config.test.fliplr_view,
#             crop_bbox=crop_bbox)
        
#         valid_dataset = MiewIdDataset(
#             csv=df_val,
#             transforms=get_valid_transforms((config.data.image_size[0], config.data.image_size[1])),
#             fliplr=config.test.fliplr,
#             fliplr_view=config.test.fliplr_view,
#             crop_bbox=crop_bbox)
        
#         # Create distributed samplers
#         if self.is_distributed:
#             train_sampler = DistributedSampler(
#                 train_dataset,
#                 num_replicas=world_size,
#                 rank=rank,
#                 shuffle=True
#             )
#             valid_sampler = DistributedSampler(
#                 valid_dataset,
#                 num_replicas=world_size,
#                 rank=rank,
#                 shuffle=False
#             )
#         else:
#             train_sampler = None
#             valid_sampler = None
        
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=config.engine.train_batch_size // world_size,  # Scale batch size
#             num_workers=config.engine.num_workers,
#             sampler=train_sampler,
#             shuffle=(train_sampler is None),
#             pin_memory=True,
#             drop_last=True)

#         valid_loader = torch.utils.data.DataLoader(
#             valid_dataset,
#             batch_size=config.engine.valid_batch_size // world_size,  # Scale batch size
#             num_workers=config.engine.num_workers,
#             sampler=valid_sampler,
#             shuffle=False,
#             pin_memory=True,
#             drop_last=False)

#         if self.is_distributed:
#             device = torch.device(f'cuda:{local_rank}')
#         else:
#             device = torch.device(config.engine.device)

#         if config.model_params.n_classes != n_train_classes:
#             print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the dataset - ({n_train_classes}).")
#             config.model_params.n_classes = n_train_classes

#         if config.model_params.loss_module == 'arcface_subcenter_dynamic':
#             margin_min = 0.2
#             margin_max = config.model_params.margin
#             tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
#             margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min
#         else:
#             margins = None

#         if not self.model:
#             self.model = MiewIdNet(**dict(config.model_params))
#             if rank == 0:
#                 print('Initialized model')

#         model = self.model
#         model.to(device)
        
#         # Wrap model with DDP for distributed training
#         if self.is_distributed:
#             model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

#         loss_fn = fetch_loss()

#         if config.model_params.loss_module == 'elastic_arcface':
#             final_in_features = model.module.final_in_features if self.is_distributed else model.final_in_features
#             criterion = ElasticArcFace(loss_fn=loss_fn, in_features=final_in_features, out_features=config.model_params.n_classes)
#             criterion.to(device)

#         elif config.model_params.loss_module == 'arcface_subcenter_dynamic':
#             if margins is None:
#                 margins = [0.3] * n_train_classes
#             final_in_features = model.module.final_in_features if self.is_distributed else model.final_in_features
#             criterion = ArcFaceSubCenterDynamic(
#                     loss_fn=loss_fn,
#                     embedding_dim=final_in_features, 
#                     output_classes=config.model_params.n_classes, 
#                     margins=margins,
#                     s=config.model_params.s,
#                     k=config.model_params.k)
#             criterion.to(device)
#         else:
#             raise NotImplementedError("Loss module not recognized")

#         optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=config.scheduler_params.lr_start)
#         scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

#         if config.engine.use_swa:
#             swa_model = AveragedModel(model)
#             swa_model.to(device)
#             swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
#             swa_start = config.swa_params.swa_start
#         else:
#             swa_model = None
#             swa_scheduler = None
#             swa_start = None

#         if rank == 0:
#             write_config(config, config_path_out)

#         if finetune:
#             for param in model.parameters():
#                 param.requires_grad = False

#             optimizer = torch.optim.Adam(list(criterion.parameters()), lr=config.scheduler_params.lr_start) 
#             print('frozen parameters')

#             scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

#             if config.engine.use_swa:
#                 swa_model = AveragedModel(model)
#                 swa_model.to(device)
#                 swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
#                 swa_start = config.swa_params.swa_start
#             else:
#                 swa_model = None
#                 swa_scheduler = None
#                 swa_start = None

#             if rank == 0:
#                 write_config(config, config_path_out)

#             epochs_orig = self.config.engine.epochs
#             self.config.engine.epochs = 3
#             print('Finetuning Stage 1')
#             use_wandb_context = config.engine.use_wandb and rank == 0
#             if use_wandb_context:
#                 with WandbContext(config):
#                     best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
#                                             swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
#             else:
#                 best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=False,
#                                         swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)

#                 print('Finetuning Stage 2')
#                 for param in model.parameters():
#                     param.requires_grad = True
        
#                 optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=config.scheduler_params.lr_start)
#                 scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))
        
#                 if config.engine.use_swa:
#                     swa_model = AveragedModel(model)
#                     swa_model.to(device)
#                     swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
#                     swa_start = config.swa_params.swa_start
#                 else:
#                     swa_model = None
#                     swa_scheduler = None
#                     swa_start = None
        
#                 if rank == 0:
#                     write_config(config, config_path_out)
        
#                 self.config.engine.epochs = epochs_orig
        

#                 best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
#                                         swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)

#             return best_score
#         else:
#             use_wandb_context = config.engine.use_wandb and rank == 0
#             if use_wandb_context:
#                 with WandbContext(config):
#                     best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
#                                             swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
#             else:
#                 best_score, best_cmc, model = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=False,
#                                         swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
        
#         # Cleanup distributed training
#         if self.is_distributed:
#             dist.destroy_process_group()

#         return best_score

# def parse_args():
#     parser = argparse.ArgumentParser(description="Load configuration file.")
#     parser.add_argument(
#         '--config',
#         type=str,
#         default='configs/default_config.yaml',
#         help='Path to the YAML configuration file. Default: configs/default_config.yaml'
#     )
#     parser.add_argument(
#         '--nodes',
#         type=int,
#         default=1,
#         help='Number of nodes for distributed training'
#     )
#     parser.add_argument(
#         '--gpus-per-node',
#         type=int,
#         default=1,
#         help='Number of GPUs per node'
#     )
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_args()
#     config_path = args.config
#     config = get_config(config_path)
#     trainer = Trainer(config)
#     trainer.run()




# train_parallel.py

from wbia_miew_id.logging_utils import WandbContext
from wbia_miew_id.etl import preprocess_data, print_intersect_stats, load_preprocessed_mapping, preprocess_dataset
from wbia_miew_id.losses import fetch_loss
from wbia_miew_id.schedulers import MiewIdScheduler
from wbia_miew_id.helpers import get_config, write_config, update_bn
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
import time
import datetime


def _set_nccl_env_defaults():
    # You can override any of these by exporting the var before launch
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")         # keep IB off on this node
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")     # force loopback for single-node
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")        # TEMP: disable CUDA P2P/IPC to test hangs
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")        # avoid SHM transport issues


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

    def _is_main(self):
        return (not self.is_distributed) or self.rank == 0

    def setup_distributed(self):
        _set_nccl_env_defaults()

        # Prefer torchrun env if present
        if os.getenv("LOCAL_RANK") is not None:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        elif os.getenv("SLURM_PROCID") is not None:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            local_rank = int(os.environ["SLURM_LOCALID"])
            # master addr/port for SLURM
            node_list = os.environ["SLURM_NODELIST"]
            master_addr = node_list.split(",")[0].split("[")[0]
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(local_rank)
        else:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1

        if self.is_distributed:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(seconds=300),
            )
            # Avoid invalid device ordinal if visibility is restricted
            dev_count = torch.cuda.device_count()
            torch.cuda.set_device(local_rank if local_rank < dev_count else 0)

        return rank, world_size, local_rank

    def _sanity_allreduce_nccl(self, device):
        if not self.is_distributed:
            return
        print(f"[Rank {self.rank}] NCCL sanity all_reduce (pre-DDP) ...")
        x = torch.ones(1, device=device) * (self.rank + 1)
        dist.all_reduce(x)  # will hang here if NCCL transport is unhappy
        if self._is_main():
            print(f"[DDP sanity pre-DDP] all_reduce OK, sum={x.item()}")

    def _sanity_allreduce_gloo(self):
        # CPU-only: prove PyTorch distributed itself is fine, and isolate NCCL issues.
        if not self.is_distributed:
            return
        print(f"[Rank {self.rank}] Gloo control test (CPU all_reduce) ...")
        try:
            g = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=60))
            x = torch.tensor([self.rank + 1], dtype=torch.int32)
            dist.all_reduce(x, group=g)
            if self._is_main():
                print(f"[Gloo control] all_reduce OK, sum={int(x.item())}")
            dist.destroy_process_group(g)
        except Exception as e:
            print(f"[Rank {self.rank}] Gloo control test FAILED: {repr(e)}")

    def run_fn(self, model, train_loader, valid_loader, criterion, optimizer, scheduler,
               device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
        best_score = -1e9
        best_cmc = None

        for epoch in range(self.config.engine.epochs):
            if self.is_distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            print(f"[Rank {self.rank}] === Epoch {epoch} | starting train ===")
            train_loss = train_fn(
                train_loader, model, criterion, optimizer, device,
                scheduler=scheduler, epoch=epoch,
                use_wandb=(use_wandb and self._is_main()),
                swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler
            )

            if self.is_distributed:
                torch.cuda.synchronize(device)
                dist.barrier()

            print(f"[Rank {self.rank}] === Epoch {epoch} | starting eval (main rank only) ===")
            eval_groups = self.config.data.test.eval_groups

            valid_score = None
            valid_cmc = None

            if self._is_main():
                if eval_groups:
                    valid_score, valid_cmc = group_eval_fn(self.config, eval_groups, model)
                    print(f"[Rank {self.rank}] Group average score: {valid_score}")
                else:
                    valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb)
                    print(f"[Rank {self.rank}] Valid score: {valid_score}")

            if self.is_distributed:
                score_tensor = torch.tensor(
                    [-1.0 if valid_score is None else float(valid_score)],
                    device=device
                )
                dist.broadcast(score_tensor, src=0)
                valid_score = score_tensor.item()

            if valid_score > best_score:
                best_score = valid_score
                best_cmc = valid_cmc
                if self._is_main():
                    to_save = model.module if isinstance(model, DDP) else model
                    torch.save(to_save.state_dict(), f'{checkpoint_dir}/model_best.bin')
                    print(f"[Rank {self.rank}] best model found for epoch {epoch}")

            if self.is_distributed:
                dist.barrier()

        if swa_model and self._is_main():
            print("[Main] Updating SWA batchnorm statistics...")
            update_bn(train_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')

        return best_score, best_cmc, model

    def run(self, finetune=False):
        # Setup distributed training
        rank, world_size, local_rank = self.setup_distributed()

        config = self.config
        checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
        if self._is_main():
            os.makedirs(checkpoint_dir, exist_ok=True)
            print('Checkpoints will be saved at: ', checkpoint_dir)
        print(f"[Rank {self.rank}] Setup done. local_rank={self.local_rank}, world_size={self.world_size}")
        if self._is_main():
            print("[Main] NCCL env:",
                    {k: os.environ.get(k) for k in ["NCCL_DEBUG","NCCL_ASYNC_ERROR_HANDLING","NCCL_IB_DISABLE","NCCL_SOCKET_IFNAME","NCCL_P2P_DISABLE","NCCL_SHM_DISABLE"]})

        config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
        config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

        self.set_seed_torch(config.engine.seed)

        # -------------------- Data prep --------------------
        print(f"[Rank {self.rank}] Starting TRAIN preprocess_data: {config.data.train.anno_path}")
        t0 = time.time()
        df_train = preprocess_data(
            config.data.train.anno_path,
            name_keys=config.data.name_keys,
            convert_names_to_ids=True,
            viewpoint_list=config.data.viewpoint_list,
            n_filter_min=config.data.train.n_filter_min,
            n_subsample_max=config.data.train.n_subsample_max,
            use_full_image_path=config.data.use_full_image_path,
            images_dir=config.data.images_dir,
        )
        print(f"[Rank {self.rank}] TRAIN preprocess_data done in {time.time()-t0:.2f}s | shape={getattr(df_train,'shape',None)}")
        df_val = preprocess_data(
            config.data.val.anno_path,
            name_keys=config.data.name_keys,
            convert_names_to_ids=True,
            viewpoint_list=config.data.viewpoint_list,
            n_filter_min=config.data.val.n_filter_min,
            n_subsample_max=config.data.val.n_subsample_max,
            use_full_image_path=config.data.use_full_image_path,
            images_dir=config.data.images_dir,
        )
        print(f"[Rank {self.rank}] VAL preprocess_data done in {time.time()-t0:.2f}s | shape={getattr(df_val,'shape',None)}")

        if self._is_main():
            print_intersect_stats(df_train, df_val, individual_key='name_orig')

        n_train_classes = df_train['name'].nunique()
        crop_bbox = config.data.crop_bbox

        if config.data.preprocess_images.apply:
            if config.data.preprocess_images.preprocessed_dir is None:
                preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
            else:
                preprocess_dir_images = config.data.preprocess_images.preprocessed_dir

            if self._is_main():
                if os.path.exists(preprocess_dir_images) and not config.data.preprocess_images.force_apply:
                    print('[Main] Preprocessed images directory found at: ', preprocess_dir_images)
                else:
                    print('[Main] Running preprocess_dataset(...)')
                    preprocess_dataset(config, preprocess_dir_images)

            if self.is_distributed:
                dist.barrier()
                if self._is_main():
                    print('[Main] Barrier after optional image preprocessing')

            print(f"[Rank {self.rank}] Loading preprocessed mappings from {preprocess_dir_images}")
            df_train = load_preprocessed_mapping(df_train, preprocess_dir_images)
            df_val = load_preprocessed_mapping(df_val, preprocess_dir_images)
            crop_bbox = False

        # -------------------- Datasets and Loaders --------------------
        print(f"[Rank {self.rank}] Creating datasets...")
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
        print(f"[Rank {self.rank}] Datasets ready | train={len(train_dataset)} val={len(valid_dataset)}")

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            valid_sampler = None  # rank-0 only eval
        else:
            train_sampler = None
            valid_sampler = None

        num_workers = 0
        timeout = 0
        pin_memory = False

        print(f"[Rank {self.rank}] Creating DataLoaders (safe mode: num_workers={num_workers}, timeout={timeout})...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=max(1, config.engine.train_batch_size // max(1, world_size)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            timeout=timeout,
        )

        if self._is_main():
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=max(1, config.engine.valid_batch_size),
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=valid_sampler,
                shuffle=False,
                drop_last=False,
                timeout=timeout,
            )
        else:
            valid_loader = None
        print(f"[Rank {self.rank}] DataLoaders created")

        # -------------------- Device --------------------
        if self.is_distributed:
            device = torch.device(f'cuda:{self.local_rank if self.local_rank < torch.cuda.device_count() else 0}')
        else:
            device = torch.device(config.engine.device)
        print(f"[Rank {self.rank}] Using device: {device}")

        # -------------------- Model and Criterion Wrapper --------------------
        print(f"[Rank {self.rank}] Creating model and criterion wrapper...")

        if config.model_params.n_classes != n_train_classes:
            print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) "
                    f"which is different from actual n_train_classes in the dataset - ({n_train_classes}).")
            config.model_params.n_classes = n_train_classes

        # Create the backbone model
        self.model = MiewIdNet(**dict(config.model_params))
        final_in_features = self.model.final_in_features

        # Create the criterion based on the config
        loss_fn = fetch_loss()
        if config.model_params.loss_module == 'elastic_arcface':
            criterion = ElasticArcFace(loss_fn=loss_fn, in_features=final_in_features,
                                        out_features=config.model_params.n_classes)
        elif config.model_params.loss_module == 'arcface_subcenter_dynamic':
            margins = None
            if hasattr(config.model_params, 'loss_config') and 'dynamic_margin' in config.model_params.loss_config and config.model_params.loss_config.dynamic_margin:
                margin_min = 0.2
                margin_max = config.model_params.margin
                tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
                margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min
            else:
                margins = [config.model_params.margin] * n_train_classes
            criterion = ArcFaceSubCenterDynamic(
                loss_fn=loss_fn,
                embedding_dim=final_in_features,
                output_classes=config.model_params.n_classes,
                margins=margins,
                s=config.model_params.s,
                k=config.model_params.k
            )
        else:
            raise NotImplementedError("Loss module not recognized")

        # Combine the backbone and the criterion into a single module
        model_with_criterion = CombinedModelWithCriterion(self.model, criterion).to(device)

        # ---- NCCL sanity BEFORE DDP ----
        self._sanity_allreduce_nccl(device)
        self._sanity_allreduce_gloo()

        # ---- Wrap with DDP (simple flags) ----
        if self.is_distributed:
            model = DDP(
                model_with_criterion,
                device_ids=[self.local_rank if self.local_rank < torch.cuda.device_count() else 0],
                output_device=(self.local_rank if self.local_rank < torch.cuda.device_count() else 0),
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            print(f"[Rank {self.rank}] DDP wrapper complete")
        else:
            model = model_with_criterion

        optimizer = torch.optim.Adam(
            model.parameters(), # Optimize ALL parameters of the combined model
            lr=config.scheduler_params.lr_start
        )
        scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

        # -------------------- SWA and W&B --------------------
        if config.engine.use_swa:
            # SWA model also needs to be the combined one
            swa_model = AveragedModel(model.module if self.is_distributed else model)
            swa_model.to(device)
            swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
            swa_start = config.swa_params.swa_start
        else:
            swa_model = None
            swa_scheduler = None
            swa_start = None

        if self._is_main():
            write_config(config, config_path_out)

        # -------------------- Train --------------------
        use_wandb_context = config.engine.use_wandb and self._is_main()
        if use_wandb_context:
            with WandbContext(config):
                best_score, best_cmc, model = self.run_fn(
                    model, train_loader, valid_loader, model.module.criterion if self.is_distributed else model.criterion, optimizer, scheduler,
                    device, checkpoint_dir, use_wandb=config.engine.use_wandb,
                    swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start
                )
        else:
            best_score, best_cmc, model = self.run_fn(
                model, train_loader, valid_loader, model.module.criterion if self.is_distributed else model.criterion, optimizer, scheduler,
                device, checkpoint_dir, use_wandb=False,
                swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start
            )

        # Cleanup distributed training
        if self.is_distributed:
            dist.barrier()
            dist.destroy_process_group()

        return best_score

class CombinedModelWithCriterion(torch.nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        embeddings = self.model(*args, **kwargs)
        return embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the YAML configuration file. Default: configs/default_config.yaml')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--gpus-per-node', type=int, default=1, help='Number of GPUs per node')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)
    trainer = Trainer(config)
    trainer.run()
