# import torch
# from tqdm.auto import tqdm
# import pandas as pd
# import numpy as np
# import wandb

# from wbia_miew_id.metrics import AverageMeter, compute_distance_matrix, compute_calibration, eval_onevsall, topk_average_precision, precision_at_k, get_accuracy
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast

# def _unwrap(model):
#     return model.module if isinstance(model, DDP) else model

# def extract_embeddings(data_loader, model, device):
#     base = _unwrap(model)
#     base.eval()
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     embeddings, labels = [], []
#     with torch.no_grad():
#         for batch in tk0:
#             with autocast(enabled=torch.cuda.is_available()):
#                 batch_embeddings = base.extract_feat(batch["image"].to(device, non_blocking=True))
#             batch_embeddings = batch_embeddings.detach().cpu().numpy()
#             image_idx = batch["image_idx"].tolist()
#             embeddings.append(pd.DataFrame(batch_embeddings, index=image_idx))
#             labels.extend(batch['label'].tolist())
#     embeddings = pd.concat(embeddings).values
#     assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"
#     return embeddings, labels

# def extract_logits(data_loader, model, device):
#     base = _unwrap(model)
#     base.eval()
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     logits, labels = [], []
#     with torch.no_grad():
#         for batch in tk0:
#             with autocast(enabled=torch.cuda.is_available()):
#                 batch_logits = base.extract_logits(
#                     batch["image"].to(device, non_blocking=True),
#                     batch["label"].to(device, non_blocking=True),
#                 ).detach().cpu()
#             image_idx = batch["image_idx"].tolist()
#             logits.append(pd.DataFrame(batch_logits.numpy(), index=image_idx))
#             labels.extend(batch['label'].tolist())
#     logits = pd.concat(logits).values
#     assert not np.isnan(logits).sum(), "NaNs found in extracted logits"
#     return logits, labels

# def calculate_matches(embeddings, labels, embeddings_db=None, labels_db=None, dist_metric='cosine', ranks=list(range(1, 21)), mask_matrix=None):

#     q_pids = np.array(labels)
    
#     qf = torch.Tensor(embeddings)
#     if embeddings_db is not None:
#         print('embeddings_db not note')
#         dbf = torch.Tensor(embeddings_db)
#         labels_db = np.array(labels_db)
#     else:
#         dbf = qf
#         labels_db = np.array(labels)
#         mask_matrix_diagonal = np.full((embeddings.shape[0], embeddings.shape[0]), False)
#         np.fill_diagonal(mask_matrix_diagonal, True)
#         if mask_matrix is not None:
#             mask_matrix = np.logical_or(mask_matrix_diagonal, mask_matrix)
#         else:
#             mask_matrix = mask_matrix_diagonal

#     distmat = compute_distance_matrix(qf, dbf, dist_metric)

#     distmat = distmat.numpy()

#     if mask_matrix is not None:
#         assert mask_matrix.shape == distmat.shape, "Mask matrix must have same shape as distance matrix"
#         distmat[mask_matrix] = np.inf

#     print("Computing CMC and mAP ...")

#     mAP = topk_average_precision(q_pids, distmat, names_db=labels_db, k=None)
#     cmc, match_mat, topk_idx, topk_names = precision_at_k(q_pids, distmat, names_db=labels_db, ranks=ranks, return_matches=True)
#     print(f"Computed rank metrics on {match_mat.shape[0]} examples")

#     return mAP, cmc, (embeddings, q_pids, distmat)

# def calculate_calibration(logits, labels, logits_db=None, labels_db=None):

#     q_pids = np.array(labels)
#     confidences = torch.softmax(torch.Tensor(logits), dim=1)
#     top_confidences, pred_labels = confidences.max(dim=1)
#     pred_labels = pred_labels.numpy()
#     top_confidences = top_confidences.numpy()

#     print("Computing ECE ...")
#     results = compute_calibration(q_pids, pred_labels, top_confidences, num_bins=10)
#     ece = results['expected_calibration_error']
#     print(f"Computed ECE on {pred_labels.shape[0]} examples")

#     return ece, (logits, q_pids, top_confidences, pred_labels)

# def log_results(mAP, cmc, tag='Avg', use_wandb=True):
#     ranks=[1, 5, 10, 20]
#     print(f"** {tag} Results **")
#     print("mAP: {:.1%}".format(mAP))
#     print("CMC curve")
#     for r in ranks:
#         print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
#         if use_wandb: wandb.log({f"{tag} - Rank-{r:<3}": cmc[r - 1]})
    
#     if use_wandb: wandb.log({f"{tag} - mAP": mAP})

# def eval_fn(data_loader, model, device, use_wandb=True, return_outputs=False):

#     embeddings, labels = extract_embeddings(data_loader, model, device)
#     mAP, cmc, (embeddings, q_pids, distmat) = calculate_matches(embeddings, labels)


#     log_results(mAP, cmc, use_wandb=use_wandb)

#     if return_outputs:
#         return mAP, cmc, (embeddings, q_pids, distmat)
#     else:
#         return mAP, cmc


import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import wandb

from wbia_miew_id.metrics import (
    AverageMeter, compute_distance_matrix, compute_calibration,
    eval_onevsall, topk_average_precision, precision_at_k, get_accuracy
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

def _unwrap(model):
    return model.module if isinstance(model, DDP) else model

def extract_embeddings(data_loader, model, device):
    base = _unwrap(model)
    base.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tk0:
            with autocast(enabled=torch.cuda.is_available()):
                imgs = batch["image"].to(device, non_blocking=True)
                batch_embeddings = base.extract_feat(imgs)
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            image_idx = batch["image_idx"].tolist()
            embeddings.append(pd.DataFrame(batch_embeddings, index=image_idx))
            labels.extend(batch["label"].tolist())
    embeddings = pd.concat(embeddings).values
    labels = np.array(labels)
    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"
    return embeddings, labels

def extract_logits(data_loader, model, device):
    base = _unwrap(model)
    base.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    logits, labels = [], []
    with torch.no_grad():
        for batch in tk0:
            with autocast(enabled=torch.cuda.is_available()):
                imgs = batch["image"].to(device, non_blocking=True)
                lbls = batch["label"].to(device, non_blocking=True)
                batch_logits = base.extract_logits(imgs, lbls).detach().cpu()
            image_idx = batch["image_idx"].tolist()
            logits.append(pd.DataFrame(batch_logits.numpy(), index=image_idx))
            labels.extend(batch["label"].tolist())
    logits = pd.concat(logits).values
    assert not np.isnan(logits).sum(), "NaNs found in extracted logits"
    return logits, labels

def calculate_matches(embeddings, labels, embeddings_db=None, labels_db=None,
                      dist_metric='cosine', ranks=list(range(1, 21)), mask_matrix=None):
    q_pids = np.array(labels)
    qf = torch.tensor(embeddings)
    if embeddings_db is not None:
        dbf = torch.tensor(embeddings_db)
        labels_db = np.array(labels_db)
    else:
        dbf = qf
        labels_db = np.array(labels)
        mask_matrix_diagonal = np.full((embeddings.shape[0], embeddings.shape[0]), False)
        np.fill_diagonal(mask_matrix_diagonal, True)
        mask_matrix = mask_matrix_diagonal if mask_matrix is None else np.logical_or(mask_matrix_diagonal, mask_matrix)

    distmat = compute_distance_matrix(qf, dbf, dist_metric).numpy()
    if mask_matrix is not None:
        assert mask_matrix.shape == distmat.shape, "Mask matrix must have same shape as distance matrix"
        distmat[mask_matrix] = np.inf

    print("Computing CMC and mAP ...")
    mAP = topk_average_precision(q_pids, distmat, names_db=labels_db, k=None)
    cmc, match_mat, _, _ = precision_at_k(q_pids, distmat, names_db=labels_db, ranks=ranks, return_matches=True)
    print(f"Computed rank metrics on {match_mat.shape[0]} examples")
    return mAP, cmc, (embeddings, q_pids, distmat)

def log_results(mAP, cmc, tag='Avg', use_wandb=True):
    ranks=[1, 5, 10, 20]
    print(f"** {tag} Results **")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
        if use_wandb:
            wandb.log({f"{tag} - Rank-{r:<3}": cmc[r - 1]})
    if use_wandb:
        wandb.log({f"{tag} - mAP": mAP})

def eval_fn(data_loader, model, device, use_wandb=True, return_outputs=False):
    # Local (single-process) evaluation. Rank 0 calls this in DDP.
    embeddings, labels = extract_embeddings(data_loader, model, device)
    mAP, cmc, (embeddings, q_pids, distmat) = calculate_matches(embeddings, labels)
    log_results(mAP, cmc, use_wandb=use_wandb)

    if return_outputs:
        return mAP, cmc, (embeddings, q_pids, distmat)
    else:
        return mAP, cmc



# import torch
# import torch.distributed as dist
# from tqdm.auto import tqdm
# import pandas as pd
# import numpy as np
# import wandb

# from wbia_miew_id.metrics import AverageMeter, compute_distance_matrix, compute_calibration, eval_onevsall, topk_average_precision, precision_at_k, get_accuracy
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast

# def _unwrap(model):
#     return model.module if isinstance(model, DDP) else model

# def extract_embeddings(data_loader, model, device):
#     base = _unwrap(model)
#     base.eval()
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     embeddings, labels = [], []
#     with torch.no_grad():
#         for batch in tk0:
#             with autocast(enabled=torch.cuda.is_available()):
#                 batch_embeddings = base.extract_feat(batch["image"].to(device, non_blocking=True))
#             batch_embeddings = batch_embeddings.detach().cpu().numpy()
#             image_idx = batch["image_idx"].tolist()
#             embeddings.append(pd.DataFrame(batch_embeddings, index=image_idx))
#             labels.extend(batch['label'].tolist())
#     embeddings = pd.concat(embeddings).values
#     assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"
#     return embeddings, np.array(labels) # Return labels as numpy array

# def extract_logits(data_loader, model, device):
#     base = _unwrap(model)
#     base.eval()
#     tk0 = tqdm(data_loader, total=len(data_loader))
#     logits, labels = [], []
#     with torch.no_grad():
#         for batch in tk0:
#             with autocast(enabled=torch.cuda.is_available()):
#                 batch_logits = base.extract_logits(
#                     batch["image"].to(device, non_blocking=True),
#                     batch["label"].to(device, non_blocking=True),
#                 ).detach().cpu()
#             image_idx = batch["image_idx"].tolist()
#             logits.append(pd.DataFrame(batch_logits.numpy(), index=image_idx))
#             labels.extend(batch['label'].tolist())
#     logits = pd.concat(logits).values
#     assert not np.isnan(logits).sum(), "NaNs found in extracted logits"
#     return logits, labels

# def calculate_matches(embeddings, labels, embeddings_db=None, labels_db=None, dist_metric='cosine', ranks=list(range(1, 21)), mask_matrix=None):
#     q_pids = np.array(labels)
#     qf = torch.Tensor(embeddings)
#     if embeddings_db is not None:
#         dbf = torch.Tensor(embeddings_db)
#         labels_db = np.array(labels_db)
#     else:
#         dbf = qf
#         labels_db = np.array(labels)
#         mask_matrix_diagonal = np.full((embeddings.shape[0], embeddings.shape[0]), False)
#         np.fill_diagonal(mask_matrix_diagonal, True)
#         if mask_matrix is not None:
#             mask_matrix = np.logical_or(mask_matrix_diagonal, mask_matrix)
#         else:
#             mask_matrix = mask_matrix_diagonal
#     distmat = compute_distance_matrix(qf, dbf, dist_metric).numpy()
#     if mask_matrix is not None:
#         assert mask_matrix.shape == distmat.shape, "Mask matrix must have same shape as distance matrix"
#         distmat[mask_matrix] = np.inf
#     print("Computing CMC and mAP ...")
#     mAP = topk_average_precision(q_pids, distmat, names_db=labels_db, k=None)
#     cmc, match_mat, _, _ = precision_at_k(q_pids, distmat, names_db=labels_db, ranks=ranks, return_matches=True)
#     print(f"Computed rank metrics on {match_mat.shape[0]} examples")
#     return mAP, cmc, (embeddings, q_pids, distmat)

# def log_results(mAP, cmc, tag='Avg', use_wandb=True):
#     ranks=[1, 5, 10, 20]
#     print(f"** {tag} Results **")
#     print("mAP: {:.1%}".format(mAP))
#     print("CMC curve")
#     for r in ranks:
#         print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
#         if use_wandb: wandb.log({f"{tag} - Rank-{r:<3}": cmc[r - 1]})
#     if use_wandb: wandb.log({f"{tag} - mAP": mAP})

# def eval_fn(data_loader, model, device, use_wandb=True, return_outputs=False):
    
#     # Step 1: Each process extracts embeddings for its slice of data
#     embeddings, labels = extract_embeddings(data_loader, model, device)

#     # Step 2: Gather embeddings and labels from all processes if in distributed mode
#     if dist.is_initialized():
#         world_size = dist.get_world_size()
        
#         # Create placeholder lists on each process
#         gathered_embeddings = [None] * world_size
#         gathered_labels = [None] * world_size
        
#         # Gather the Python objects from all processes
#         dist.all_gather_object(gathered_embeddings, embeddings)
#         dist.all_gather_object(gathered_labels, labels)

#         # On the main process (rank 0), concatenate the gathered results
#         if dist.get_rank() == 0:
#             embeddings = np.concatenate(gathered_embeddings)
#             labels = np.concatenate(gathered_labels)

#     # Step 3: Only the main process computes and logs metrics on the full dataset
#     mAP, cmc, distmat, q_pids = None, None, None, None
#     if not dist.is_initialized() or dist.get_rank() == 0:
#         print(f"Calculating metrics on full validation set of {len(labels)} samples...")
#         mAP, cmc, (embeddings, q_pids, distmat) = calculate_matches(embeddings, labels)
        
#         # Only log to wandb from the main process
#         log_results(mAP, cmc, use_wandb=(use_wandb and dist.get_rank() == 0))
    
#     # Step 4: Synchronize all processes to ensure consistent return values
#     if dist.is_initialized():
#         dist.barrier() # Wait for rank 0 to finish computation
#         # Broadcast the final results from rank 0 to all other processes
#         results_list = [mAP] + cmc
#         results_tensor = torch.tensor(results_list, device=device)
#         dist.broadcast(results_tensor, src=0)
#         mAP = results_tensor[0].item()
#         cmc = results_tensor[1:].tolist()

#     if return_outputs:
#         # Note: Only rank 0 will have the full distance matrix
#         return mAP, cmc, (embeddings, q_pids, distmat)
#     else:
#         return mAP, cmc