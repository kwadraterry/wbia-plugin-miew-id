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
