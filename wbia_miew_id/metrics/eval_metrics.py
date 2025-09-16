import torch
import numpy as np

def precision_at_k(names, distmat, names_db=None, ranks=list(range(1, 21)), return_matches=False):
    """Computes precision at k given a distance matrix. 
    Assumes the distance matrix is square and does one-vs-all evaluation"""
    # assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"

    if names_db is None:
        names_db = names

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names_db)

    max_k = max(ranks)

    topk_idx = output.topk(max_k)[1][:, :] ### 
    topk_names = ids_tensor[topk_idx]

    match_mat = topk_names == y[:, None].expand(topk_names.shape)

    scores = []
    for k in ranks:
        match_mat_k = match_mat[:, :k]
        rank_mat = match_mat_k.any(axis=1)

        score = rank_mat.sum() / len(rank_mat)
        scores.append(score)

    if return_matches:
        return scores, match_mat, topk_idx, topk_names
    else:
        return scores

def get_accuracy(outputs, targets):
    predicted = torch.Tensor(outputs.argmax(1))
    total = targets.size(0)
    correct = predicted.eq(
        targets.detach().cpu()).sum().item()
    return correct / total

def topk_average_precision(names, distmat, names_db=None, k=None):
    """Computes top-k average precision given a distance matrix.
    Assumes the distance matrix is square and does one-vs-all evaluation"""
    # assert distmat.shape[0] == distmat.shape[1], "Distance matrix must be square"

    if names_db is None:
        names_db = names

    output = torch.Tensor(distmat[:, :]) * -1
    y = torch.Tensor(names[:]).squeeze(0)
    ids_tensor = torch.Tensor(names_db)

    if k==None: 
        k = output.shape[1] 
    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)

    _topk = output.topk(k)[1][:, :]
    topk = ids_tensor[_topk]

    match_mat = topk == y[:, None].expand(topk.shape)

    # Masked variables have values of np.inf, they should not be considered for calculation
    match_mat_mask = torch.isinf(torch.gather(output, -1, _topk))
    match_mat[match_mat_mask] = False

    rel_mat = match_mat.sum(axis=1)
    cum_mat = match_mat.cumsum(dim=1)

    ap_mat = ((cum_mat * score_array) * match_mat).sum(axis=1) / rel_mat

    return ap_mat.nan_to_num(0).mean().item()

def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins

    Taken from https://github.com/hollance/reliability-diagrams
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }
