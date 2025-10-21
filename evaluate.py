import json
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import numpy as np

def compute_auarc(scores, labels, descending=True):
    """
    Area Under Accuracy-Coverage Curve (AUARC).
    scores: list/array, confidence scores (higher=more confident)
    labels: list/array, binary ground truth (1=correct, 0=wrong)
    descending: sort scores descending (default True)
    Returns: auarc (float)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    idx = np.argsort(scores)[::-1] if descending else np.argsort(scores)
    sorted_labels = labels[idx]
    N = len(labels)
    cum_correct = np.cumsum(sorted_labels)
    ks = np.arange(1, N + 1)
    accs = cum_correct / ks
    coverages = ks / N
    auarc = float(np.trapz(accs, coverages))
    return auarc

def compute_ece(confidences, labels, n_bins=10, normalize=True):
    """
    Expected Calibration Error (ECE).
    confidences: list/array, confidence scores (can be any range, will be normalized to [0,1] if normalize=True)
    labels: list/array, binary ground truth (1=correct, 0=wrong)
    n_bins: number of bins (default 10)
    Returns: ece (float)
    """
    conf = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if normalize:
        mn, mx = conf.min(), conf.max()
        conf = (conf - mn) / (mx - mn) if mx > mn else np.full_like(conf, 0.5)
    N = len(conf)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        prop = mask.sum() / N
        avg_conf = conf[mask].mean()
        acc = labels[mask].mean()
        ece += prop * abs(avg_conf - acc)
    return float(ece)

# 主计算流程
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True, help="Include .json suffix")
    args = parser.parse_args()

    dataset = args.dataset
    file_name = args.file_name
    with open(f'./results/{dataset}/{file_name}') as f:
        data = json.load(f)
    with open(f'./results/{dataset}/generated_answers_greedy.json') as f:
        greedy_data = json.load(f)
    with open(f'./results/{dataset}/generated_answers_topk_topp.json') as f:
        topk_topp_data = json.load(f)

    y_true = defaultdict(list)
    y_score = defaultdict(list)

    for item, greedy_item, topp_item in zip(data, greedy_data, topk_topp_data):
        y_true['greedy'].append(greedy_item['label'])
        y_true['topk_topp'].append(topp_item['label'])
        labels = item["label"]
        for index in labels:
            y_true[index].append(labels[index])
        uncertainties = item["uncertainty"]
        for index in uncertainties:
            y_score[index].append(-uncertainties[index])

    for label in y_true:
        print(f"accuracy for label {label}: {sum(y_true[label]) / len(y_true[label])}")
        for uncertainty in y_score:
            auroc = roc_auc_score(y_true[label], y_score[uncertainty])
            print(f"{label} + {uncertainty} AUROC: {auroc:.4f}")
            auarc = compute_auarc(y_score[uncertainty], y_true[label])
            print(f"{label} + {uncertainty} AUARC: {auarc:.4f}")
            ece = compute_ece(y_score[uncertainty], y_true[label])
            print(f"{label} + {uncertainty} ECE: {ece:.4f}")
            print("\n")
        print("\n\n")
