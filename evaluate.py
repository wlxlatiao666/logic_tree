import json
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import numpy as np
import sys

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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    parser.add_argument("--lt_file", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    sc_file_path = f"./results/{args.model}/{dataset}/{args.sc_file}"
    lt_file_path = f"./results/{args.model}/{dataset}/{args.lt_file}"
    sys.stdout = open(f'./results/{args.model}/{dataset}/output_all_relevance.txt', 'w', encoding='utf-8')
    with open(lt_file_path) as f:
        data_lt = json.load(f)
    with open(f'./results/{args.model}/{dataset}/generated_answers_greedy.json') as f:
        data_greedy = json.load(f)
    # with open(f'./results/{dataset}/generated_answers_topk_topp.json') as f:
    #     topk_topp_data = json.load(f)
    with open(sc_file_path) as f:
        data_sc = json.load(f)

    y_true = defaultdict(list)
    y_score = defaultdict(list)
    pe_true_nothreshold = []
    pe_true_threshold = []
    pe_passk = []
    greedy_score = []
    # topp_score = []
    pe_score = []

    for item, pe_item, greedy_item in zip(data_lt, data_sc, data_greedy):
        y_true['greedy'].append(greedy_item['label'])
        greedy_score.append(-greedy_item['avg_logprob'])
        # y_true['topk_topp'].append(topp_item['label'])
        # topp_score.append(-topp_item['avg_logprob'])
        pe_true_nothreshold.append(pe_item['label_nothreshold'])
        pe_true_threshold.append(pe_item['label_threshold'])
        pe_passk.append(pe_item['passk'])
        pe_score.append(-pe_item['predictive_entropy'])
        labels = item["label"]
        for index in labels:
            y_true[index].append(labels[index])
        uncertainties = item["uncertainty"]
        for index in uncertainties:
            y_score[index].append(-uncertainties[index])

    print("Greedy Results:")
    auroc = roc_auc_score(y_true['greedy'], greedy_score)
    print(f"Greedy AUROC: {auroc:.4f}")
    auarc = compute_auarc(greedy_score, y_true['greedy'])
    print(f"Greedy AUARC: {auarc:.4f}")
    ece = compute_ece(greedy_score, y_true['greedy'])
    print(f"Greedy ECE: {ece:.4f}")

    # print("\n\nTop-K/Top-P Results:")
    # auroc = roc_auc_score(y_true['topk_topp'], topp_score)  
    # print(f"Top-K/Top-P AUROC: {auroc:.4f}")
    # auarc = compute_auarc(topp_score, y_true['topk_topp'])
    # print(f"Top-K/Top-P AUARC: {auarc:.4f}")
    # ece = compute_ece(topp_score, y_true['topk_topp'])
    # print(f"Top-K/Top-P ECE: {ece:.4f}")   

    print("\n\nPredictive Entropy Results:")
    print("accuracy(no threshold)):", sum(pe_true_nothreshold) / len(pe_true_nothreshold))
    print("accuracy(threshold)):", sum(pe_true_threshold) / len(pe_true_threshold))
    print("pass@k:", sum(pe_passk) / len(pe_passk))
    auroc = roc_auc_score(y_true['greedy'], pe_score)
    print(f"Predictive Entropy AUROC(greedy label): {auroc:.4f}")
    auroc = roc_auc_score(pe_true_nothreshold, pe_score)
    print(f"Predictive Entropy AUROC(pe label no threshold): {auroc:.4f}")
    auroc = roc_auc_score(pe_true_threshold, pe_score)
    print(f"Predictive Entropy AUROC(pe label threshold): {auroc:.4f}")
    auarc = compute_auarc(pe_score, y_true['greedy'])
    print(f"Predictive Entropy AUARC(greedy label): {auarc:.4f}")
    auarc = compute_auarc(pe_score, pe_true_nothreshold)
    print(f"Predictive Entropy AUARC(pe label no threshold): {auarc:.4f}")
    auarc = compute_auarc(pe_score, pe_true_threshold)
    print(f"Predictive Entropy AUARC(pe label threshold): {auarc:.4f}")
    ece = compute_ece(pe_score, y_true['greedy'])
    print(f"Predictive Entropy ECE(greedy label): {ece:.4f}")
    ece = compute_ece(pe_score, pe_true_nothreshold)
    print(f"Predictive Entropy ECE(pe label no threshold): {ece:.4f}")
    ece = compute_ece(pe_score, pe_true_threshold)
    print(f"Predictive Entropy ECE(pe label threshold): {ece:.4f}")


    print("\n\nLogic Tree Results:")
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
