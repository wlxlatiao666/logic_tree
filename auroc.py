import json
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score

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

    y_true = defaultdict(list)
    y_score = defaultdict(list)

    for item in data:
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
        print("\n\n")
