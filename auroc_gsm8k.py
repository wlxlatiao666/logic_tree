import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from answer_parser import parse_gsm8k_answer, parse_model_answer

# 主计算流程
def main():
    with open('./logic_result_topptopk_100.json') as f:
        data = json.load(f)
    
    with open('./results_gsm8k/generated_answers_greedy.json') as f:
        data_greedy = json.load(f)

    with open('./results_gsm8k/generated_answers_topk_topp.json') as f:
        data_topktopp = json.load(f)

    y_true = defaultdict(list)
    y_score = defaultdict(list)
    diversity_list = []

    for item in data:
        labels = item["label"]
        for index in labels:
            y_true[index].append(labels[index])
        uncertainties = item["uncertainty"]
        for index in uncertainties:
            y_score[index].append(-uncertainties[index])
        if item["diversity"]!=0.0:
            diversity_list.append(item["diversity"])
    
    print(f"diversity mean: {np.mean(diversity_list):.4f}")
    
    for item in data_greedy:
        is_correct = int(parse_gsm8k_answer(item["gt_answer"]) == parse_model_answer(item["greedy_answer"]))
        y_true["greedy"].append(is_correct)
    for item in data_topktopp:
        is_correct = int(parse_gsm8k_answer(item["gt_answer"]) == parse_model_answer(item["greedy_answer"]))
        y_true["topktopp"].append(is_correct)

    for label in y_true:
        print(f"accuracy for label {label}: {sum(y_true[label]) / len(y_true[label])}")
        for uncertainty in y_score:
            auroc = roc_auc_score(y_true[label], y_score[uncertainty])
            print(f"{label} + {uncertainty} AUROC: {auroc:.4f}")
        print("\n\n")
    # 计算AUROC（数值越大表示不确定性越高的错误预测）
    # print(entropy)
    # # print(f"y_true: {y_true}")
    # # print(f"y_score: {y_score}")
    # auroc = roc_auc_score(y_true, y_score)
    # print(f"accuracy: {sum(y_true)/len(y_true):.4f}")
    # print(f"AUROC: {auroc:.4f}")


if __name__ == "__main__":
    # main("avg_logprob")
    main()
