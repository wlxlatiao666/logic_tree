import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from answer_parser import parse_gsm8k_answer, parse_model_answer

# 主计算流程
def main():
    with open('./logic_output_gsm8k_noimportance.json') as f:
        data = json.load(f)

    y_true = []
    y_score = []

    for item in data:
        # 计算预测熵
        # ent = item[entropy]
        labels = item["labels"]
        uncertainties = item["uncertainties"]

        y_true.extend(labels)
        y_score.extend([-u for u in uncertainties])

    # 计算AUROC（数值越大表示不确定性越高的错误预测）
    # print(entropy)
    # print(f"y_true: {y_true}")
    # print(f"y_score: {y_score}")
    auroc = roc_auc_score(y_true, y_score)
    print(f"accuracy: {sum(y_true)/len(y_true):.4f}")
    print(f"AUROC: {auroc:.4f}")

    # 保存中间结果
    with open('auroc_results.json', 'w') as f:
        json.dump({
            'auroc': auroc,
            'predictive_entropy': y_score,
            'labels': y_true
        }, f, indent=2)

if __name__ == "__main__":
    # main("avg_logprob")
    main()
