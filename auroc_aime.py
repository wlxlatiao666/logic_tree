import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from answer_parser import parse_model_answer

# 主计算流程
def main(entropy: str):
    with open('uncertainty_results_7B_aime.json') as f:
        data = json.load(f)

    y_true = []
    y_score = []

    for item in data:
        # 计算正确性标签
        gt_ans = str(item["gt_answer"]).strip()
        ml_ans = parse_model_answer(item["greedy_answer"])
        is_correct = int(gt_ans == ml_ans)
        
        # 计算预测熵
        ent = item[entropy]

        y_true.append(is_correct)
        y_score.append(-ent)

    # 计算AUROC（数值越大表示不确定性越高的错误预测）
    print(entropy)
    print(f"y_true: {y_true}")
    print(f"y_score: {y_score}")
    auroc = roc_auc_score(y_true, y_score)
    print(f"AUROC: {auroc:.4f}")

    # 保存中间结果
    with open('auroc_results.json', 'w') as f:
        json.dump({
            'auroc': auroc,
            'predictive_entropy': y_score,
            'labels': y_true
        }, f, indent=2)

if __name__ == "__main__":
    main("logic_uncertainty")
    main("single_uncertainty")
    main("complexity")
