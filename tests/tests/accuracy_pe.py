import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from answer_parser import parse_gsm8k_answer, parse_model_answer

# 计算预测熵

# 主计算流程
def main():
    with open('../results_gsm8k/generated_answers_5samples.json') as f:
        data = json.load(f)

    labels_greedy = []
    labels_sample = []

    for item in data:
        # 计算正确性标签
        gt_ans = parse_gsm8k_answer(item["gt_answer"])
        greedy_ans = parse_model_answer(item["greedy_answer"])
        
        # 计算预测熵
        sampled_answers = [parse_model_answer(a) for a in item["sampled_answers"]]
        
        sample_true = False
        for answer in sampled_answers:
            if gt_ans == answer:
                sample_true = True
                break
        labels_sample.append(int(sample_true))
        labels_greedy.append(int(gt_ans == greedy_ans))

    print("accuracy_sample:", sum(labels_sample) / len(labels_sample))
    print("accuracy_greedy:", sum(labels_greedy) / len(labels_greedy))

if __name__ == "__main__":
    main()