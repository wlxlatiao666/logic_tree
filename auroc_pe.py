import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from typing import List
from answer_parser import parse_gsm8k_answer, parse_model_answer
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('/mnt/public/gpfs-jd/code/weilongxuan/all-mpnet-base-v2')
# 计算预测熵
def calculate_entropy(answers):
    answer_counts = {}
    for ans in answers:
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    probs = np.array(list(answer_counts.values())) / len(answers)
    return entropy(probs)

def calculate_diversity(texts: List[str]):
    if len(texts) < 2:
        return 0.0

    distances = []
    embeddings = []
    for text in texts:
        embeddings.append(embedder.encode(text, convert_to_tensor=True))

    for i in range(len(texts)):
        current_distances = []
        for j in range(len(texts)):
            if i != j:
                cos_sim = util.cos_sim(embeddings[i], embeddings[j])
                current_distances.append(1 - cos_sim)
        distances.append(sum(current_distances) / len(current_distances))
    
    diversity = sum(distances)
    return diversity.item()

# 主计算流程
def main():
    with open('./results_gsm8k/generated_answers_5samples1.json') as f:
        data = json.load(f)

    y_true = []
    pre_ents = []
    token_ents = []

    y_score = []
    diversity_list = []

    for item in data[:100]:
        # 计算正确性标签
        gt_ans = parse_gsm8k_answer(item["gt_answer"])
        ml_ans = parse_model_answer(item["greedy_answer"])
        is_correct = int(gt_ans == ml_ans)
        
        # 计算预测熵
        sampled_answers = [parse_model_answer(a) for a in item["sampled_answers"]]
        pre_ent = calculate_entropy(sampled_answers)
        token_ent = sum(item["sampled_entropies"]) / len(item["sampled_entropies"])
        
        y_true.append(is_correct)
        pre_ents.append(pre_ent)
        token_ents.append(token_ent)

        diversity = calculate_diversity(item["sampled_answers"])
        diversity_list.append(diversity)
        print(f"diversity: {diversity:.3f}")

    best_alpha = 0
    best_auroc = 0
    alpha_values = np.linspace(0, 1, 101)

    for alpha in alpha_values:
        y_score = [-alpha * pre_ent - (1 - alpha) * token_ent for pre_ent, token_ent in zip(pre_ents, token_ents)]
        auroc = roc_auc_score(y_true, y_score)
        print(f"alpha: {alpha:.4f}, auroc: {auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = alpha

    # 计算AUROC（数值越大表示不确定性越高的错误预测）
    # print(f"y_true: {y_true}")
    print(f"best_alpha: {best_alpha:.4f}")
    print(f"best_auroc: {best_auroc:.4f}")

    print(f"average diversity: {sum(diversity_list) / len(diversity_list):.3f}")

if __name__ == "__main__":
    main()