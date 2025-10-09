import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from answer_parser import parse_gsm8k_answer, parse_model_answer
import matplotlib.pyplot as plt

# 加载数据
with open('./results_gsm8k/version2_result_topptopk.json') as f:
    data = json.load(f)

data_uncertainty = []
data_label = []
for item in data:
    data_uncertainty.append(item["uncertainty"])
    data_label.append(item["label"])

df1 = pd.DataFrame(data_uncertainty)
df2 = pd.DataFrame(data_label)

# 生成正确性标签 (1: 答案正确，0: 错误)
df2['correct'] = df2.apply(lambda x: 1 if x['answer_vote_weighted_threshold'] else 0, axis=1)

# 归一化特征
features = df1[['weighted_leaf_avg_entropy', 'predictive_entropy_weighted']].values
features = (features - features.mean(axis=0)) / features.std(axis=0)

# 参数搜索空间
alpha_values = np.linspace(0, 1, 101)
beta_values = 1 - alpha_values

best_auc = 0
best_params = {}
results = []

for alpha, beta in zip(alpha_values, beta_values):
    # 计算加权分数
    combined_score = alpha * features[:,0] + beta * features[:,1]
    
    # 计算AUROC（注意标签取反，不确定性越高表示错误可能性越大）
    auc = roc_auc_score(df2['correct'], -combined_score)
    
    results.append({'alpha': alpha, 'beta': beta, 'auc': auc})
    
    if auc > best_auc:
        best_auc = auc
        best_params = {'alpha': alpha, 'beta': beta}

# 保存结果
# pd.DataFrame(results).to_csv('weighted_combination_results.csv', index=False)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, [r['auc'] for r in results], marker='o')
plt.xlabel('Alpha (logic_uncertainty weight)')
plt.ylabel('AUROC')
plt.title('Weighted Combination Performance')
plt.grid(True)
plt.savefig('parameter_search.png')
plt.close()

print(f"最佳参数组合：{best_params}，最高AUROC：{best_auc:.4f}")
