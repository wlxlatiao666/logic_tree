import json
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 读取JSON文件
with open('results/uncertainty_results_7B_100_gsm8k.json') as f:
    data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(data)[['width', 'complexity']]

# 计算皮尔逊相关系数
corr, p_value = pearsonr(df['width'], df['complexity'])
print(f'Pearson相关系数: {corr:.3f} (p={p_value:.3f})')

# 绘制散点图
plt.figure(figsize=(10,6))
plt.scatter(df['width'], df['complexity'], alpha=0.7)
plt.title('Width vs Complexity Correlation')
plt.xlabel('Tree Width')
plt.ylabel('Complexity Score')
plt.grid(True)
plt.savefig('width_complexity_correlation.png')