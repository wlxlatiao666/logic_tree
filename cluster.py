from utils import parse_model_answer

import json
from collections import Counter
import matplotlib.pyplot as plt
from utils import parse_model_answer

# 读取数据
with open('./results/reclor/logic_tree_results_100.json', 'r') as f:
    data = json.load(f)

answer_counts = []

for item in data:
    texts = item['texts']
    # 解析所有答案
    answers = [parse_model_answer(t) for t in texts]
    # 合并同类项
    unique_answers = set(answers)
    answer_counts.append(len(unique_answers))

print(answer_counts)  # 长度为100的列表

# 统计频率分布
freq = Counter(answer_counts)
print("频率分布：", freq)

# 画直方图
plt.figure(figsize=(8,4))
plt.hist(answer_counts, bins=range(1, max(answer_counts)+2), align='left', rwidth=0.8)
plt.xlabel('Unique Answers per Query')
plt.ylabel('Frequency')
plt.title('Distribution of Unique Answers per Query (100 samples)')
plt.xticks(range(1, max(answer_counts)+1))
plt.show()