import json
from collections import defaultdict
result = defaultdict(list)

path = "./results/gsm8k/logic_tree_results_10.json"

with open(path, "r") as f:
    data = json.load(f)

for item in data:
    split_positions = item["split_positions"]
    lengths = item["lengths"]
    # print(split_positions)
    # print(lengths)
    for i, split_position in enumerate(split_positions):
        for j, position in enumerate(split_position):
            result[j+1].append(position / lengths[i])
            # print("result: ",result)

for key in result:
    avg_position = sum(result[key]) / len(result[key])
    print(f"第 {key} 次分支的平均位置: {avg_position * 100:.2f}%")
# print(result)