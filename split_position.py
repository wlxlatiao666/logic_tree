import json
from collections import defaultdict
result_per = defaultdict(list)
result_num = defaultdict(list)

path = "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/logic_tree/results/reclor/logic_tree_results_100_topk_topp_nomerge.json"

with open(path, "r") as f:
    data = json.load(f)

for item in data:
    split_positions = item["split_positions"]
    lengths = item["lengths"]
    # print(split_positions)
    # print(lengths)
    for i, split_position in enumerate(split_positions):
        for j, position in enumerate(split_position):
            result_per[j+1].append(position / lengths[i])
            result_num[j+1].append(position)
            # print("result: ",result)

for key in result_per:
    avg_position = sum(result_per[key]) / len(result_per[key])
    print(f"第 {key} 次分支的平均位置: {avg_position * 100:.2f}%")
for key in result_num:
    avg_num = sum(result_num[key]) / len(result_num[key])
    print(f"第 {key} 次分支的平均token数: {avg_num:.2f}")