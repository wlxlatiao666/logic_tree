import json

sc_path = "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/logic_tree/results/reclor/generated_answers_5samples.json"
lt_path = "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/logic_tree/results/reclor/logic_tree_results_100_metrics.json"

with open(sc_path) as f:
    sc_data = json.load(f)
with open(lt_path) as f:
    lt_data = json.load(f)

sc_pe_list = []
sc_entropy_list = []
lt_pe_list = []
lt_entropy_list = []
for item_sc, item_lt in zip(sc_data, lt_data):
    sc_pe_list.append(item_sc["predictive_entropy"])
    lt_pe_list.append(item_lt["uncertainty"]["predictive_entropy_weighted"])
    # sc_entropy_list.extend(item_sc["sampled_entropies"])
    # lt_entropy_list.extend(item_lt["entropies"])

print("Avg sc pe: ", sum(sc_pe_list) / len(sc_pe_list))
print("Avg lt pe: ", sum(lt_pe_list) / len(lt_pe_list))
# print("Avg sc entropy:", sum(sc_entropy_list) / len(sc_entropy_list))
# print("Avg lt entropy:", sum(lt_entropy_list) / len(lt_entropy_list))