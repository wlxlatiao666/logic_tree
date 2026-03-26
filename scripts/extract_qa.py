import json

input_file = "/Users/weilongxuan/codes/logic_tree/data/olympiadbench_zh/train_all.json"
output_file = "/Users/weilongxuan/codes/logic_tree/data/olympiadbench_zh/test.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

qa_pairs = []
for item in data:
    qa_pairs.append({
        "question": item["question"],
        "answer": item["final_answer"]
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
