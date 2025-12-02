import json

input_file = "/Users/weilongxuan/codes/logic_tree/data/scibench/test_all.json"
output_file = "/Users/weilongxuan/codes/logic_tree/data/scibench/test.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

qa_pairs = []
for item in data:
    qa_pairs.append({
        "question": item["problem_text"],
        "answer": item["answer_number"],
        "unit": item["unit"]
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
