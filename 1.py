import json
from utils import parse_gsm8k_answer, parse_model_answer

with open('./results/gsm8k/generated_answers_topk_topp.json', 'r') as f:
    data = json.load(f)

results = []
for item in data:
    label = int(parse_gsm8k_answer(item['gt_answer']) == parse_model_answer(item['greedy_answer']))
    results.append(
    {
        "question": item['question'],
        "gt_answer": item['gt_answer'],
        "greedy_answer": item['greedy_answer'],
        "label": label,
        "avg_logprob": item['avg_logprob']
    }    
    )

with open('./results/gsm8k/generated_answers_topk_topp.json', 'w') as f:
    json.dump(results, f, indent=2)