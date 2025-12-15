import argparse
import json
from utils import match_answer, parse_model_answer, get_gt_answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    file_path = f"./results/{args.model}/{args.dataset}/{args.sc_file}"
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    results = []
    for item in data:
        gt_answer = get_gt_answer(dataset, item["original_data"])
        answer_buckets = {}
        for text in item["sampled_answers"]:
            parsed_answer = parse_model_answer(text)
            if parsed_answer:
                answer_buckets[parsed_answer] = answer_buckets.get(parsed_answer, 0) + 1
        passk = any([match_answer(gt_answer, ans, dataset) for ans in answer_buckets.keys()])
        final_answer_vote_weighted = max(answer_buckets.items(), key=lambda x: x[1]) if answer_buckets else ""
        final_answer_vote_weighted_nothreshold = final_answer_vote_weighted[0]
        final_answer_vote_weighted_threshold = final_answer_vote_weighted[0] if final_answer_vote_weighted[1] > 0.5 * len(item["sampled_answers"]) else ""
        label_nothreshold = match_answer(gt_answer, final_answer_vote_weighted_nothreshold, dataset)
        label_threshold = match_answer(gt_answer, final_answer_vote_weighted_threshold, dataset)
        del item["label"]
        item["passk"] = int(passk)
        item["label_nothreshold"] = int(label_nothreshold)
        item["label_threshold"] = int(label_threshold)
        results.append(item)

    with open(file_path, 'w', encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
