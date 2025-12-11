import json
import argparse
import logging
import math
from typing import List
from sentence_transformers import SentenceTransformer, util
from utils import parse_model_answer, get_gt_answer, match_answer

# embedder = SentenceTransformer('/mnt/public/gpfs-jd/code/weilongxuan/all-mpnet-base-v2')

# def calculate_diversity(texts: List[str]):
#     if len(texts) < 2:
#         return 0.0

#     distances = []
#     embeddings = []
#     for text in texts:
#         embeddings.append(embedder.encode(text, convert_to_tensor=True))

#     for i in range(len(texts)):
#         current_distances = []
#         for j in range(len(texts)):
#             if i != j:
#                 cos_sim = util.cos_sim(embeddings[i], embeddings[j])
#                 current_distances.append(1 - cos_sim)
#         distances.append(sum(current_distances) / len(current_distances))
    
#     diversity = sum(distances)
#     return diversity.item()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True, help="Include .json suffix")
    args = parser.parse_args()

    dataset = args.dataset
    file_name = args.file_name
    # dataset_path = f"./data/{dataset}/test.json"
    input_path = f'./results/{args.model}/{dataset}/{file_name}'
    file_name = file_name.replace(".json", "")
    output_path = f'./results/{args.model}/{dataset}/{file_name}_metrics.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    results = []
    sce = []
    for i, item in enumerate(data):
        original_data = item['original_data']
        gt_answer = get_gt_answer(dataset, original_data)

        probs = item["probs"]
        entropies = item["entropies"]
        texts = item["texts"]

        uncertainties = {}
        labels = {}

        # calculate uncertainty metrics
        uncertainties["weighted_leaf_avg_entropy"] = sum(p * e for p, e in zip(probs, entropies))
        max_prob_index = probs.index(max(probs))
        uncertainties["max_leaf_avg_entropy"] = entropies[max_prob_index]
        uncertainties["avg_leaf_avg_entropy"] = sum(e for e in entropies) / len(entropies)
        uncertainties["complexity"] = item["complexity"]

        answer_buckets = {}
        for j, text in enumerate(texts):
            parsed_answer = parse_model_answer(text)
            if parsed_answer:
                try: 
                    answer_buckets[parsed_answer] = answer_buckets.get(parsed_answer, 0.0) + probs[j]
                except:
                    continue
        total = sum(answer_buckets.values())
        pe = 0.0
        if total > 0:
            probabilities = [p / total for p in answer_buckets.values()]
            pe = -sum(p * math.log(p) for p in probabilities if p > 0)
        uncertainties["predictive_entropy_weighted"] = float(pe)

        probs_entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0)
        uncertainties["probs_entropy"] = float(probs_entropy)
        
        # calculate labels
        final_answer_vote_weighted = max(answer_buckets.items(), key=lambda x: x[1]) if answer_buckets else ""
        final_answer_vote_weighted_nothreshold = final_answer_vote_weighted[0]
        labels["answer_vote_weighted_nothreshold"] = int(match_answer(gt_answer, final_answer_vote_weighted_nothreshold, dataset)) if gt_answer else 0

        final_answer_vote_weighted_threshold = final_answer_vote_weighted[0] if final_answer_vote_weighted[1] > 0.5 else ""
        labels["answer_vote_weighted_threshold"] = int(match_answer(gt_answer, final_answer_vote_weighted_threshold, dataset)) if gt_answer else 0

        min_entropy_index = entropies.index(min(entropies))
        final_answer_min_entropy = parse_model_answer(texts[min_entropy_index])
        labels["answer_min_entropy"] = int(match_answer(gt_answer, final_answer_min_entropy, dataset)) if gt_answer else 0
            
        passk = False
        for answer in answer_buckets:
            if match_answer(gt_answer, answer, dataset):
                passk = True
                break
        labels["pass@k"] = int(passk)

        # diversity = calculate_diversity(texts)
        # sce.append(diversity * 10000 / item["num_new_tokens"])
        results.append({
            "gt_answer": gt_answer,
            "num_new_tokens": item["num_new_tokens"],
            "label": labels,
            "uncertainty": uncertainties
        })
        logger.info(f"Processed {i+1} items")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"\nResults saved to {output_path}")