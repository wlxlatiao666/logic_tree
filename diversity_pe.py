import json
import argparse
import math
from typing import List
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('/mnt/public/gpfs-jd/code/weilongxuan/all-mpnet-base-v2')

def calculate_diversity(texts: List[str]):
    if len(texts) < 2:
        return 0.0

    distances = []
    embeddings = []
    for text in texts:
        embeddings.append(embedder.encode(text, convert_to_tensor=True))

    for i in range(len(texts)):
        current_distances = []
        for j in range(len(texts)):
            if i != j:
                cos_sim = util.cos_sim(embeddings[i], embeddings[j])
                current_distances.append(1 - cos_sim)
        distances.append(sum(current_distances) / len(current_distances))
    
    diversity = sum(distances)
    return diversity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True, help="Include .json suffix")
    args = parser.parse_args()

    dataset = args.dataset
    file_name = args.file_name
    # dataset_path = f"./data/{dataset}/test.json"
    input_path = f'./results/{dataset}/{file_name}'
    file_name = file_name.replace(".json", "")
    output_path = f'./results/{dataset}/{file_name}_metrics.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    results = []
    sce = []
    for i, item in enumerate(data):
        texts = item["sampled_answers"]
        diversity = calculate_diversity(texts)
        sce.append(diversity * 10000 / sum(item["num_tokens"]))
        results.append({
            "diversity": diversity,
            "num_tokens": item["num_tokens"],
        })
        print(f"Processed {i+1} items")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_path}")
    print(f"sce list: {sce}")
    print(f"sce mean: {sum(sce)/len(sce)}")
