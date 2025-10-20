import torch
import json
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from logic_tree_decode import logic_branch_decode, compute_avg_branching_factor
from threshold import get_threshold
from utils import generate_usr_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()

    # load model
    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    embedder = SentenceTransformer('/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/all-mpnet-base-v2')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load system prompt
    dataset = args.dataset
    test_size = args.test_size
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    # compute threshold
    thr = get_threshold(tokenizer, model, dataset)
    print(f"Threshold (95th percentile): {thr}")

    # generate
    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        root, leaves = logic_branch_decode(tokenizer, model, embedder, prompt=prompt, tau=thr)
        avg_b = compute_avg_branching_factor(root)
        avg_d = sum(leaf.depth for leaf in leaves) / len(leaves)

        texts = [tokenizer.decode(leaf.ids, clean_up_tokenization_spaces=False) for leaf in leaves]
        results.append({
            "original_data": item,
            "num_leaves": len(leaves),
            "avg_branching_factor": avg_b,
            "avg_depth": avg_d,
            "texts": texts
        })
        print(f"Processed {i+1} items")
    end_time = time.time()
    duration = end_time - start_time
    print(f"generate {len(results)} results in {duration:.2f} seconds({duration/60:.2f} minutes)")

    if test_size == -1:
        output_path = f"./results/{dataset}/logic_tree_results_all.json"
    else:
        output_path = f"./results/{dataset}/logic_tree_results_{test_size}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)