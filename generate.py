import torch
import json
import argparse
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode
from utils import generate_usr_prompt
from threshold import get_threshold

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--sample", type=bool, default=False)
    args = parser.parse_args()

    # load model
    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load system prompt
    dataset = args.dataset
    test_size = args.test_size
    sample = args.sample
    logger.info(f"Sample: {sample}")
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    thr = get_threshold(tokenizer, model, dataset)
    logger.info(f"Threshold for {dataset}: {thr}")

    # generate
    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, prompt=prompt, sample=sample, tau=thr, branches_m=3)

        complexity = sum(leaf.prob * leaf.depth for leaf in leaves)
        probs = [leaf.prob for leaf in leaves]
        texts = [leaf.text for leaf in leaves]
        entropies = [-leaf.cum_logprob / leaf.length if leaf.length > 0 else 0.0 for leaf in leaves]
        split_positions = [leaf.split_positions for leaf in leaves]
        lengths = [leaf.length for leaf in leaves]
        results.append({
            "original_data": item,
            "num_new_tokens": new_tokens_cnt,
            "num_leaves": len(leaves),
            "complexity": complexity,
            "probs": probs,
            "entropies": entropies,
            "split_positions": split_positions,
            "lengths": lengths,
            "texts": texts
        })
        logger.info(f"Processed {i+1} items")
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"generate {len(results)} results in {duration:.2f} seconds({duration/60:.2f} minutes)")

    if test_size == -1:
        if sample:
            output_path = f"./results/{dataset}/logic_tree_results_all_topk_topp.json"
        else:
            output_path = f"./results/{dataset}/logic_tree_results_all_greedy.json"
    else:
        if sample:
            output_path = f"./results/{dataset}/logic_tree_results_{test_size}_topk_topp_nomerge.json"
        else:
            output_path = f"./results/{dataset}/logic_tree_results_{test_size}_greedy.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)