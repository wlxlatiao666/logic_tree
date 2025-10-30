import torch
import json
import argparse
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode
from utils import generate_usr_prompt
from threshold import get_threshold

if __name__ == "__main__":
    logging.basicConfig(
        filename='./logs/app.log',  # 使用绝对路径
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--times", type=int, required=True)
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
    times = args.times
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)
    index = 0
    item = data[index]
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    thr = get_threshold(tokenizer, model, dataset)
    logger.info(f"Threshold (98th percentile): {thr}")

    # generate
    results = []
    start_time = time.time()
    for _ in range(times):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, prompt=prompt, sample=True, tau=thr, branches_m=3)

        complexity = sum(leaf.prob * leaf.depth for leaf in leaves)
        probs = [leaf.prob for leaf in leaves]
        texts = [leaf.text for leaf in leaves]
        entropies = [-leaf.cum_logprob / leaf.length if leaf.length > 0 else 0.0 for leaf in leaves]
        results.append({
            "original_data": item,
            "num_new_tokens": new_tokens_cnt,
            "num_leaves": len(leaves),
            "complexity": complexity,
            "probs": probs,
            "entropies": entropies,
            "texts": texts
        })
    end_time = time.time()
    duration = end_time - start_time

    output_path = f"./results/{dataset}/logic_tree_results_{times}repeats.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)