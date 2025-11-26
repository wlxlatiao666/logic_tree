import torch
import os
import json
import argparse
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode
from utils import generate_usr_prompt
from threshold import get_threshold

logger = logging.getLogger(__name__)

model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B"
}

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--num_leaves", type=int, default=20)
    parser.add_argument("--tau", type=int, default=80)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # load model
    model_name = args.model
    model_dir = model_to_dir[model_name]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        # 可选：添加日志记录选择的设备
        logger.info(f"使用GPU设备: {args.device}")
    else:
        device = "cpu"
        logger.info("CUDA不可用，使用CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load system prompt
    dataset = args.dataset
    test_size = args.test_size
    num_leaves = args.num_leaves
    tau = args.tau
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    thr = get_threshold(tokenizer, model, device, dataset, tau=tau)
    logger.info(f"Threshold for {dataset}: {thr}")

    # generate
    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, device, prompt=prompt, sample=True, tau=thr, branches_m=3, max_leaves=num_leaves)

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
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_all_leaves{num_leaves}_threshold{tau}.json"
    else:
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_{test_size}_leaves{num_leaves}_threshold{tau}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)