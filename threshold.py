import argparse
import json
import math
import logging
from collections import deque
from typing import List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import generate_usr_prompt

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_threshold(tokenizer, model, dataset: str, max_items: int = 10, max_gen_tokens: int = 1024) -> float:
    model.eval()

    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)[:max_items]

    with open("./sys_prompt.json", 'r') as f:
        sys_prompts = json.load(f)
    if dataset not in sys_prompts:
        raise ValueError(f"dataset {dataset} not found in sys_prompt.json")
    sys_prompt = sys_prompts[dataset]

    entropies: List[float] = []

    softmax = torch.nn.functional.softmax

    for item in data:
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        # attention_mask = inputs.get("attention_mask", None)

        past = None
        cur_ids = input_ids

        for _ in range(max_gen_tokens):
            with torch.no_grad():
                out = model(input_ids=cur_ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            past = out.past_key_values

            probs = softmax(logits, dim=-1)
            next_id = int(torch.argmax(logits).item())
            next_prob = float(probs[next_id].item())

            token_entropy = -math.log(max(next_prob, 1e-12))
            entropies.append(token_entropy)

            # prepare next input
            if next_id == tokenizer.eos_token_id:
                break
            cur_ids = torch.tensor([[next_id]], device=device, dtype=torch.int)
            # attention_mask = None

    if len(entropies) == 0:
        return float('nan')
    arr = np.array(entropies)
    threshold = float(np.percentile(arr, 99.5))
    # min_v = float(np.min(arr))
    # max_v = float(np.max(arr))
    # threshold = min_v + 0.95 * (max_v - min_v)
    return threshold


if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    thr = get_threshold(tokenizer, model, dataset)
    logger.info(f"Threshold: {thr}")
