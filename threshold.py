import argparse
import json
import math
from collections import deque
from typing import List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import generate_usr_prompt

# reuse WINDOW_SIZE constant from logic_tree_decode_new if available
try:
    from logic_tree_decode_new import WINDOW_SIZE
except Exception:
    WINDOW_SIZE = 10


def get_threshold(dataset: str, model_name: str = None, max_items: int = 10, max_gen_tokens: int = 1024) -> float:
    """Compute entropy threshold from first `max_items` entries in dataset.

    Procedure:
    - load ./data/{dataset}/test.json and take first `max_items` items
    - for each item, build prompt using `utils.generate_usr_prompt` and system prompt from `sys_prompt.json`
    - autoregressively generate tokens greedily; for each generated token compute token entropy = -log(p_selected)
    - maintain sliding window of size WINDOW_SIZE over token entropies; whenever window is full compute average and append to list
    - stop generation on EOS or after `max_gen_tokens` tokens
    - return 95th percentile of collected average entropies (or NaN if none)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name fallback: try to read from env or default to local Qwen path used in generate.py
    if model_name is None:
        model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)[:max_items]

    with open("./sys_prompt.json", 'r') as f:
        sys_prompts = json.load(f)
    if dataset not in sys_prompts:
        raise ValueError(f"dataset {dataset} not found in sys_prompt.json")
    sys_prompt = sys_prompts[dataset]

    entropy_averages: List[float] = []

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
        # sliding window for entropies
        window = deque(maxlen=WINDOW_SIZE)

        for _ in range(max_gen_tokens):
            with torch.no_grad():
                out = model(input_ids=cur_ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            past = out.past_key_values

            probs = softmax(logits, dim=-1)
            next_id = int(torch.argmax(logits).item())
            next_prob = float(probs[next_id].item())

            token_entropy = -math.log(max(next_prob, 1e-12))
            window.append(token_entropy)
            if len(window) == WINDOW_SIZE:
                avg_H = sum(window) / len(window)
                entropy_averages.append(avg_H)

            # prepare next input
            if next_id == tokenizer.eos_token_id:
                break
            cur_ids = torch.tensor([[next_id]], device=device, dtype=torch.int)
            # attention_mask = None

    if len(entropy_averages) == 0:
        return float('nan')
    arr = np.array(entropy_averages)
    threshold = float(np.percentile(arr, 95))
    return threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    dataset = args.dataset
    thr = get_threshold(dataset, model_name=args.model_name)
    print(f"Threshold (95th percentile): {thr}")