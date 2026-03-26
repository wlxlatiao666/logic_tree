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

def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.log_softmax(logits, dim=-1)

def get_waad_per_head(attn: torch.Tensor, W: int) -> float:
    """
    attn: [num_heads, seq_len, seq_len]
    For each head, compute WAAD, then average the lowest 30% heads as final WAAD.
    """
    num_heads, _, seq_len = attn.shape
    cur_idx = seq_len - 1
    waad_per_head = []
    if cur_idx > 0:
        past_indices = torch.arange(0, cur_idx, device=attn.device)
        deltas = (cur_idx - past_indices).float()
        weights = torch.clamp(deltas, max=W)
        for h in range(num_heads):
            attns_to_prev = attn[h, -1, :-1]
            waad = float((attns_to_prev * weights).sum().item())
            waad_per_head.append(waad)
    else:
        waad_per_head = [0.0 for _ in range(num_heads)]
    # 取最小的30% head
    k = max(1, int(num_heads * 0.3))
    waad_per_head_sorted = sorted(waad_per_head)
    waad_final = float(np.mean(waad_per_head_sorted[:k]))
    return waad_final

def normalized_entropy_from_logprobs(logprobs: torch.Tensor) -> float:
    """ logprobs: [V] """
    probs = logprobs.exp()
    ent = -(probs * logprobs).sum().item()
    # ent_max = math.log(probs.numel())
    return ent


def get_threshold(tokenizer, model, device, dataset: str, tau: int = 80, tau_importance: int = 80, max_items: int = 100, max_gen_tokens: int = 1024) -> float:
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
    importance_scores: List[float] = []
    waads: List[float] = []

    for item in data:
        usr_prompt = generate_usr_prompt(dataset, item)
        # prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{usr_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        # prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        # attention_mask = inputs.get("attention_mask", None)

        past = None
        cur_ids = input_ids

        for _ in range(max_gen_tokens):
            with torch.no_grad():
                out = model(input_ids=cur_ids, past_key_values=past, use_cache=True, output_attentions=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            past = out.past_key_values
            attentions = out.attentions[-1][0]  # [num_heads, seq_len, seq_len]
            # print(attentions.shape)
            # print(attentions)

            next_id = int(torch.argmax(logits).item())

            logprobs = log_softmax(logits)
            token_entropy = normalized_entropy_from_logprobs(logprobs)
            entropies.append(token_entropy)
            
            # 对所有头取平均
            attn_avg = attentions.mean(dim=0)  # shape: [seq_len, seq_len]
            # print(attn_avg)
            # print(attn_avg.shape)
            # 当前位置对前面所有位置的最大注意力
            # sum of attention to previous positions (for debug)
            # sum_attn = attn_avg[-1, :-1].sum().item()
            # print(f"sum_attn: {sum_attn}")
            importance = float(torch.max(attn_avg[-1, :-1]).item())
            importance_scores.append(importance)

            # WAAD: for each head, compute WAAD, then take mean of lowest 30% heads
            waad = get_waad_per_head(attentions, W=10)
            waads.append(waad)
            # print(f"WAAD_t (W=10, lowest 30% mean): {waad}")

            # prepare next input
            if next_id == tokenizer.eos_token_id:
                break
            cur_ids = torch.tensor([[next_id]], device=device, dtype=torch.int)
            # attention_mask = None

    if len(entropies) == 0:
        return float('nan')
    arr = np.array(entropies)
    threshold = float(np.percentile(arr, tau))
    
    # 计算重要性分数的阈值
    importance_arr = np.array(importance_scores)
    importance_threshold = float(np.percentile(importance_arr, tau_importance))

    # waad_arr = np.array(waads) if len(waads) > 0 else np.array([0.0])
    # waad_threshold = float(np.percentile(waad_arr, tau))
    
    logger.info(f"Entropy threshold ({tau}th percentile): {threshold:.4f}")
    logger.info(f"Importance score threshold ({tau_importance}th percentile): {importance_threshold:.4f}")
    logger.info(f"Entropy - min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}")
    logger.info(f"Importance scores - min: {importance_arr.min():.4f}, max: {importance_arr.max():.4f}, mean: {importance_arr.mean():.4f}")
    
    return threshold, importance_threshold


if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="eager").to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    thr = get_threshold(tokenizer, model, device, dataset)
    logger.info(f"Threshold: {thr}")
