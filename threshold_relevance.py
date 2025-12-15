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


def normalized_entropy_from_logprobs(logprobs: torch.Tensor) -> float:
    """ logprobs: [V] """
    probs = logprobs.exp()
    ent = -(probs * logprobs).sum().item()
    # ent_max = math.log(probs.numel())
    return ent

def compute_importance_scores(attentions: torch.Tensor) -> torch.Tensor:
    """
    计算基于注意力的重要性分数。
    
    根据公式: a_max(i) = max_{j>i} A_{j,i}
    其中 A 是注意力矩阵，A[j,i] 表示位置j对位置i的注意力权重。
    
    Args:
        attentions: shape [num_heads, seq_len, seq_len]，多头注意力矩阵
                   attentions[h, j, i] 表示第h个头中位置j对位置i的权重
    
    Returns:
        importance_scores: shape [seq_len]，每个位置的重要性分数（对所有头取平均）
    """
    # 处理多头注意力：先对所有头取平均得到单个注意力矩阵
    # attentions shape: [num_heads, seq_len, seq_len]
    if attentions.dim() == 3:
        # 多头情况，对头维度求平均
        attn_avg = attentions.mean(dim=0)  # shape: [seq_len, seq_len]
    else:
        # 已经是单个注意力矩阵
        attn_avg = attentions
    
    seq_len = attn_avg.shape[0]
    importance_scores = torch.zeros(seq_len, device=attn_avg.device, dtype=attn_avg.dtype)
    
    # 对每个位置 i，计算所有 j>i 对位置 i 的最大注意力权重
    for i in range(seq_len):
        if i < seq_len - 1:
            # 获取所有 j>i 的行，取第i列的最大值
            importance_scores[i] = torch.max(attn_avg[i+1:, i])
        # 如果 i 是最后一个位置，没有 j>i，保持为0
    
    return importance_scores

def get_threshold(tokenizer, model, device, dataset: str, tau: int = 80, max_items: int = 1, max_gen_tokens: int = 1024) -> float:
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
                out = model(input_ids=cur_ids, past_key_values=past, use_cache=True, output_attentions=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            past = out.past_key_values
            attentions = out.attentions[-1][0]  # [num_heads, seq_len, seq_len]
            # print(attentions.shape)

            next_id = int(torch.argmax(logits).item())

            logprobs = log_softmax(logits)
            token_entropy = normalized_entropy_from_logprobs(logprobs)
            entropies.append(token_entropy)
            
            # 计算当前位置的重要性分数
            # 获取该位置在所有之前位置的注意力的最大值（衡量该位置对之前位置的关注度）
            seq_len = attentions.shape[-1]  # seq_len 在最后一维
            current_pos = seq_len - 1
            if current_pos > 0:
                # 对所有头取平均
                attn_avg = attentions.mean(dim=0)  # shape: [seq_len, seq_len]
                # print(attn_avg.shape)
                # 当前位置对前面所有位置的最大注意力
                importance = float(torch.max(attn_avg[-1, :current_pos]).item())
                # importance = 0.0
            else:
                importance = 0.0
            importance_scores.append(importance)

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
    importance_threshold = float(np.percentile(importance_arr, tau))
    
    logger.info(f"Entropy threshold ({tau}th percentile): {threshold:.4f}")
    logger.info(f"Importance score threshold ({tau}th percentile): {importance_threshold:.4f}")
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
