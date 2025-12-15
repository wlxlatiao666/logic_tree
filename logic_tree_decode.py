# -*- coding: utf-8 -*-
"""
Logic-Branch Decoding with HF Transformers (minimal prototype)
- Model: GPT-2 (changeable)
- Strategy: branch only at high-entropy logical connectives
- Outputs: tree, leaves, uncertainty score U
"""

import math
import heapq
import json
import os
import time
import copy
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# 触发阈值
TAU = 0.80     # 归一化熵阈值（触发分叉）
TAU_IMPORTANCE = 0.50
BRANCHES_M = 3      # 每次分叉产生的分支数
MAX_LEAVES = 20
MAX_NEW_TOKENS = 32768

TEMPERATURE = 0.7
TOPK = 50
NUCLEUS_P = 0.9

# ====== Utilities ======
def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.log_softmax(logits, dim=-1)

def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)

def normalized_entropy_from_logprobs(logprobs: torch.Tensor) -> float:
    """ logprobs: [V] """
    probs = logprobs.exp()
    ent = -(probs * logprobs).sum().item()
    # ent_max = math.log(probs.numel())
    return ent

def topk_or_nucleus_filter(logits: torch.Tensor, topk: int, p: float) -> torch.Tensor:
    """Keep tokens in top-k by prob, and also keep minimal set reaching cumulative p (nucleus)."""
    probs = softmax(logits)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    # nucleus
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumsum <= p
    if nucleus_mask.sum() == 0:
        nucleus_mask[0] = True
    k_mask = torch.arange(probs.numel(), device=probs.device) < topk
    # combine
    keep_sorted_mask = torch.logical_or(nucleus_mask, k_mask)
    keep_idx = sorted_idx[keep_sorted_mask]
    mask = torch.full_like(probs, fill_value=False, dtype=torch.bool)
    mask[keep_idx] = True
    # set -inf for filtered
    filtered = logits.clone()
    filtered[~mask] = -float("inf")
    return filtered

def sample_one_from_logits(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)
    return int(torch.multinomial(probs, num_samples=1).item())

def greedy_sample(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())

# todo: 改善连接词识别逻辑
def tid_to_clean_token(tokenizer, tid: int) -> str:
    """Decode single token id and clean leading spaces/subword markers."""
    s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    return s.strip().lower()

def stop_condition(token_id: int, tokenizer) -> bool:
    return token_id == tokenizer.eos_token_id

# ====== Tree structures ======
@dataclass(order=True)
class PrioritizedItem:
    sort_index: Tuple[int, float] = field(init=False)
    depth: int
    neg_logprob: float
    node: "Node" = field(compare=False)
    past: Optional[Tuple[torch.Tensor, Optional[tuple]]] = field(compare=False, default=None)
    # past = (input_ids, past_key_values)

    def __post_init__(self):
        self.sort_index = (self.depth, self.neg_logprob)

@dataclass
class Node:
    text: str = field(default="")
    cum_logprob: float = field(default=0.0)
    prob: float = field(default=0.0)
    length: int = field(default=0)
    depth: int = field(default=0)
    children: List["Node"] = field(default_factory=list)
    split_positions: List[int] = field(default_factory=list)


# ====== Core decoding ======
@torch.no_grad()
def logic_branch_decode(
    tokenizer, model, device, prompt: str, sample: bool = False,
    tau: float = TAU, tau_importance: float = TAU_IMPORTANCE, branches_m: int = BRANCHES_M,
    max_leaves: int = MAX_LEAVES, max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    topk: int = TOPK, nucleus_p: float = NUCLEUS_P
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_kv = None
    # logger.info("token_id: ", input_ids)
    # logger.info("key_value: ", past_kv)

    root = Node(text="", cum_logprob=0.0, prob=1.0, length=0, depth=0)
    frontier: List[PrioritizedItem] = []
    heapq.heappush(frontier, PrioritizedItem(depth=0, neg_logprob=0.0, node=root, past=(input_ids, past_kv)))
    leaves: List[Node] = []
    new_tokens_cnt = 0

    while frontier:
        item = heapq.heappop(frontier)
        depth, node, (cur_ids, cur_past) = item.depth, item.node, item.past

        # This path generation loop
        for _ in range(max_new_tokens):
            # one-step forward using last token id and past_kv
            # logger.info("cur_id: ", cur_ids)
            if cur_past is None:
                out = model(input_ids=cur_ids, attention_mask=attention_mask, use_cache=True)
            else:
                out = model(input_ids=cur_ids, past_key_values=cur_past, use_cache=True)
                # logger.info("cur_token: ", tokenizer.decode(cur_ids[0, -1].item()), "cur_ids: ", cur_ids, "key_value: ", cur_past[0][0].shape)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            cur_past = out.past_key_values
            # logger.info("key_value: ", cur_past)

            logprobs = log_softmax(logits)
            H_norm = normalized_entropy_from_logprobs(logprobs)
            # is_split_point = node.text.endswith((".","?","!","\n")) and H_norm >= tau
            # random_factor = random.random()
            # is_split_point = H_norm >= tau and random_factor < 0.2
            top1_id = int(torch.argmax(logits).item())
            top1_ids = torch.tensor([[top1_id]], device=device)
            out1 = model(input_ids=top1_ids, past_key_values=cur_past, use_cache=True, output_attentions=True)
            attentions = out1.attentions[-1][0] 
            attn_avg = attentions.mean(dim=0)
            importance = float(torch.max(attn_avg[-1, :]).item())

            is_split_point = H_norm >= tau and importance >= tau_importance

            if len(leaves) + len(frontier) + 1 >= max_leaves:
                is_split_point = False

            if is_split_point:
                node.split_positions.append(node.length)
                filt_probs = softmax(logits)
                top_vals, top_idx = torch.topk(filt_probs, k=branches_m)
                top_idx = top_idx.tolist()
                top_vals = top_vals.tolist()

                # connective mass
                conn_candidates: List[Tuple[int, float]] = []
                for tid, p in zip(top_idx, top_vals):
                    conn_candidates.append((tid, p))
                conn_candidates.sort(key=lambda x: x[1], reverse=True)
                conn_candidates = conn_candidates[:max_leaves-len(leaves)-len(frontier)]

                children = []
                items = []
                # materialize children, commit one token for each branch
                total_p = sum(p for _, p in conn_candidates)
                for tid, p in conn_candidates:
                    new_ids = torch.tensor([[tid]], device=device)
                    child_text = node.text + tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                    new_tokens_cnt += 1
                    child_logprob = node.cum_logprob + math.log(max(p, 1e-12))
                    child_prob = node.prob * p / total_p
                    # logger.info("node_prob: ", node.prob, "child_prob: ", child_prob)
                    child_length = node.length + 1
                    child = Node(text=child_text, cum_logprob=child_logprob, prob=child_prob, length=child_length, depth=depth+1, split_positions=node.split_positions.copy())

                    tmp_ids, tmp_past = new_ids, copy.deepcopy(cur_past)

                    children.append(child)
                    items.append(PrioritizedItem(
                        depth=depth + 1,
                        neg_logprob=-child.cum_logprob,
                        node=child,
                        past=(tmp_ids, tmp_past)
                    ))

                for child, it in zip(children, items):
                    node.children.append(child)
                    cur_ids = it.past[0]
                    if stop_condition(cur_ids[0, 0], tokenizer):
                        leaves.append(it.node)
                        continue
                    heapq.heappush(frontier, it)
                break
            else:
                # regular decoding with low temperature
                if sample:
                    logits_f = topk_or_nucleus_filter(logits, topk=topk, p=nucleus_p)
                    next_id = sample_one_from_logits(logits_f, temperature=temperature)
                    next_prob = softmax(logits_f)[next_id].item()
                else:
                    next_id = greedy_sample(logits)
                    next_prob = softmax(logits)[next_id].item()

                node.text += tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
                new_tokens_cnt += 1
                node.cum_logprob += math.log(max(next_prob, 1e-12))
                # 更新token熵和长度
                node.length += 1

                cur_ids = torch.tensor([[next_id]], device=device)

                # stopping rules
                if stop_condition(next_id, tokenizer):
                    leaves.append(node)
                    break

        # if we exited loop without adding to leaves and cannot go deeper, finalize
        if node not in leaves and len(node.children) == 0:
            leaves.append(node)

    return leaves, new_tokens_cnt


def pretty_print_tree(node: Node, prefix: str = "", depth: int = 1, step: int = 1, parent_prefix: str = ""):
    # 新增公共前缀长度计算
    common_prefix_len = len(os.path.commonprefix([parent_prefix, node.text]))
    
    # 按公共前缀分割文本
    split_index = common_prefix_len if common_prefix_len > 0 else None
    prefix_part = node.text[:split_index] if split_index else ""
    new_text_part = node.text[split_index:] if split_index else node.text
    
    # 构建带层级和步骤的显示格式
    connector = "└── " if not node.children else "├── "
    branch = "│   " if node.children else "    "
    
    # 输出分割后的文本部分
    logger.info(f"L{depth}-S{step} {prefix}{connector}{new_text_part}")
    
    # 递归处理子节点
    for i, child in enumerate(node.children):
        new_prefix = f"{prefix}{branch}"
        new_depth = depth + 1
        new_step = i + 1
        
        # 传递当前节点的完整前缀用于下次公共前缀计算
        pretty_print_tree(child, new_prefix, new_depth, new_step, parent_prefix=node.text)

# ====== Demo ======
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="eager").to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(f"./sys_prompt.json", "r") as f:
        system_prompt = json.load(f)["gsm8k"]
    # system_prompt = ''
    query = '''Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?'''
    # query = '''Context: In a business whose owners and employees all belong to one family, the employees can be paid exceptionally low wages. Hence, general operating expenses are much lower than they would be for other business ventures, making profits higher. So a family business is a family' s surest road to financial prosperity.
    # Question: The reasoning in the argument is flawed because the argument
    # A. ignores the fact that in a family business, paying family members low wages may itself reduce the family's prosperity
    # B. presumes, without providing justification, that family members are willing to work for low wages in a family business because they believe that doing so promotes the family's prosperity
    # C. ignores the fact that businesses that achieve high levels of customer satisfaction are often profitable even if they pay high wages
    # D. presumes, without providing justification, that only businesses with low general operating expenses can succeed'''
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    for i in range(10):
        # torch.cuda.manual_seed_all(41)
        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, device, prompt=prompt, sample=True, branches_m=3)

        logger.info("\n--- Logic Tree ---")
        pretty_print_tree(root)

        logger.info("\n--- Leaves ---")
        # total_prob = sum(leaf.prob for leaf in leaves)
        for i, leaf in enumerate(leaves):
            txt = leaf.text.replace("\n", " ")
            logger.info(f"[{i:02d}] p={leaf.prob:.3f}  text_tail='{txt}'")
        logger.info(f"new_tokens_cnt: {new_tokens_cnt}")


if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
