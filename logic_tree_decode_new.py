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
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, deque
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer, util
from utils import generate_usr_prompt

import sys

sys.stdout = open(f'./logs/output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w', encoding='utf-8', buffering=1)
# set_seed(41)

# ====== Config ====== 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 触发阈值
TAU = 0.50     # 归一化熵阈值（触发分叉）
TAU_SIM = 0.60    # 语义相似度阈值（触发分叉）
NUM_BRANCHES = 3      # 每次分叉产生的分支数
WINDOW_SIZE = 10
TEMPERATURE = 1.0
TOPK = 50
NUCLEUS_P = 0.9
MAX_LEAVES = 128
MAX_TOKENS = 1024

# ====== Utilities ======
def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)

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

def stop_condition(token_id: int, tokenizer) -> bool:
    return token_id == tokenizer.eos_token_id


# ====== Tree structures ======
@dataclass(order=True)
class PrioritizedItem:
    sort_index: int = field(init=False)
    depth: int
    # neg_logprob: float
    node: "Node" = field(compare=False)
    past: Optional[Tuple[torch.Tensor, Optional[tuple]]] = field(compare=False, default=None)
    # past = (input_ids, past_key_values)

    def __post_init__(self):
        self.sort_index = self.depth

@dataclass
class Node:
    # text: str = field(default="")
    ids: List[int] = field(default_factory=list)
    # cum_logprob: float = field(default=0.0)
    entropy_window: deque = field(init=False)
    # prob: float = field(default=1.0)
    length: int = field(default=0)
    # depth: int = field(default=0)
    children: List["Node"] = field(default_factory=list)
    is_leaf: bool = field(default=False)

# ====== Core decoding ======
@torch.no_grad()
def logic_branch_decode(
    tokenizer, model, embedder, prompt: str,
    tau: float = TAU, tau_sim: float = TAU_SIM, num_branches: int = NUM_BRANCHES, window_size: int = WINDOW_SIZE,
    temperature: float = TEMPERATURE,
    topk: int = TOPK, nucleus_p: float = NUCLEUS_P, max_leaves: int = MAX_LEAVES, max_tokens: int = MAX_TOKENS
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    init_past = out.past_key_values

    root = Node()
    frontier: List[PrioritizedItem] = []
    heapq.heappush(frontier, PrioritizedItem(depth=0, node=root, past=(input_ids, None)))
    leaves: List[Node] = []

    num_leaves = 0
    while frontier:
        item = heapq.heappop(frontier)
        depth, node, (cur_ids, cur_past) = item.depth, item.node, item.past
        # print(f"current node text: '{tokenizer.decode(node.ids, clean_up_tokenization_spaces=False)}', depth: {depth}, frontier size: {len(frontier)}, leaves: {num_leaves}\n")
        node.entropy_window = deque(maxlen=window_size)
        while True:
            if len(node.ids) >= max_tokens:
                return root, leaves
            out = model(input_ids=cur_ids, past_key_values=cur_past, use_cache=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            cur_past = out.past_key_values
            next_id = greedy_sample(logits)
            next_prob = softmax(logits)[next_id].item()

            node.ids.append(next_id)
            # node.cum_logprob += math.log(max(next_prob, 1e-12))
            node.length += 1
            cur_ids = torch.tensor([[next_id]], device=DEVICE, dtype=torch.int)

            # logprobs = log_softmax(logits)
            token_entropy = -math.log(max(next_prob, 1e-5))
            node.entropy_window.append(token_entropy)
            if len(node.entropy_window) < window_size:
                is_split_point = False
            else:
                avg_H = sum(node.entropy_window) / len(node.entropy_window)
                is_split_point = avg_H >= tau
                if is_split_point:
                    print("=== Split Point Triggered ===")
                    print(f"Last {window_size} tokens:",tokenizer.decode(node.ids[-window_size:], clean_up_tokenization_spaces=False), " | AvgH:", avg_H)
                    print('\n')
                else:
                    print("=== Split Point Not Triggered ===")
                    print(f"Last {window_size} tokens:",tokenizer.decode(node.ids[-window_size:], clean_up_tokenization_spaces=False), " | AvgH:", avg_H)
                    print('\n')

            if is_split_point:
                node.ids = node.ids[:-window_size]
                # node.cum_logprob
                node.length -= window_size
                cur_ids = torch.tensor([node.ids], device=DEVICE, dtype=torch.int)
                embeddings = []
                for _ in range(num_branches):
                    child = Node(ids=copy.deepcopy(node.ids), length=node.length)
                    child.entropy_window = deque(maxlen=window_size)
                    if cur_ids.numel() == 0:
                        tmp_ids, tmp_past = input_ids, None
                    else:
                        tmp_ids, tmp_past = cur_ids, copy.deepcopy(init_past)
                    for _ in range(window_size):
                        out = model(input_ids=tmp_ids, past_key_values=tmp_past, use_cache=True)
                        logits = out.logits[:, -1, :].squeeze(0)
                        tmp_past = out.past_key_values
                        logits = topk_or_nucleus_filter(logits, topk=topk, p=nucleus_p)
                        next_id = sample_one_from_logits(logits, temperature=temperature)
                        next_prob = softmax(logits)[next_id].item()
                        token_entropy = -math.log(max(next_prob, 1e-5))
                        child.entropy_window.append(token_entropy)
                        child.ids.append(next_id)
                        # child.cum_logprob += math.log(max(next_prob, 1e-12))
                        child.length += 1
                        tmp_ids = torch.tensor([[next_id]], device=DEVICE, dtype=torch.int)
                        if stop_condition(next_id, tokenizer):
                            child.is_leaf = True
                            break
                    
                    skip_child = False
                    # avg_H = sum(child.entropy_window) / len(child.entropy_window)
                    avg_H = 0.0
                    if avg_H >= tau:
                        skip_child = True
                    else:
                        current_embedding = embedder.encode(tokenizer.decode(child.ids[len(node.ids):], clean_up_tokenization_spaces=False), convert_to_tensor=True)
                        for emb in embeddings:
                            sim = util.cos_sim(current_embedding, emb)
                            if sim > tau_sim:
                                skip_child = True
                                break
                    if not skip_child:
                        node.children.append(child)
                        embeddings.append(current_embedding)
                        if not child.is_leaf:
                            heapq.heappush(frontier, PrioritizedItem(depth=depth+1, node=child, past=(tmp_ids, tmp_past)))
                            # print("=== New Branch Created ===")
                            # print("child text:",tokenizer.decode(child.ids, clean_up_tokenization_spaces=False))
                            # print('\n')
                        else:
                            leaves.append(child)
                            # print("=== New Leaf Created ===")
                            # print("leaf text:",tokenizer.decode(child.ids, clean_up_tokenization_spaces=False))
                            # print('\n')
                            num_leaves += 1
                            if num_leaves >= max_leaves:
                                return root, leaves
                break
            else:
                if stop_condition(next_id, tokenizer):
                    leaves.append(node)
                    # print("=== New Leaf Created ===")
                    # print("leaf text:",tokenizer.decode(node.ids, clean_up_tokenization_spaces=False))
                    # print('\n')
                    num_leaves += 1
                    if num_leaves >= max_leaves:
                        return root, leaves
                    node.is_leaf = True
                    break

    return root, leaves

def pretty_print_tree(tokenizer, node: Node, depth: int = 1, index: int = 1):
    print(f"Layer {depth}, Node {index} text:\n'{tokenizer.decode(node.ids, clean_up_tokenization_spaces=False)}'")
    for i, child in enumerate(node.children):
        pretty_print_tree(tokenizer, child, depth + 1, i + 1)


def compute_avg_branching_factor(root: Node) -> float:
    """Compute average branching factor for the tree rooted at `root`.

    Average branching factor = average number of children among internal nodes.
    Returns 0.0 if there are no internal nodes.
    """
    internal_counts = []

    def dfs(node: Node):
        if node.children:
            internal_counts.append(len(node.children))
            for c in node.children:
                dfs(c)

    dfs(root)
    if not internal_counts:
        return 0.0
    return sum(internal_counts) / len(internal_counts)

# ====== Demo ======
def main(dataset: str):
    model_name = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct" 
    embedder = SentenceTransformer('/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/all-mpnet-base-v2')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(f"./sys_prompt.json", "r") as f:
        system_prompt = json.load(f)[dataset]
    with open(f"./data/{dataset}/test.json", "r") as f:
        data = json.load(f)[:3]
    for i, item in enumerate(data):
        query = generate_usr_prompt(dataset, item)
        print(f"Query {i+1}: {query}\n")
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        for i in range(1):
            root, leaves = logic_branch_decode(tokenizer, model, embedder, prompt=prompt, num_branches=3)

            print("\n--- Logic Tree ---")
            pretty_print_tree(tokenizer, root)

            # compute and print average branching factor as a measure of tree complexity
            avg_b = compute_avg_branching_factor(root)
            print(f"Average branching factor: {avg_b:.3f}")

            print("\n--- Leaves ---")
            for i, leaf in enumerate(leaves):
                txt = tokenizer.decode(leaf.ids, clean_up_tokenization_spaces=False)
                print(f"[{i:02d}] text_tail='{txt}'")

    # system_prompt = ''
    # query = 'Solve the quadratic equation$x^2-6x+9={5-2x}^2$'
    # prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    # root, leaves = logic_branch_decode(tokenizer, model, prompt=prompt, num_branches=3)

    # print("\n--- Logic Tree ---")
    # pretty_print_tree(tokenizer, root)

    # print("\n--- Leaves ---")
    # for i, leaf in enumerate(leaves):
    #     txt = tokenizer.decode(leaf.ids, clean_up_tokenization_spaces=False)
    #     print(f"[{i:02d}] text_tail='{txt}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset

    start_time = time.time()
    main(dataset)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
