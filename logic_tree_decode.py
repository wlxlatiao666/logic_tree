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
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer, util

import sys

sys.stdout = open(f'./logs/output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w', encoding='utf-8', buffering=1)
# set_seed(41)
embedder = SentenceTransformer('/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/all-mpnet-base-v2')

# ====== Config ====== 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONNECTIVES = {
#     "then", "but", "however", "therefore", "thus", "so", "because",
#     "hence", "yet", "although", "though", "instead", "whereas",
#     "nonetheless", "nevertheless", "consequently", "furthermore", "moreover", "meanwhile",
#     "first", "second", "next", "last", "after", "finally", "besides"
# }

# 触发阈值
TAU = 0.80     # 归一化熵阈值（触发分叉）
BRANCHES_M = 3      # 每次分叉产生的分支数
MAX_TIMES = 50

TEMPERATURE = 0.7
TOPK = 50
NUCLEUS_P = 0.9
STEPS_BRANCH = 10

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

# def is_connective_token(tokenizer, tid: int) -> bool:
#     txt = tid_to_clean_token(tokenizer, tid)
#     return txt in CONNECTIVES

def stop_condition(token_id: int, tokenizer) -> bool:
    return token_id == tokenizer.eos_token_id

def group_by_semantic_similarity(tokenizer, conn_candidates, sim_threshold=0.5):
    tokens = [tid_to_clean_token(tokenizer, tid) for tid, _ in conn_candidates]
    embeddings = embedder.encode(tokens, convert_to_tensor=True)
    groups = []
    used = set()
    for i, token in enumerate(tokens):
        if token in used:
            continue
        group = [i]
        for j in range(i+1, len(tokens)):
            if j in used:
                continue
            # sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            sim = util.cos_sim(embeddings[i], embeddings[j])
            if sim > sim_threshold:
                group.append(j)
                used.add(tokens[j])
        used.add(token)
        groups.append(group)
    # 合并概率
    merged = []
    for group in groups:
        rep_tid = conn_candidates[group[0]][0]
        prob_sum = sum(conn_candidates[idx][1] for idx in group)
        merged.append((rep_tid, prob_sum))
    return merged

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

# def calculate_entropy(node: Node):
#     if not node.children:  # 叶子节点
#         node.entropy = -node.cum_logprob / node.length if node.length > 0 else 0.0
#         return
#     # 非叶子节点，递归计算子节点entropy
#     child_entropies = []
#     for child in node.children:
#         calculate_entropy(child)
#         child_entropies.append(child.entropy)
    
#     node.entropy = sum(child_entropies) / len(child_entropies) if child_entropies else 0.0
#     return

# def collect_leaves(node: Node, leaves: List[Node]):
#     if not node.children:
#         leaves.append(node)
#     for child in node.children:
#         collect_leaves(child, leaves)

def calculate_diversity(leaves: List[Node]):
    if len(leaves) < 2:
        return 0.0

    distances = []
    embeddings = []
    for leaf in leaves:
        embeddings.append(embedder.encode(leaf.text, convert_to_tensor=True))

    for i in range(len(leaves)):
        current_distances = []
        for j in range(len(leaves)):
            if i != j:
                cos_sim = util.cos_sim(embeddings[i], embeddings[j])
                current_distances.append(1 - cos_sim)
        distances.append(sum(current_distances) / len(current_distances))
    
    diversity = sum(distances)
    return diversity.item()


# ====== Core decoding ======
@torch.no_grad()
def logic_branch_decode(
    tokenizer, model, prompt: str, sample: bool = False,
    tau: float = TAU, branches_m: int = BRANCHES_M,
    max_times: int = MAX_TIMES,
    temperature: float = TEMPERATURE,
    topk: int = TOPK, nucleus_p: float = NUCLEUS_P, steps_branch: int = STEPS_BRANCH
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_kv = None
    # print("token_id: ", input_ids)
    # print("key_value: ", past_kv)

    root = Node(text="", cum_logprob=0.0, prob=1.0, length=0, depth=0)
    frontier: List[PrioritizedItem] = []
    heapq.heappush(frontier, PrioritizedItem(depth=0, neg_logprob=0.0, node=root, past=(input_ids, past_kv)))
    leaves: List[Node] = []
    times = 1
    new_tokens_cnt = 0

    while frontier:
        item = heapq.heappop(frontier)
        depth, node, (cur_ids, cur_past) = item.depth, item.node, item.past

        # This path generation loop
        while True:
            # one-step forward using last token id and past_kv
            # print("cur_id: ", cur_ids)
            if cur_past is None:
                out = model(input_ids=cur_ids, attention_mask=attention_mask, use_cache=True)
            else:
                out = model(input_ids=cur_ids, past_key_values=cur_past, use_cache=True)
                # print("cur_token: ", tokenizer.decode(cur_ids[0, -1].item()), "cur_ids: ", cur_ids, "key_value: ", cur_past[0][0].shape)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            cur_past = out.past_key_values
            # print("key_value: ", cur_past)

            logprobs = log_softmax(logits)
            H_norm = normalized_entropy_from_logprobs(logprobs)
            # is_split_point = node.text.endswith((".","?","!","\n")) and H_norm >= tau
            random_factor = random.random()
            is_split_point = H_norm >= tau and random_factor < 0.5

            if times >= max_times:
                is_split_point = False

            if is_split_point:
                filt_probs = softmax(logits)
                top_vals, top_idx = torch.topk(filt_probs, k=branches_m)
                top_idx = top_idx.tolist()
                top_vals = top_vals.tolist()

                # connective mass
                conn_candidates: List[Tuple[int, float]] = []
                for tid, p in zip(top_idx, top_vals):
                    conn_candidates.append((tid, p))
                conn_candidates.sort(key=lambda x: x[1], reverse=True)

                child_embeddings = []
                children = []
                items = []
                # materialize children, commit one token for each branch
                total_p = sum(p for _, p in conn_candidates)
                for tid, p in conn_candidates:
                    # print("token: ", tokenizer.decode([tid], clean_up_tokenization_spaces=False), " prob: ", p, " total_p: ", total_p)
                    new_ids = torch.tensor([[tid]], device=DEVICE)
                    child_text = node.text + tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                    new_tokens_cnt += 1
                    child_logprob = node.cum_logprob + math.log(max(p, 1e-12))
                    child_prob = node.prob * p / total_p
                    # print("node_prob: ", node.prob, "child_prob: ", child_prob)
                    child_length = node.length + 1
                    child = Node(text=child_text, cum_logprob=child_logprob, prob=child_prob, length=child_length, depth=depth+1)

                    tmp_ids, tmp_past = new_ids, copy.deepcopy(cur_past)
                    # print("cur_past: ", cur_past, cur_past[0][0].shape)
                    skip_child = False
                    # all_head_attentions = None
                    if not stop_condition(tid, tokenizer):
                        for i in range(steps_branch):  # short span
                            out2 = model(input_ids=tmp_ids, past_key_values=tmp_past, use_cache=True)
                            # print("cur_token: ", tokenizer.decode(tmp_ids[0, -1].item()), "cur_ids: ", tmp_ids, "key_value: ", tmp_past[0][0].shape)
                            # attentions = out2.attentions[-1][0]
                            # head_attentions = attentions[:, -1, child_length-1].unsqueeze(-1)
                            # seq_length = attentions.shape[-1]
                            # head_attentions = head_attentions * seq_length
                            # all_head_attentions = torch.cat([all_head_attentions, head_attentions], dim=1) if all_head_attentions is not None else head_attentions
                            logits2 = out2.logits[:, -1, :].squeeze(0)
                            tmp_past = out2.past_key_values
                            # diversify sampling
                            if sample:
                                logits2 = topk_or_nucleus_filter(logits2, topk=topk, p=nucleus_p)
                                next_id = sample_one_from_logits(logits2, temperature=temperature)
                            else:
                                next_id = greedy_sample(logits2)
                            next_prob = softmax(logits2)[next_id].item()
                            child.text += tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
                            new_tokens_cnt += 1
                            child.cum_logprob += math.log(max(next_prob, 1e-12))
                            child.length += 1
                            tmp_ids = torch.tensor([[next_id]], device=DEVICE)
                            if stop_condition(next_id, tokenizer):
                                break

                        # max_per_head = torch.max(all_head_attentions, dim=-1).values
                        # avg_max = torch.mean(max_per_head).item()
                        # print("token: ", tokenizer.decode([tid], clean_up_tokenization_spaces=False), " avg_max: ", avg_max)
                        # if avg_max < 0.4:
                        #     skip_child = True

                    current_embedding = embedder.encode(child.text[len(node.text):], convert_to_tensor=True)
                    for index, embedding in enumerate(child_embeddings): 
                        sim = util.cos_sim(embedding, current_embedding)
                        if sim > 0.6:
                            children[index].prob += child.prob # children[index]和items[index].node引用了同一个node
                            skip_child = True
                            break
                    if not skip_child:
                        child_embeddings.append(current_embedding)
                        children.append(child)
                        items.append(PrioritizedItem(
                            depth=depth + 1,
                            neg_logprob=-child.cum_logprob,
                            node=child,
                            past=(tmp_ids, tmp_past)
                        ))
                        # print("tmp_past: ", tmp_past, tmp_past[0][0].shape)

                for child in children:
                    node.children.append(child)
                for it in items:
                    cur_ids = it.past[0]
                    if stop_condition(cur_ids[0, 0], tokenizer):
                        leaves.append(it.node)
                        continue
                    heapq.heappush(frontier, it)
                    times += 1
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

                cur_ids = torch.tensor([[next_id]], device=DEVICE)

                # stopping rules
                if stop_condition(next_id, tokenizer):
                    leaves.append(node)
                    break

        # if we exited loop without adding to leaves and cannot go deeper, finalize
        if node not in leaves and len(node.children) == 0:
            leaves.append(node)

    return root, leaves, new_tokens_cnt


def logsumexp_torch(xs: List[float]) -> float:
    t = torch.tensor(xs, dtype=torch.float32)
    return float(torch.logsumexp(t, dim=0).item())

def bucketize_answers(texts: List[str]) -> Dict[str, int]:
    def last_sentence(s: str) -> str:
        s = s.strip()
        for sep in [".", "!", "?", "\n"]:
            if sep in s:
                parts = s.split(sep)
                if parts[-1] == "":
                    parts = parts[:-1]
                if parts:
                    return parts[-1].strip().lower()
        return s.lower()
    buckets: Dict[str, int] = {}
    for t in texts:
        key = last_sentence(t)[:80]  # clip
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


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
    print(f"L{depth}-S{step} {prefix}{connector}{new_text_part}")
    
    # 递归处理子节点
    for i, child in enumerate(node.children):
        new_prefix = f"{prefix}{branch}"
        new_depth = depth + 1
        new_step = i + 1
        
        # 传递当前节点的完整前缀用于下次公共前缀计算
        pretty_print_tree(child, new_prefix, new_depth, new_step, parent_prefix=node.text)

# ====== Demo ======
def main():
    model_name = "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="eager").to(DEVICE)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(f"./sys_prompt.json", "r") as f:
        system_prompt = json.load(f)["reclor"]
    # system_prompt = ''
    # query = '''Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?'''
    query = '''Context: In a business whose owners and employees all belong to one family, the employees can be paid exceptionally low wages. Hence, general operating expenses are much lower than they would be for other business ventures, making profits higher. So a family business is a family' s surest road to financial prosperity.
    Question: The reasoning in the argument is flawed because the argument
    A. ignores the fact that in a family business, paying family members low wages may itself reduce the family's prosperity
    B. presumes, without providing justification, that family members are willing to work for low wages in a family business because they believe that doing so promotes the family's prosperity
    C. ignores the fact that businesses that achieve high levels of customer satisfaction are often profitable even if they pay high wages
    D. presumes, without providing justification, that only businesses with low general operating expenses can succeed'''
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    for i in range(3):
        # torch.cuda.manual_seed_all(41)
        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, prompt=prompt, sample=True, M=3)

        print("\n--- Logic Tree ---")
        pretty_print_tree(root)

        print("\n--- Leaves ---")
        # total_prob = sum(leaf.prob for leaf in leaves)
        for i, leaf in enumerate(leaves):
            txt = leaf.text.replace("\n", " ")
            print(f"[{i:02d}] p={leaf.prob:.3f}  text_tail='{txt}'")
        print(f"new_tokens_cnt: {new_tokens_cnt}")

        # diversity = calculate_diversity(leaves)
        # print("diversity", diversity)
        # print(f"Diversity score: {diversity:.3f}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
