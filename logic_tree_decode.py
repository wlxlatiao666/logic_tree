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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer, util
import sys

sys.stdout = open('output11.log', 'w', encoding='utf-8')
# set_seed(41)
embedder = SentenceTransformer('/mnt/public/gpfs-jd/code/weilongxuan/all-mpnet-base-v2')

# ====== Config ======
MODEL_NAME = "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONNECTIVES = {
    "then", "but", "however", "therefore", "thus", "so", "because",
    "hence", "yet", "although", "though", "instead", "whereas",
    "nonetheless", "nevertheless", "consequently", "furthermore", "moreover", "meanwhile",
    "first", "second", "next", "last", "after", "finally", "besides"
}

# 触发阈值
TAU = 1.00          # 归一化熵阈值（触发分叉）
BETA = 0.20         # top候选中连接词的总概率质量阈值（辅判）
BRANCHES_M = 3      # 每次分叉产生的分支数(暂时没用)
MAX_DEPTH = 5       # 最大分叉层数
MAX_NEW_TOKENS = 2048
MAX_NODES = 250

TEMPERATURE_BASE = 1.0   # 非分叉段解码温度
TEMPERATURE_BRANCH = 0.9 # 分叉段拓展温度
TOPK = 50
NUCLEUS_P = 0.9
TOPK_BRANCH = 5
NUCLEUS_P_BRANCH = 0.99
P_LOWER_BOUND = 0.01


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

# todo: 改善连接词识别逻辑
def tid_to_clean_token(tokenizer, tid: int) -> str:
    """Decode single token id and clean leading spaces/subword markers."""
    s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    return s.strip().lower()

def is_connective_token(tokenizer, tid: int) -> bool:
    txt = tid_to_clean_token(tokenizer, tid)
    return txt in CONNECTIVES

# def stop_condition(decoded_text: str) -> bool:
#     return decoded_text.endswith((".", "?", "!", "\n"))

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
    text: str
    cum_logprob: float
    prob: float
    length: int
    depth: int
    split_entropy: Optional[float] = None
    is_split: bool = False
    children: List["Node"] = field(default_factory=list)


# ====== Core decoding ======
@torch.no_grad()
def logic_branch_decode(
    tokenizer, model, prompt: str,
    tau: float = TAU, beta: float = BETA, M: int = BRANCHES_M,
    max_depth: int = MAX_DEPTH, max_new_tokens: int = MAX_NEW_TOKENS,
    max_nodes: int = MAX_NODES,
    temperature_base: float = TEMPERATURE_BASE,
    temperature_branch: float = TEMPERATURE_BRANCH,
    topk: int = TOPK, nucleus_p: float = NUCLEUS_P, topk_branch: int = TOPK_BRANCH, nucleus_p_branch: float = NUCLEUS_P_BRANCH, p_lower_bound: float = P_LOWER_BOUND
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
    nodes_cnt = 1
    max_width = 0
    depth_counter = defaultdict(int)

    count = 0
    while frontier and nodes_cnt < max_nodes:
        item = heapq.heappop(frontier)
        depth_counter[item.depth] += 1
        max_width = max(max_width, depth_counter[item.depth])
        depth, node, (cur_ids, cur_past) = item.depth, item.node, item.past

        # This path generation loop
        steps = 0
        while steps < max_new_tokens:
            # one-step forward using last token id and past_kv
            # print("cur_id: ", cur_ids)
            if cur_past is None:
                out = model(input_ids=cur_ids, attention_mask=attention_mask, use_cache=True)
            else:
                out = model(input_ids=cur_ids, past_key_values=cur_past, use_cache=True)
            logits = out.logits[:, -1, :].squeeze(0)  # [V]
            cur_past = out.past_key_values
            # print("key_value: ", cur_past)

            logprobs = log_softmax(logits)
            H_norm = normalized_entropy_from_logprobs(logprobs)

            # Build candidate filter set
            current_topk_branch = min(topk_branch + node.depth, 10)
            filtered = topk_or_nucleus_filter(logits, topk=current_topk_branch, p=nucleus_p_branch)
            filt_probs = softmax(filtered)
            top_vals, top_idx = torch.topk(filt_probs, k=min(current_topk_branch, filt_probs.numel()))
            top_idx = top_idx.tolist()
            top_vals = top_vals.tolist()
            total_p = sum(top_vals)

            # connective mass
            conn_candidates: List[Tuple[int, float]] = []
            for tid, p in zip(top_idx, top_vals):
                if p <= 0:
                    continue
                # 过滤掉概率过低的分支
                if p / total_p * node.prob < p_lower_bound:
                    continue
                conn_candidates.append((tid, p))
            conn_candidates.sort(key=lambda x: x[1], reverse=True)

            # split decision
            top1_id = int(torch.argmax(softmax(logits)).item())
            is_top1_conn = is_connective_token(tokenizer, top1_id)
            # is_split_point = (is_top1_conn and H_norm >= tau) or (conn_mass >= beta and H_norm >= (tau * 0.9))
            # is_split_point = is_top1_conn and H_norm >= tau
            is_split_point = node.text.endswith((".","?","!","\n")) and H_norm >= tau
            # is_split_point = H_norm >= tau

            # if count == 0:
            #     is_split_point = True
            #     count += 1

            if len(conn_candidates) <= 1:
                is_split_point = False
            else:
                total_p = sum(p for _, p in conn_candidates)
                for index in range(len(conn_candidates)):
                    conn_candidates[index] = (conn_candidates[index][0], conn_candidates[index][1] / total_p)
                conn_candidates = group_by_semantic_similarity(tokenizer, conn_candidates)

            if is_split_point and depth < max_depth:
                node.is_split = True
                node.split_entropy = H_norm

                # choose branches: prefer multiple connective tokens; if not enough, fill with other candidates
                branches = conn_candidates

                # materialize children, commit one token for each branch
                total_p = sum(p for _, p in branches)
                for tid, p in branches:
                    new_ids = torch.tensor([[tid]], device=DEVICE)
                    child_text = node.text + tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                    child_logprob = node.cum_logprob + math.log(max(p, 1e-12))
                    child_prob = node.prob * p / total_p
                    child_length = node.length + 1
                    child = Node(text=child_text, cum_logprob=child_logprob, prob=child_prob, length=child_length, depth=depth+1)

                    # continue a short span at higher temperature to differentiate branches
                    # (decode until we meet a punctuation or a few tokens)
                    # tmp_ids, tmp_past = new_ids, cur_past
                    # for _ in range(8):  # short span
                    #     out2 = model(input_ids=tmp_ids, past_key_values=tmp_past, use_cache=True)
                    #     logits2 = out2.logits[:, -1, :].squeeze(0)
                    #     tmp_past = out2.past_key_values
                    #     # diversify sampling
                    #     logits2 = topk_or_nucleus_filter(logits2, topk=topk, p=nucleus_p)
                    #     next_id = sample_one_from_logits(logits2, temperature=temperature_branch)
                    #     next_prob = softmax(logits2)[next_id].item()
                    #     child.text += tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
                    #     child.cum_logprob += math.log(max(next_prob, 1e-12))
                    #     child.length += 1
                    #     tmp_ids = torch.tensor([[next_id]], device=DEVICE)
                    #     if stop_condition(next_id, tokenizer):
                    #         break

                    node.children.append(child)
                    heapq.heappush(frontier, PrioritizedItem(
                        depth=depth + 1,
                        neg_logprob=-child.cum_logprob,
                        node=child,
                        past=(new_ids, cur_past)
                    ))
                    nodes_cnt += 1

                # end current path expansion at the split
                break

            else:
                # regular decoding with low temperature
                logits_f = topk_or_nucleus_filter(logits, topk=topk, p=nucleus_p)
                next_id = sample_one_from_logits(logits_f, temperature=temperature_base)
                next_prob = softmax(logits_f)[next_id].item()

                node.text += tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
                node.cum_logprob += math.log(max(next_prob, 1e-12))
                # 更新token熵和长度
                node.length += 1

                cur_ids = torch.tensor([[next_id]], device=DEVICE)
                steps += 1

                # stopping rules
                if stop_condition(next_id, tokenizer) or steps >= max_new_tokens:
                    leaves.append(node)
                    break

        # if we exited loop without adding to leaves and cannot go deeper, finalize
        if node not in leaves and (depth >= max_depth or len(node.children) == 0):
            leaves.append(node)

    # ---- compute uncertainty ----
    leaf_avg_entropies = [-leaf.cum_logprob / max(1, leaf.length) for leaf in leaves]
    # 加权平均
    # total_prob = sum(leaf.prob for leaf in leaves)
    weighted_entropy = sum(leaf.prob * avg_ent for leaf, avg_ent in zip(leaves, leaf_avg_entropies))
    # 惩罚项
    alpha = 0.0
    complexity = sum(leaf.prob * leaf.depth for leaf in leaves) / max_depth

    # todo
    # penalty = alpha * len(leaves) / max_nodes
    U = weighted_entropy + alpha * complexity
    return root, leaves, U, complexity, max_width


# def collect_split_entropies(node: Node) -> List[float]:
#     vals = []
#     if node.is_split and node.split_entropy is not None:
#         vals.append(node.split_entropy)
#     for ch in node.children:
#         vals.extend(collect_split_entropies(ch))
#     return vals

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


def pretty_print_tree(node: Node, indent: str = "", is_last: bool = True):
    branch = "└─" if is_last else "├─"
    split_info = f" [SPLIT H={node.split_entropy:.2f}]" if node.is_split else ""
    preview = node.text.replace("\n", " ")
    print(f"{indent}{branch} txt:'{preview}'  lp={node.cum_logprob:.2f}{split_info}")
    next_indent = indent + ("   " if is_last else "│  ")
    for i, ch in enumerate(node.children):
        pretty_print_tree(ch, indent=next_indent, is_last=(i == len(node.children) - 1))


# ====== Demo ======
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('sys_prompt.txt', 'r') as f:
        system_prompt = f.read().strip()
    query = '''A student comes home after school. He has three tasks: finish his homework, eat dinner, and play video games. Please reason step by step about what order he should do these things, and make sure to use logical connectors such as therefore, however, but, next.'''
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    for _ in range(3):
        # torch.cuda.manual_seed_all(41)
        root, leaves, U, _ = logic_branch_decode(tokenizer, model, prompt=prompt)

        print("\n--- Logic Tree ---")
        pretty_print_tree(root)

        print("\n--- Leaves ---")
        total_prob = sum(leaf.prob for leaf in leaves)
        for i, leaf in enumerate(leaves):
            txt = leaf.text.replace("\n", " ")
            print(f"[{i:02d}] p={leaf.prob / total_prob:.3f}  text_tail='{txt}'")

        # Uncertainty
        print(f"\nUncertainty score U = {U:.3f}")
        print("(U combines tree leaf entropy, average split entropy, and answer disagreement.)")


if __name__ == "__main__":
    main()
