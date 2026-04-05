"""
分析采样准确率与题目难度的关系。
难度 = 1 - pass_rate（两种方法合并计算的通过率）
对比 tree 采样 vs 随机采样在不同难度区间的 pass@k 和 majority vote 准确率。
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
# from math_judger import MathJudger
# from equivalence import is_equiv_math, is_equiv_scibench
from utils import parse_model_answer, get_gt_answer, match_answer
from concurrent.futures import ThreadPoolExecutor

dataset = "math500"
TREE_FILE = f"results/Qwen2.5-7B-Instruct/{dataset}/logic_tree_results_all_leaves20_threshold80_ddp.json"
SC_FILE = f"results/Qwen2.5-7B-Instruct/{dataset}/generated_answers_20samples_ray_8gpus.json"
N_BINS = 5

# judger = MathJudger()

def extract_answer(text):
    # 提取 <answer>...</answer> 或 \boxed{} 中的内容
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text

def is_correct(pred_text, gt):
    try:
        pred = parse_model_answer(pred_text)
        return match_answer(gt, pred, dataset)
    except:
        return False

def evaluate_item(item):
    gt = get_gt_answer("math500", item['original_data'])
    texts = item["texts"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: is_correct(t, gt), texts))
    return results
def evaluate_item_2(item):
    # gt = item["original_data"]["answer"]
    gt = get_gt_answer("math500", item['original_data'])

    texts = item["sampled_answers"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: is_correct(t, gt), texts))
    return results

print("Loading data...")
with open(TREE_FILE) as f:
    tree_data = json.load(f)
with open(SC_FILE) as f:
    sc_data = json.load(f)

assert len(tree_data) == len(sc_data), "数据条数不一致"
N = len(tree_data)

print(f"Evaluating {N} questions...")
tree_correct = []  # list of list[bool]
sc_correct = []

for i, (t_item, s_item) in enumerate(zip(tree_data, sc_data)):
    if i % 50 == 0:
        print(f"  {i}/{N}")
    tree_correct.append(evaluate_item(t_item))
    sc_correct.append(evaluate_item_2(s_item))

# 用两种方法合并的通过率作为难度基准
combined_pass = [
    (sum(t) + sum(s)) / (len(t) + len(s))
    for t, s in zip(tree_correct, sc_correct)
]
difficulty = [1 - p for p in combined_pass]  # 难度 = 1 - 通过率

# 按难度排序后等分为 N_BINS 组（每组大小相等）
sorted_idx = np.argsort(difficulty)
groups = np.array_split(sorted_idx, N_BINS)

def bin_label(indices):
    d_vals = [difficulty[i] for i in indices]
    return f"{min(d_vals):.2f}-{max(d_vals):.2f}"

bin_labels = [bin_label(g) for g in groups]

tree_passk_by_bin = [[int(any(tree_correct[i])) for i in g] for g in groups]
sc_passk_by_bin   = [[int(any(sc_correct[i]))   for i in g] for g in groups]
tree_mv_by_bin    = [[int(sum(tree_correct[i]) > len(tree_correct[i]) / 2) for i in g] for g in groups]
sc_mv_by_bin      = [[int(sum(sc_correct[i])   > len(sc_correct[i])   / 2) for i in g] for g in groups]

print("\n=== 结果（按难度分组）===")
print(f"{'难度区间':<20} {'样本数':>6} {'Tree pass@k':>12} {'SC pass@k':>10} {'Tree MV':>10} {'SC MV':>8}")
for b in range(N_BINS):
    n = len(tree_passk_by_bin[b])
    t_pk = np.mean(tree_passk_by_bin[b])
    s_pk = np.mean(sc_passk_by_bin[b])
    t_mv = np.mean(tree_mv_by_bin[b])
    s_mv = np.mean(sc_mv_by_bin[b])
    print(f"{bin_labels[b]:<20} {n:>6} {t_pk:>12.3f} {s_pk:>10.3f} {t_mv:>10.3f} {s_mv:>8.3f}")

# 相关性分析
tree_passk_per_q = [int(any(r)) for r in tree_correct]
sc_passk_per_q = [int(any(r)) for r in sc_correct]
tree_mv_per_q = [int(sum(r) > len(r) / 2) for r in tree_correct]
sc_mv_per_q = [int(sum(r) > len(r) / 2) for r in sc_correct]

print("\n=== 与难度的 Pearson 相关系数 ===")
print(f"Tree pass@k vs difficulty: {np.corrcoef(difficulty, tree_passk_per_q)[0,1]:.4f}")
print(f"SC   pass@k vs difficulty: {np.corrcoef(difficulty, sc_passk_per_q)[0,1]:.4f}")
print(f"Tree MV     vs difficulty: {np.corrcoef(difficulty, tree_mv_per_q)[0,1]:.4f}")
print(f"SC   MV     vs difficulty: {np.corrcoef(difficulty, sc_mv_per_q)[0,1]:.4f}")

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(N_BINS)

for ax, (t_data, s_data, title) in zip(axes, [
    (tree_passk_by_bin, sc_passk_by_bin, "pass@k"),
    (tree_mv_by_bin, sc_mv_by_bin, "Majority Vote"),
]):
    t_vals = [np.mean(t_data[b]) for b in range(N_BINS)]
    s_vals = [np.mean(s_data[b]) for b in range(N_BINS)]
    w = 0.35
    ax.bar(x - w/2, t_vals, w, label="Tree Sampling")
    ax.bar(x + w/2, s_vals, w, label="Random Sampling")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=15)
    ax.set_xlabel("Difficulty (1 - pass rate)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Difficulty ({title})")
    ax.legend()
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("results/difficulty_vs_accuracy.png", dpi=150)
print("\nPlot saved to results/difficulty_vs_accuracy.png")
