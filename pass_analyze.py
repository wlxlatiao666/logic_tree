import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import get_gt_answer, match_answer, parse_model_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    parser.add_argument("--lt_file", type=str, required=True)
    # parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    print("model:", args.model)
    print("dataset:", args.dataset)
    dataset = args.dataset
    sc_file_path = f"./results/{args.model}/{dataset}/{args.sc_file}"
    lt_file_path = f"./results/{args.model}/{dataset}/{args.lt_file}"

    with open(sc_file_path, 'r') as f:
        data_sc = json.load(f)
    with open(lt_file_path, 'r') as f:
        data_lt = json.load(f)

    # 图1 pass@all随总token数变化
    # max_n = args.n
    accuracy_sc_list = []
    accuracy_lt_list = []
    for item_sc, item_lt in zip(data_sc, data_lt):
        original_data_sc = item_sc['original_data']
        original_data_lt = item_lt['original_data']
        sc_texts = item_sc['sampled_answers']
        lt_texts = item_lt['texts']
        gt_answer = get_gt_answer(dataset, original_data_sc)
        correct_flags_sc = [match_answer(gt_answer, parse_model_answer(dataset, ans), dataset) for ans in sc_texts]
        correct_flags_lt = [match_answer(gt_answer, parse_model_answer(dataset, ans), dataset) for ans in lt_texts]
        accuracy_sc = sum(correct_flags_sc) / len(correct_flags_sc) if correct_flags_sc else 0.0
        accuracy_lt = sum(correct_flags_lt) / len(correct_flags_lt) if correct_flags_lt else 0.0
        accuracy_sc_list.append(accuracy_sc)
        accuracy_lt_list.append(accuracy_lt)

    accuracy_sc = np.array(accuracy_sc_list)
    accuracy_lt = np.array(accuracy_lt_list)

    # 1. LT正确率分布
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    print("\n=== LT正确率分布 ===")
    print(f"{'区间':<12} {'SC':>8} {'LT':>8}")
    for i, label in enumerate(labels):
        sc_cnt = np.sum((accuracy_sc >= bins[i]) & (accuracy_sc < bins[i+1]))
        lt_cnt = np.sum((accuracy_lt >= bins[i]) & (accuracy_lt < bins[i+1]))
        print(f"{label:<12} {sc_cnt:>8} {lt_cnt:>8}")

    # 2. SC vs LT 优劣分析
    diff = accuracy_sc - accuracy_lt  # >0: SC优, <0: LT优
    sc_better = diff > 0
    lt_better = diff < 0
    tie = diff == 0
    print(f"\n=== SC vs LT ===")
    print(f"SC优于LT: {sc_better.sum()} 样本, 平均优势: {diff[sc_better].mean():.3f}")
    print(f"LT优于SC: {lt_better.sum()} 样本, 平均优势: {(-diff[lt_better]).mean():.3f}")
    print(f"持平:     {tie.sum()} 样本")

    # 3. 优势与题目难度的相关性（用SC正确率代表难度）
    print(f"\n=== LT优势与题目难度的相关性 ===")
    print(f"(难度 = SC正确率，diff = LT - SC)")
    lt_diff = accuracy_lt - accuracy_sc
    corr = np.corrcoef(accuracy_sc, lt_diff)[0, 1]
    print(f"Pearson相关系数(SC难度 vs LT-SC): {corr:.4f}")

    # 按SC难度分组看LT的优势
    print(f"\n{'SC难度区间':<10} {'样本数':>6} {'LT-SC均值':>8} {'LT胜率':>8} {'SC胜率':>7} {'持平率':>8}")
    for i, label in enumerate(labels):
        mask = (accuracy_sc >= bins[i]) & (accuracy_sc < bins[i+1])
        if mask.sum() == 0:
            continue
        d = lt_diff[mask]
        print(f"{label:<12} {mask.sum():>8} {d.mean():>12.3f} {(d>0).mean():>10.2%} {(d<0).mean():>10.2%} {(d==0).mean():>10.2%}")

    # 4. 特殊情况统计
    print(f"\n=== 特殊情况 ===")
    sc0_lt_pos = np.sum((accuracy_sc == 0) & (accuracy_lt > 0))
    sc100_lt_less = np.sum((accuracy_sc == 1.0) & (accuracy_lt < 1.0))
    print(f"SC=0% 且 LT>0%（LT救回）:    {sc0_lt_pos}")
    print(f"SC=100% 且 LT<100%（LT退步）: {sc100_lt_less}")
    
    lt0_sc_pos = np.sum((accuracy_lt == 0) & (accuracy_sc > 0))
    lt100_sc_less = np.sum((accuracy_lt == 1.0) & (accuracy_sc < 1.0))
    print(f"LT=0% 且 SC>0%（LT救回）:    {lt0_sc_pos}")
    print(f"LT=100% 且 SC<100%（LT退步）: {lt100_sc_less}")
    
    sc0_num = np.sum(accuracy_sc == 0)
    sc100_num = np.sum(accuracy_sc == 1.0)
    print(f"原始SC=0%:   {sc0_num}")
    print(f"原始SC=100%:   {sc100_num}")
    
    lt0_num = np.sum(accuracy_lt == 0)
    lt100_num = np.sum(accuracy_lt == 1.0)
    print(f"LT=0%:   {lt0_num}")
    print(f"LT=100%:   {lt100_num}")
    print('\n\n\n')
