import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import get_gt_answer, match_answer, parse_model_answer

def calculate_pass_at_k(correct_flags, k):
    """
    计算pass@k值
    correct_flags: 布尔值列表，表示每个response是否正确
    k: 选择的response数量
    """
    n = len(correct_flags)
    if n == 0:
        return 0.0
    
    # 如果k >= n，则pass@k就是是否至少有一个正确
    if k >= n:
        return 1.0 if any(correct_flags) else 0.0
    
    # 计算所有可能的错误组合数（从错误的response中选k个）
    num_incorrect = sum(1 for correct in correct_flags if not correct)
    if num_incorrect < k:
        # 如果错误的数量小于k，那么无论怎么选都至少有一个正确
        return 1.0
    
    # 计算从错误的response中选k个的组合数
    incorrect_combinations = math.comb(num_incorrect, k)
    # 计算总的组合数
    total_combinations = math.comb(n, k)
    # pass@k = 1 - (错误的组合数 / 总组合数)
    return 1.0 - (incorrect_combinations / total_combinations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    parser.add_argument("--lt_file", type=str, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    print("model:", args.model)
    print("dataset:", args.dataset)
    dataset = args.dataset
    sc_file_path = f"./results/{args.model}/{dataset}/{args.sc_file}"
    lt_file_path = f"./results/{args.model}/{dataset}/{args.lt_file}"

    with open(lt_file_path, 'r') as f:
        data_lt = json.load(f)
    with open(sc_file_path, 'r') as f:
        data_sc = json.load(f)[:len(data_lt)]

    # 图1 pass@all随总token数变化
    max_n = args.n
    pass_rates = []
    avg_token_counts = []

    for n in range(1, max_n + 1):
        passes = []
        token_usages = []
        for item in data_sc:
            answers = item['sampled_answers'][:n]   
            original_data = item['original_data']
            gt_answer = get_gt_answer(dataset, original_data)
            # 假设有token_counts字段，统计前n个response的token数
            token_counts = item['num_tokens'][:n]
            # 判断是否有一个response等于gt_answer（可自定义等价判断）
            is_pass = any(match_answer(gt_answer, parse_model_answer(ans), dataset) for ans in answers)
            passes.append(is_pass)
            token_usages.append(sum(token_counts))
        pass_rate = np.mean(passes)
        avg_token = np.mean(token_usages)
        pass_rates.append(pass_rate)
        avg_token_counts.append(avg_token)

    lt_passes = []
    for item in data_lt:
        original_data = item['original_data']
        gt_answer = get_gt_answer(dataset, original_data)
        is_pass = any(match_answer(gt_answer, parse_model_answer(ans), dataset) for ans in item['texts'])
        lt_passes.append(is_pass)
    lt_tokens = [item['num_new_tokens'] for item in data_lt]
    lt_pass_rate = np.mean(lt_passes)
    lt_avg_tokens = np.mean(lt_tokens)

    plt.figure(1)
    plt.plot(avg_token_counts, pass_rates, marker='o', label='Multi-chain')
    plt.scatter([lt_avg_tokens], [lt_pass_rate], color='red', marker='*', s=150, label='Entropy-Tree')
    plt.xlabel('Total tokens per question')
    plt.ylabel('Pass@all')
    plt.title('Pass@all vs. total tokens per question')
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(f"./results/{args.model}/{dataset}/pass_at_all_{max_n}_avgdis.png")
    plt.close(1)

    # 图2 pass@k随k的变化
    sc_pass_at_k = []
    lt_pass_at_k = []

    # 首先预处理数据，计算每个item的正确标志列表
    sc_correct_flags_list = []
    for item in data_sc:
        original_data = item['original_data']
        gt_answer = get_gt_answer(dataset, original_data)
        # 计算每个sampled_answer是否正确
        correct_flags = [match_answer(gt_answer, parse_model_answer(ans), dataset) for ans in item['sampled_answers'][:max_n]]
        sc_correct_flags_list.append(correct_flags)

    # 预处理Logic Tree数据
    lt_correct_flags_list = []
    for item in data_lt:
        original_data = item['original_data']
        gt_answer = get_gt_answer(dataset, original_data)
        # 计算每个text是否正确
        correct_flags = [match_answer(gt_answer, parse_model_answer(text), dataset) for text in item['texts']]
        lt_correct_flags_list.append(correct_flags)

    # 计算Sampling的pass@k
    for k in range(1, max_n + 1):
        # 对每个item计算pass@k，然后求平均
        item_pass_at_k = []
        for correct_flags in sc_correct_flags_list:
            pass_at_k = calculate_pass_at_k(correct_flags, k)
            item_pass_at_k.append(pass_at_k)
            # print("sc:",pass_at_k)
        avg_pass_at_k = np.mean(item_pass_at_k)
        sc_pass_at_k.append(avg_pass_at_k)
        print(f"SC pass@{k}:{avg_pass_at_k}")

    # 计算Logic Tree的pass@k
    for k in range(1, max_n + 1):
        # 对每个item计算pass@k，然后求平均
        item_pass_at_k = []
        for correct_flags in lt_correct_flags_list:
            pass_at_k = calculate_pass_at_k(correct_flags, k)
            item_pass_at_k.append(pass_at_k)
            # print("lt:",pass_at_k)
        avg_pass_at_k = np.mean(item_pass_at_k)
        lt_pass_at_k.append(avg_pass_at_k)
        print(f"Entropy-Tree pass@{k}:{avg_pass_at_k}")
    print('\n\n')

    # 绘制图2
    plt.figure(2)
    k_values = list(range(1, max_n + 1))
    plt.plot(k_values, sc_pass_at_k[:max_n], marker='o', label='Multi-chain')
    plt.plot(k_values, lt_pass_at_k, marker='s', label='Entropy-Tree')
    # print(sc_pass_at_k)
    # print(lt_pass_at_k)
    plt.xlabel('k')
    plt.ylabel('Pass@k')
    plt.title('Pass@k vs. k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(f"./results/{args.model}/{dataset}/pass_at_k_{max_n}_avgdis.png")
    plt.close(2)
    
    # 图3 token总数随采样数n的变化
    # plt.figure(3)
    # # Sampling的token总数曲线（使用之前计算的avg_token_counts）
    # n_values = list(range(1, max_n + 1))
    # plt.plot(n_values, avg_token_counts, marker='o', label='Sampling')
    # # Logic Tree的token总数点
    # # 找到与Logic Tree平均token数最接近的n值，作为点的x坐标
    # # 这里我们可以使用max_n作为点的x坐标，或者使用其他合适的值
    # lt_n_value = max_k  # 或者选择其他合适的值作为x坐标
    # plt.scatter([lt_n_value], [lt_avg_tokens], color='red', marker='*', s=150, label='Logic Tree')
    # plt.xlabel('Sampling number n')
    # plt.ylabel('Total tokens')
    # plt.title('Total tokens vs. sampling number n')
    # plt.grid(True)
    # plt.legend()
    
    # plt.figure(4)
    # plt.plot(avg_token_counts, sc_pass_at_k, marker='o', label='Sampling')
    # plt.scatter([lt_avg_tokens], [lt_pass_at_k[-1]], color='red', marker='*', s=150, label='Logic Tree')
    # plt.xlabel('Total tokens per question')
    # plt.ylabel('Pass@k')
    # plt.title('Pass@k vs. total tokens per question')
    # plt.grid(True)
    # plt.legend()

    # # 显示所有图表
    # plt.tight_layout()
    # plt.show()