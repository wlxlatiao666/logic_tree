import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import get_gt_answer, match_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    sc_file_path = f"./results/{dataset}/{args.sc_file}"

    with open(sc_file_path, 'r') as f:
        data = json.load(f)

    max_n = 20
    pass_rates = []
    avg_token_counts = []

    for n in range(1, max_n + 1):
        passes = []
        token_usages = []
        for item in data:
            answers = item['sampled_answers'][:n]
            original_data = item['original_data']
            gt_answer = get_gt_answer(dataset, original_data)
            # 假设有token_counts字段，统计前n个response的token数
            token_counts = item['num_tokens'][:n]
            # 判断是否有一个response等于gt_answer（可自定义等价判断）
            is_pass = any(match_answer(gt_answer, ans, dataset) for ans in answers)
            passes.append(is_pass)
            token_usages.append(sum(token_counts))
        pass_rate = np.mean(passes)
        avg_token = np.mean(token_usages)
        pass_rates.append(pass_rate)
        avg_token_counts.append(avg_token)

    plt.plot(avg_token_counts, pass_rates, marker='o')
    plt.xlabel('Average tokens per response')
    plt.ylabel('Pass rate')
    plt.title('Pass rate vs. average tokens per response')
    plt.grid(True)
    plt.show()