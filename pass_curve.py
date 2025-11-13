import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from utils import get_gt_answer, match_answer, parse_model_answer
=======
from utils import get_gt_answer, match_answer
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
<<<<<<< HEAD
    parser.add_argument("--lt_file", type=str, required=True)
=======
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6
    args = parser.parse_args()

    dataset = args.dataset
    sc_file_path = f"./results/{dataset}/{args.sc_file}"
<<<<<<< HEAD
    lt_file_path = f"./results/{dataset}/{args.lt_file}"

    with open(lt_file_path, 'r') as f:
        data_lt = json.load(f)
    with open(sc_file_path, 'r') as f:
        data_sc = json.load(f)[:len(data_lt)]
=======

    with open(sc_file_path, 'r') as f:
        data = json.load(f)
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6

    max_n = 20
    pass_rates = []
    avg_token_counts = []

    for n in range(1, max_n + 1):
        passes = []
        token_usages = []
<<<<<<< HEAD
        for item in data_sc:
            answers = item['sampled_answers'][:n]   
=======
        for item in data:
            answers = item['sampled_answers'][:n]
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6
            original_data = item['original_data']
            gt_answer = get_gt_answer(dataset, original_data)
            # 假设有token_counts字段，统计前n个response的token数
            token_counts = item['num_tokens'][:n]
            # 判断是否有一个response等于gt_answer（可自定义等价判断）
<<<<<<< HEAD
            is_pass = any(match_answer(gt_answer, parse_model_answer(ans), dataset) for ans in answers)
=======
            is_pass = any(match_answer(gt_answer, ans, dataset) for ans in answers)
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6
            passes.append(is_pass)
            token_usages.append(sum(token_counts))
        pass_rate = np.mean(passes)
        avg_token = np.mean(token_usages)
        pass_rates.append(pass_rate)
        avg_token_counts.append(avg_token)

<<<<<<< HEAD
    lt_passes = [item['label']['pass@k'] for item in data_lt]
    lt_tokens = [item['num_new_tokens'] for item in data_lt]
    lt_pass_rate = np.mean(lt_passes)
    lt_avg_tokens = np.mean(lt_tokens)

    plt.plot(avg_token_counts, pass_rates, marker='o', label='Sampling')
    plt.scatter([lt_avg_tokens], [lt_pass_rate], color='red', marker='*', s=150, label='Logic Tree')
=======
    plt.plot(avg_token_counts, pass_rates, marker='o')
>>>>>>> e61740ae5bbb3ae8ab718ce3a51b961001ad88d6
    plt.xlabel('Average tokens per response')
    plt.ylabel('Pass rate')
    plt.title('Pass rate vs. average tokens per response')
    plt.grid(True)
    plt.show()