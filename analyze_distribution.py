import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from utils import parse_model_answer

def get_answer_distribution(json_path, weighted=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    answer_counter = defaultdict(float)
    for item in data:
        texts = item.get('texts', [])
        if weighted and 'probs' in item:
            probs = item['probs']
            for t, p in zip(texts, probs):
                ans = parse_model_answer(t)
                answer_counter[ans] += float(p)
        else:
            for t in texts:
                ans = parse_model_answer(t)
                answer_counter[ans] += 1.0
    return answer_counter

def plot_distribution(counter, title):
    labels, counts = zip(*sorted(counter.items()))
    plt.bar(labels, counts)
    plt.xlabel('Answer')
    plt.ylabel('Weighted Count')
    plt.title(title)
    plt.show()

def kl_divergence(p_counter, q_counter, smooth=1e-8):
    all_keys = set(p_counter.keys()) | set(q_counter.keys())
    p = np.array([p_counter.get(k, 0) for k in all_keys], dtype=float)
    q = np.array([q_counter.get(k, 0) for k in all_keys], dtype=float)
    p = (p + smooth) / (p.sum() + smooth * len(p))
    q = (q + smooth) / (q.sum() + smooth * len(q))
    kl = np.sum(p * np.log(p / q))
    return kl

index = 0
with open('/Users/weilongxuan/codes/logic_tree/results/reclor/generated_answers_50samples_3.json', 'r') as f:
    data_sc = json.load(f)[index:index+1]
with open('./results/reclor/logic_tree_results_10repeats.json', 'r') as f:
    data_lt = json.load(f)[index:index+1]
    
for item_sc, item_lt in zip(data_sc, data_lt):
    answer_counter_sc = defaultdict(float)
    answer_counter_lt = defaultdict(float)
    texts_sc = item_sc.get('sampled_answers', [])
    for t in texts_sc[:20]:
        ans = parse_model_answer(t)
        answer_counter_sc[ans] += 1.0
    texts_lt = item_lt.get('texts', [])
    probs_lt = item_lt.get('probs', [])
    for t, p in zip(texts_lt, probs_lt):
        ans = parse_model_answer(t)
        answer_counter_lt[ans] += float(p)

    # plot_distribution(answer_counter_sc, 'Distribution from generated_answers_50samples_3.json')
    # plot_distribution(answer_counter_lt, 'Weighted Distribution from logic_tree_results_100.json')
    print(answer_counter_sc)
    print(answer_counter_lt)

    # kl_value = kl_divergence(answer_counter_sc, answer_counter_lt)
    # print(f'KL散度（generated_answers_50samples_3.json || logic_tree_results_100.json，probs加权）: {kl_value:.4f}')