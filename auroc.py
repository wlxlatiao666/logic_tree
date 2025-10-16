
import argparse
import json
from sklearn.metrics import roc_auc_score
from utils import parse_gsm8k_answer, parse_model_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True, help='Main result json, e.g. logic_tree_results_all.json')
    args = parser.parse_args()

    with open(args.result_file, 'r') as f:
        results = json.load(f)
    with open('/Users/weilongxuan/codes/logic_tree/results/gsm8k/generated_answers_greedy.json', 'r') as f:
        greedy = json.load(f)

    if len(results) != len(greedy):
        print(f"Warning: result file and greedy file have different lengths ({len(results)} vs {len(greedy)}), will align by index.")

    scores = []
    labels = []
    for i, (item, gitem) in enumerate(zip(results, greedy)):
        score = -float(item["avg_branching_factor"])
        correct = int(parse_model_answer(gitem['greedy_answer']) == parse_gsm8k_answer(gitem['gt_answer']))
        scores.append(score)
        labels.append(correct)

    auroc = roc_auc_score(labels, scores)
    print(f"AUROC: {auroc:.4f}")

if __name__ == '__main__':
    main()
