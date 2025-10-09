from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import sys
import json
import torch
import math
import time
from logic_tree_decode import logic_branch_decode, Node, calculate_diversity
from answer_parser import parse_model_answer, parse_gsm8k_answer

# sys.stdout = open('output.log', 'w', encoding='utf-8')
# set_seed(42)

# def select_min_entropy_path(node: Node) -> Node:
#     current = node
#     while current.children:
#         current = min(current.children, key=lambda x: x.entropy)
#     return current

if __name__ == "__main__":
    start_time = time.time()

    MODEL_NAME = "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # print("eos_token: ", tokenizer.eos_token)

    input_path = '/mnt/public/gpfs-jd/code/weilongxuan/logic_tree/data/gsm8k/test.json'
    output_path = './logic_result_topptopk_100.json'

    with open(input_path, 'r') as f:
        test_data = json.load(f)[:100]

    with open('sys_prompt.txt', 'r') as f:
        system_prompt = f.read().strip()

    results = []
    for i, item in enumerate(test_data):
        # 构建Qwen2.5格式prompt
        question = item['question']
        gt_answer = parse_gsm8k_answer(item['answer'])
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        # 调用解码逻辑
        root, leaves, U, complexity, width = logic_branch_decode(tokenizer, model, prompt=prompt, sample=True, M=3)
            
        uncertainties = {}
        labels = {}

        uncertainties["weighted_leaf_avg_entropy"] = U # 所有叶子结点概率加权
        uncertainties["complexity"] = complexity
        uncertainties["width"] = width

        max_leaf = max(leaves, key=lambda x: x.prob)
        single_entropy = -max_leaf.cum_logprob / max_leaf.length
        uncertainties["max_leaf_avg_entropy"] = float(single_entropy) # 取概率最大的叶子结点

        uncertainties["avg_leaf_avg_entropy"] = sum(-leaf.cum_logprob / leaf.length for leaf in leaves) / len(leaves) # 所有叶子结点平均

        answer_buckets = {}
        for leaf in leaves:
            parsed_answer = parse_model_answer(leaf.text)
            if parsed_answer:
                try: 
                    key = parsed_answer
                    answer_buckets[parsed_answer] = answer_buckets.get(parsed_answer, 0.0) + leaf.prob
                except:
                    continue
        total = sum(answer_buckets.values())
        entropy = 0.0
        if total > 0:
            probabilities = [p/total for p in answer_buckets.values()]
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        uncertainties["predictive_entropy_weighted"] = float(entropy)

        probs_entropy = -sum(leaf.prob * math.log(leaf.prob) for leaf in leaves if leaf.prob > 0)
        uncertainties["probs_entropy"] = float(probs_entropy)
        
        final_answer_vote_weighted = max(answer_buckets.items(), key=lambda x: x[1]) if answer_buckets else ""
        final_answer_vote_weighted_nothreshold = final_answer_vote_weighted[0]
        labels["answer_vote_weighted_nothreshold"] = int(final_answer_vote_weighted_nothreshold == gt_answer) if gt_answer else 0

        final_answer_vote_weighted_threshold = final_answer_vote_weighted[0] if final_answer_vote_weighted[1] > 0.5 else ""
        labels["answer_vote_weighted_threshold"] = int(final_answer_vote_weighted_threshold == gt_answer) if gt_answer else 0

        min_entropy_leaf = min(leaves, key=lambda x: -x.cum_logprob / x.length)
        final_answer_min_entropy = parse_model_answer(min_entropy_leaf.text)
        labels["answer_min_entropy"] = int(final_answer_min_entropy == gt_answer) if gt_answer else 0
        
        # 获取贪婪解码结果（取第一个leaf）
        # inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # greedy_output = model.generate(
        #     inputs.input_ids,
        #     max_length=1024,
        #     num_beams=1,
        #     do_sample=False
        # )
        # greedy_answer = tokenizer.decode(greedy_output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # labels["greedy"] = int(parse_model_answer(greedy_answer) == gt_answer) if gt_answer else 0
        # topp_topk_output = model.generate(
        #     inputs.input_ids,
        #     max_length=1024,
        #     do_sample=True,
        #     top_p=0.9,
        #     top_k=50,
        #     temperature=0.7
        # )
        # topp_topk_answer = tokenizer.decode(topp_topk_output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # labels["topp_topk"] = int(parse_model_answer(topp_topk_answer) == gt_answer) if gt_answer else 0
            
        passk = False
        for answer in answer_buckets:
            if answer == gt_answer:
                passk = True
                break
        labels["pass@k"] = int(passk)
        # 记录结果
        # uncertainties = [-leaf.cum_logprob / leaf.length for leaf in leaves]
        # labels = [int(parse_model_answer(leaf.text) == parse_gsm8k_answer(item['answer'])) for leaf in leaves]

        diversity = calculate_diversity(leaves)

        results.append({
            "question": question,
            "gt_answer": gt_answer,
            "label": labels,
            "uncertainty": uncertainties,
            "diversity": diversity
        })
        print(f"Processed {i+1}/100 | U={U:.2f} | diversity={diversity:.3f}")
        # print(results[i])

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_path}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")