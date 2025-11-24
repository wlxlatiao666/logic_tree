import json
import time
import math
import logging
from datetime import datetime
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import sys
import argparse
from utils import generate_usr_prompt, parse_model_answer, get_gt_answer, match_answer

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()
    dataset = args.dataset
    num_samples = args.samples
    test_size = args.test_size

    model_name = '/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = f"./data/{dataset}/test.json"
    if test_size == -1:
        output_file = f'./results/{dataset}/generated_answers_{num_samples}samples.json'
    else:
        output_file = f'./results/{dataset}/generated_answers_{num_samples}samples_{test_size}.json'
        
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('./sys_prompt.json', 'r') as f:
        sys_prompt = json.load(f)[dataset]

    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]

    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        logger.info(f'Processing item {i}')
        usr_prompt = generate_usr_prompt(dataset, item)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        sampled_answers = []
        sampled_entropies = []
        num_tokens = []
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.90,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_ids = outputs.sequences
            generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            sampled_answers.append(generated_text)

            # 计算对数概率
            neg_logprobs = []
            input_length = inputs["input_ids"].shape[1]

            for step_idx, step_scores in enumerate(outputs.scores):
                # 获取当前步骤生成的token ID
                token_id = generated_ids[0, input_length + step_idx]
                
                # 计算对数概率
                step_logprobs = F.log_softmax(step_scores[0], dim=-1)
                logprob = step_logprobs[token_id].item()
                neg_logprobs.append(-logprob)

            # 计算平均对数概率
            avg_logprob = sum(neg_logprobs) / len(neg_logprobs) if neg_logprobs else 0.0

            sampled_entropies.append(avg_logprob)
            num_tokens.append(len(generated_ids[0][inputs["input_ids"].shape[-1]:]))

        parsed_answers = [parse_model_answer(ans) for ans in sampled_answers]
        answer_counts = Counter(parsed_answers)
        total_answers = len(parsed_answers)
        predictive_entropy = 0.0
        for count in answer_counts.values():
            p = count / total_answers
            predictive_entropy -= p * math.log(p)
        # 找出出现次数最多的答案
        most_common_answer, _ = answer_counts.most_common(1)[0]
        gt_answer = get_gt_answer(dataset, item)
        label = int(match_answer(gt_answer, most_common_answer, dataset))

        passk = 0
        for answer in parsed_answers:
            if match_answer(gt_answer, answer, dataset):
                passk = 1
                break

        results.append({
            "original_data": item,
            "sampled_answers": sampled_answers,
            "num_tokens": num_tokens,
            "sampled_entropies": sampled_entropies,
            "predictive_entropy": predictive_entropy,
            "label": label,
            "passk": passk
        })
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")

    # 保存结果
    with open(output_file, 'w', encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
