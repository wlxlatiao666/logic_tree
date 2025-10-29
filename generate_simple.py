import json
import time
import argparse
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_usr_prompt, parse_gsm8k_answer, parse_model_answer
import sys

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        filename='./logs/app.log',  # 使用绝对路径
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    sample = args.sample

    model_name = '/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = f"./data/{dataset}/test.json"
    if sample:
        output_file = f'./results/{dataset}/generated_answers_topk_topp.json'
    else:
        output_file = f'./results/{dataset}/generated_answers_greedy.json'

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('./sys_prompt.json', 'r') as f:
        sys_prompt = json.load(f)[dataset]

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    results = []
    start_time = time.time()
    for item in data:
        usr_prompt = generate_usr_prompt(dataset, item)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        if sample:
            output = model.generate(
                **inputs,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.90,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            output = model.generate(
                **inputs,
                max_length=1024,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = output.sequences
        generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            # 计算对数概率
        neg_logprobs = []
        input_length = inputs["input_ids"].shape[1]

        for step_idx, step_scores in enumerate(output.scores):
            # 获取当前步骤生成的token ID
            token_id = generated_ids[0, input_length + step_idx]
            
            # 计算对数概率
            step_logprobs = F.log_softmax(step_scores[0], dim=-1)
            logprob = step_logprobs[token_id].item()
            neg_logprobs.append(-logprob)

        # 计算平均对数概率
        avg_logprob = sum(neg_logprobs) / len(neg_logprobs) if neg_logprobs else 0.0

        if dataset == "gsm8k":
            label = int(parse_gsm8k_answer(item["answer"]) == parse_model_answer(generated_text))
        elif dataset == "reclor":
            label_to_answer = {
                0: "A",
                1: "B",
                2: "C",
                3: "D",
            }
            gt = label_to_answer[item["label"]]
            label = int(gt == parse_model_answer(generated_text))
        else:
            label = 0

        results.append({
            "original_data": item,
            "model_answer": generated_text,
            "label": label,
            "avg_logprob": avg_logprob
        })
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
