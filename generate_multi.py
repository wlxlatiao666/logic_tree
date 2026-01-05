
# ====== 多GPU all-reduce并行版本 ======
import json
import datetime
import os
import time
import math
import logging
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import sys
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import generate_usr_prompt, parse_model_answer, get_gt_answer, match_answer

logger = logging.getLogger(__name__)

model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B",
    "gpt-oss-20b": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b"
}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=10800))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_inference(rank, world_size, args):
    setup(rank, world_size)

    # 配置日志
    if rank == 0:
        fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    model_name = args.model
    model_dir = model_to_dir[model_name]
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        logger.info(f"使用GPU设备: {rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = args.dataset
    test_size = args.test_size
    num_samples = args.samples
    dataset_path = f"./data/{dataset}/test.json"
    if test_size == -1:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    else:
        with open(dataset_path, "r") as f:
            data = json.load(f)[:test_size]

    # 数据分片
    chunk_size = len(data) // world_size
    remainder = len(data) % world_size
    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size
    local_data = data[start_idx:end_idx]
    local_results = []

    with open('./sys_prompt.json', 'r') as f:
        sys_prompt = json.load(f)[dataset]

    for i, item in enumerate(local_data):
        logger.info(f'Rank {rank} Processing item {start_idx + i}')
        usr_prompt = generate_usr_prompt(dataset, item)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        sampled_answers = []
        # sampled_entropies = []
        num_tokens = []
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.90,
                # output_scores=True,
                # return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_ids = outputs
            generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            sampled_answers.append(generated_text)

            # 计算对数概率
            # neg_logprobs = []
            # input_length = inputs["input_ids"].shape[1]

            # for step_idx, step_scores in enumerate(outputs.scores):
            #     token_id = generated_ids[0, input_length + step_idx]
            #     step_logprobs = F.log_softmax(step_scores[0], dim=-1)
            #     logprob = step_logprobs[token_id].item()
            #     neg_logprobs.append(-logprob)

            # avg_logprob = sum(neg_logprobs) / len(neg_logprobs) if neg_logprobs else 0.0
            # sampled_entropies.append(avg_logprob)
            num_tokens.append(len(generated_ids[0][inputs["input_ids"].shape[-1]:]))

        parsed_answers = [parse_model_answer(ans) for ans in sampled_answers]
        answer_counts = Counter(parsed_answers)
        total_answers = len(parsed_answers)
        predictive_entropy = 0.0
        for count in answer_counts.values():
            p = count / total_answers
            predictive_entropy -= p * math.log(p)
        most_common_answer, _ = answer_counts.most_common(1)[0]
        gt_answer = get_gt_answer(dataset, item)
        label = int(match_answer(gt_answer, most_common_answer, dataset))

        passk = 0
        for answer in parsed_answers:
            if match_answer(gt_answer, answer, dataset):
                passk = 1
                break

        local_results.append({
            "original_data": item,
            "sampled_answers": sampled_answers,
            "num_tokens": num_tokens,
            # "sampled_entropies": sampled_entropies,
            "predictive_entropy": predictive_entropy,
            "label": label,
            "passk": passk,
            "global_index": start_idx + i
        })

    # 保存本地结果
    temp_dir = f"./results/temp/{model_name}/{dataset}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"rank_{rank}.json")
    with open(temp_file, 'w', encoding="utf8") as f:
        json.dump(local_results, f, ensure_ascii=False)

    logger.info(f"Rank {rank} saved {len(local_results)} items to {temp_file}")
    cleanup()

def merge_results(args):
    model_name = args.model
    dataset = args.dataset
    test_size = args.test_size
    num_samples = args.samples
    world_size = args.world_size

    temp_dir = f"./results/temp/{model_name}/{dataset}"
    final_results = []
    for rank in range(world_size):
        temp_file = os.path.join(temp_dir, f"rank_{rank}.json")
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding="utf8") as f:
                final_results.extend(json.load(f))
        else:
            print(f"Warning: Missing result file for rank {rank}")

    final_results.sort(key=lambda x: x["global_index"])
    for result in final_results:
        if "global_index" in result:
            del result["global_index"]

    if test_size == -1:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples.json'
    else:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_{test_size}.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding="utf8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Merged results saved to {output_file}")

if __name__ == '__main__':
    start_time = time.time()

    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="使用的GPU数量")
    args = parser.parse_args()

    mp.spawn(run_inference, args=(args.world_size, args), nprocs=args.world_size, join=True)
    merge_results(args)
    logger.info(f"generate results in {time.time() - start_time:.2f} seconds")
