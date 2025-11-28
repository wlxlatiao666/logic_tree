import json
import os
import time
import math
import logging
from datetime import datetime
from collections import Counter
import argparse
import numpy as np

from utils import generate_usr_prompt, parse_model_answer, get_gt_answer, match_answer

logger = logging.getLogger(__name__)

model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B"
}

# vllm import is optional at import-time: handle gracefully if not installed
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


def run_vllm_generate(model_dir, dataset, data, num_samples, output_file, device: int = 0):
    if not _HAS_VLLM:
        raise RuntimeError("vllm is not installed or failed to import. Install vllm to use this script.")

    llm = LLM(model=model_dir, device=f"cuda:{device}")

    results = []
    start_time = time.time()

    for i, item in enumerate(data):
        logger.info(f"Processing item {i}")
        usr_prompt = generate_usr_prompt(dataset, item)
        # Build chat template using existing tokenizer logic; utils should produce the same prompt text
        # We'll assume generate_usr_prompt returns the full user message string and sys prompt will be prepended
        # Read system prompt
        with open('./sys_prompt.json', 'r') as f:
            sys_prompts = json.load(f)
        sys_prompt = sys_prompts[dataset]

        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        sampled_answers = []
        sampled_entropies = []
        num_tokens = []

        # vllm can generate multiple samples by running generation separately for each sample
        for s in range(num_samples):
            sampling_params = SamplingParams(
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                max_tokens=1024,
                logits_processor=None,
            )

            out = llm.generate(prompt, sampling_params=sampling_params, verbose=False)
            # out is an iterator of Generation results; get first (and only) result
            generation = next(out)
            text = generation.outputs[0].text
            # vllm returns tokens and token logits per token in .outputs[0].token_ids and .raw_logits
            token_ids = generation.outputs[0].token_ids
            raw_logits = generation.outputs[0].raw_logits

            # compute negative logprobs per token
            neg_logprobs = []
            for logits, tid in zip(raw_logits, token_ids):
                probs = np.exp(np.asarray(logits))
                probs = probs / probs.sum()
                prob = probs[tid]
                neg_logprobs.append(-math.log(max(prob, 1e-12)))

            avg_logprob = sum(neg_logprobs) / len(neg_logprobs) if neg_logprobs else 0.0
            sampled_entropies.append(avg_logprob)
            num_tokens.append(len(token_ids))

            sampled_answers.append(text)

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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(model_to_dir.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", type=int, default=0, help='GPU id (int). Use -1 for CPU')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    num_samples = args.samples
    test_size = args.test_size
    device = args.device

    model_dir = model_to_dir[model_name]
    data_path = f"./data/{dataset}/test.json"
    if test_size == -1:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples.json'
    else:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_{test_size}.json'

    with open(data_path, 'r') as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]

    run_vllm_generate(model_dir, dataset, data, num_samples, output_file, device=device)
