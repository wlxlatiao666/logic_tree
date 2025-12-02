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


def run_vllm_generate_parallel(model_dir, dataset, data, num_samples, output_file, 
                               tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    """
    使用vLLM进行并行推理，支持多GPU张量并行和批量生成
    
    Args:
        model_dir: 模型路径
        dataset: 数据集名称
        data: 数据列表
        num_samples: 每个问题生成的样本数
        output_file: 输出文件路径
        tensor_parallel_size: 张量并行使用的GPU数量（默认1，即单卡）
        gpu_memory_utilization: GPU显存利用率（0-1之间）
    """
    if not _HAS_VLLM:
        raise RuntimeError("vllm is not installed or failed to import. Install vllm to use this script.")
    
    # 移除CUDA_VISIBLE_DEVICES设置，让vLLM自动管理多GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    
    logger.info(f"初始化vLLM，使用 {tensor_parallel_size} 个GPU进行张量并行")
    
    # 初始化LLM，启用多GPU张量并行
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,  # 多GPU张量并行
        gpu_memory_utilization=gpu_memory_utilization,  # 显存利用率
        max_model_len=32768,  # 最大序列长度
        trust_remote_code=True
    )
    
    # 读取系统提示词
    with open('./sys_prompt.json', 'r') as f:
        sys_prompts = json.load(f)
    sys_prompt = sys_prompts[dataset]
    
    # 准备所有prompts（批量处理）
    logger.info("准备prompts...")
    all_prompts = []
    prompt_to_item_idx = []  # 记录每个prompt对应的数据项索引
    
    for i, item in enumerate(data):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 为每个数据项生成num_samples个prompt（用于批量生成）
        for _ in range(num_samples):
            all_prompts.append(prompt)
            prompt_to_item_idx.append(i)
    
    logger.info(f"总共 {len(data)} 个数据项，每个生成 {num_samples} 个样本，共 {len(all_prompts)} 个prompts")
    
    # 批量生成参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_tokens=32768,
        n=1  # 每个prompt生成1个输出（我们通过重复prompt来生成多个样本）
    )
    
    start_time = time.time()
    logger.info("开始批量生成...")
    
    # 批量生成所有样本
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    
    generation_time = time.time() - start_time
    logger.info(f"批量生成完成，耗时: {generation_time:.2f}秒 ({generation_time/60:.2f}分钟)")
    
    # 组织结果
    logger.info("组织结果...")
    results = [None] * len(data)
    
    for output_idx, output in enumerate(outputs):
        item_idx = prompt_to_item_idx[output_idx]
        
        # 初始化该数据项的结果
        if results[item_idx] is None:
            results[item_idx] = {
                "original_data": data[item_idx],
                "sampled_answers": [],
                "num_tokens": [],
                "predictive_entropy": 0.0,
                "label": 0,
                "passk": 0
            }
        
        # 提取生成结果
        text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        
        results[item_idx]["sampled_answers"].append(text)
        results[item_idx]["num_tokens"].append(len(token_ids))
    
    # 计算每个数据项的统计信息
    logger.info("计算统计信息...")
    for i, result in enumerate(results):
        if result is None:
            logger.warning(f"数据项 {i} 没有生成结果")
            continue
            
        sampled_answers = result["sampled_answers"]
        parsed_answers = [parse_model_answer(ans) for ans in sampled_answers]
        
        # 计算预测熵
        answer_counts = Counter(parsed_answers)
        total_answers = len(parsed_answers)
        predictive_entropy = 0.0
        for count in answer_counts.values():
            p = count / total_answers
            predictive_entropy -= p * math.log(p)
        
        result["predictive_entropy"] = predictive_entropy
        
        # 计算准确率
        most_common_answer, _ = answer_counts.most_common(1)[0]
        gt_answer = get_gt_answer(dataset, data[i])
        label = int(match_answer(gt_answer, most_common_answer, dataset))
        result["label"] = label
        
        # 计算pass@k
        passk = 0
        for answer in parsed_answers:
            if match_answer(gt_answer, answer, dataset):
                passk = 1
                break
        result["passk"] = passk
        
        if (i + 1) % 10 == 0:
            logger.info(f"已处理 {i + 1}/{len(data)} 个数据项")
    
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"\n[运行统计] 总耗时: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    logger.info(f"平均每个数据项: {total_duration/len(data):.2f}秒")
    logger.info(f"平均每个样本: {total_duration/len(all_prompts):.2f}秒")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    
    # 添加控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(model_to_dir.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                       help='使用的GPU数量进行张量并行（默认1）')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help='GPU显存利用率，0-1之间（默认0.9）')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    num_samples = args.samples
    test_size = args.test_size
    tensor_parallel_size = args.tensor_parallel_size
    gpu_memory_utilization = args.gpu_memory_utilization

    model_dir = model_to_dir[model_name]
    data_path = f"./data/{dataset}/test.json"
    
    if test_size == -1:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_parallel_tp{tensor_parallel_size}.json'
    else:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_{test_size}_parallel_tp{tensor_parallel_size}.json'

    with open(data_path, 'r') as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]

    run_vllm_generate_parallel(
        model_dir, 
        dataset, 
        data, 
        num_samples, 
        output_file, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
