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

# vllm and ray imports
try:
    from vllm import LLM, SamplingParams
    import ray
    _HAS_VLLM_RAY = True
except Exception as e:
    _HAS_VLLM_RAY = False
    print(f"Warning: vllm or ray not available: {e}")


@ray.remote(num_gpus=1)
class VLLMWorker:
    """
    Ray worker，每个worker在一个GPU上运行独立的vLLM实例
    """
    def __init__(self, model_dir: str, gpu_id: int):
        self.gpu_id = gpu_id
        # 设置当前worker只使用指定的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 初始化vLLM
        self.llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,  # 每个worker使用单GPU
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            trust_remote_code=True
        )
        print(f"Worker initialized on GPU {gpu_id}")
    
    def generate(self, prompts, num_samples, sampling_params_dict):
        """
        在当前GPU上生成样本
        
        Args:
            prompts: prompt列表
            num_samples: 每个prompt生成的样本数
            sampling_params_dict: 采样参数字典
        
        Returns:
            生成结果列表
        """
        # 为每个prompt重复num_samples次以生成多个样本
        all_prompts = []
        prompt_to_idx = []
        
        for idx, prompt in enumerate(prompts):
            for _ in range(num_samples):
                all_prompts.append(prompt)
                prompt_to_idx.append(idx)
        
        # 创建采样参数
        sampling_params = SamplingParams(**sampling_params_dict)
        
        # 批量生成
        outputs = self.llm.generate(all_prompts, sampling_params=sampling_params)
        
        # 组织结果
        results = [{"texts": [], "token_counts": []} for _ in range(len(prompts))]
        
        for output_idx, output in enumerate(outputs):
            prompt_idx = prompt_to_idx[output_idx]
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            
            results[prompt_idx]["texts"].append(text)
            results[prompt_idx]["token_counts"].append(len(token_ids))
        
        return results


def run_vllm_generate_ray(model_dir, dataset, data, num_samples, output_file, num_gpus: int = 1):
    """
    使用Ray进行数据并行推理，每个GPU运行独立的vLLM实例
    
    Args:
        model_dir: 模型路径
        dataset: 数据集名称
        data: 数据列表
        num_samples: 每个问题生成的样本数
        output_file: 输出文件路径
        num_gpus: 使用的GPU数量
    """
    if not _HAS_VLLM_RAY:
        raise RuntimeError("vllm or ray is not installed. Install them to use this script.")
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus)
    
    logger.info(f"使用Ray进行数据并行，{num_gpus} 个GPU")
    
    # 读取系统提示词
    with open('./sys_prompt.json', 'r') as f:
        sys_prompts = json.load(f)
    sys_prompt = sys_prompts[dataset]
    
    # 准备所有prompts
    logger.info("准备prompts...")
    all_prompts = []
    for item in data:
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
        all_prompts.append(prompt)
    
    # 数据分片
    chunk_size = len(all_prompts) // num_gpus
    remainder = len(all_prompts) % num_gpus
    
    prompt_chunks = []
    data_chunks = []
    start_idx = 0
    
    for i in range(num_gpus):
        # 前remainder个worker多处理一个样本
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        prompt_chunks.append(all_prompts[start_idx:end_idx])
        data_chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    logger.info(f"数据分片完成: {[len(chunk) for chunk in prompt_chunks]}")
    
    # 创建workers
    logger.info("初始化workers...")
    workers = [VLLMWorker.remote(model_dir, gpu_id=i) for i in range(num_gpus)]
    
    # 采样参数
    sampling_params_dict = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_tokens": 32768,
        "n": 1
    }
    
    start_time = time.time()
    logger.info("开始并行生成...")
    
    # 并行生成
    futures = [
        workers[i].generate.remote(prompt_chunks[i], num_samples, sampling_params_dict)
        for i in range(num_gpus)
    ]
    
    # 等待所有worker完成
    all_results = ray.get(futures)
    
    generation_time = time.time() - start_time
    logger.info(f"并行生成完成，耗时: {generation_time:.2f}秒 ({generation_time/60:.2f}分钟)")
    
    # 合并结果
    logger.info("合并结果...")
    results = []
    
    for worker_idx, worker_results in enumerate(all_results):
        worker_data = data_chunks[worker_idx]
        
        for item_idx, item_result in enumerate(worker_results):
            item = worker_data[item_idx]
            sampled_answers = item_result["texts"]
            num_tokens = item_result["token_counts"]
            
            # 计算统计信息
            parsed_answers = [parse_model_answer(ans) for ans in sampled_answers]
            answer_counts = Counter(parsed_answers)
            total_answers = len(parsed_answers)
            
            # 预测熵
            predictive_entropy = 0.0
            for count in answer_counts.values():
                p = count / total_answers
                predictive_entropy -= p * math.log(p)
            
            # 准确率
            most_common_answer, _ = answer_counts.most_common(1)[0]
            gt_answer = get_gt_answer(dataset, item)
            label = int(match_answer(gt_answer, most_common_answer, dataset))
            
            # pass@k
            passk = 0
            for answer in parsed_answers:
                if match_answer(gt_answer, answer, dataset):
                    passk = 1
                    break
            
            results.append({
                "original_data": item,
                "sampled_answers": sampled_answers,
                "num_tokens": num_tokens,
                "predictive_entropy": predictive_entropy,
                "label": label,
                "passk": passk
            })
            logger.info(f"Processed {item_idx+1} items")
    
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"\n[运行统计] 总耗时: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    logger.info(f"平均每个数据项: {total_duration/len(data):.2f}秒")
    logger.info(f"平均每个样本: {total_duration/(len(data)*num_samples):.2f}秒")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存到: {output_file}")
    
    # 关闭Ray
    ray.shutdown()


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
    parser.add_argument("--num_gpus", type=int, default=1, 
                       help='使用的GPU数量进行数据并行（默认1）')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    num_samples = args.samples
    test_size = args.test_size
    num_gpus = args.num_gpus

    model_dir = model_to_dir[model_name]
    data_path = f"./data/{dataset}/test.json"
    
    if test_size == -1:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_ray_{num_gpus}gpus_2.json'
    else:
        output_file = f'./results/{model_name}/{dataset}/generated_answers_{num_samples}samples_{test_size}_ray_{num_gpus}gpus.json'

    with open(data_path, 'r') as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]

    run_vllm_generate_ray(
        model_dir, 
        dataset, 
        data, 
        num_samples, 
        output_file, 
        num_gpus=num_gpus
    )
