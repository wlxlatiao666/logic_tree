import torch
import datetime
import os
import json
import argparse
import time
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode_3 import logic_branch_decode
from utils import generate_usr_prompt
from threshold import get_threshold

logger = logging.getLogger(__name__)
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'
model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B"
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
    
    # 配置日志（仅在主进程记录详细日志）
    if rank == 0:
        fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)  # 其他进程只记录错误
    
    # 加载模型
    model_name = args.model
    model_dir = model_to_dir[model_name]
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        logger.info(f"使用GPU设备: {rank}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to(device)
    
    # 包装为DDP模型
    ddp_model = DDP(model, device_ids=[rank])
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    dataset = args.dataset
    test_size = args.test_size
    num_leaves = args.num_leaves
    tau = args.tau
    dataset_path = f"./data/{dataset}/test.json"
    
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]
    
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]
    
    # 仅在主进程计算阈值，然后广播给其他进程
    if rank == 0:
        thr = get_threshold(tokenizer, model, device, dataset, tau=tau)
        logger.info(f"Threshold for {dataset}: {thr}")
    else:
        thr = torch.tensor(0.0, device=device)
    
    # 广播阈值
    if rank == 0:
        dist.broadcast(torch.tensor(thr, device=device), src=0)
    else:
        dist.broadcast(thr, src=0)
        thr = thr.item()
    
    # 数据分片
    chunk_size = len(data) // world_size
    remainder = len(data) % world_size
    
    if rank < remainder:
        # 前remainder个进程每个多处理一个样本
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size
    
    local_data = data[start_idx:end_idx]
    local_results = []
    
    # 本地推理
    for i, item in enumerate(local_data):
        global_idx = start_idx + i
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用原始模型进行推理（因为logic_branch_decode不支持DDP直接调用）
        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, device, prompt=prompt, 
                                                          sample=True, tau=thr, branches_m=3, max_leaves=num_leaves)
        
        complexity = sum(leaf.prob * leaf.depth for leaf in leaves)
        probs = [leaf.prob for leaf in leaves]
        texts = [leaf.text for leaf in leaves]
        entropies = [-leaf.cum_logprob / leaf.length if leaf.length > 0 else 0.0 for leaf in leaves]
        split_positions = [leaf.split_positions for leaf in leaves]
        lengths = [leaf.length for leaf in leaves]
        
        local_results.append({
            "original_data": item,
            "num_new_tokens": new_tokens_cnt,
            "num_leaves": len(leaves),
            "complexity": complexity,
            "probs": probs,
            "entropies": entropies,
            "split_positions": split_positions,
            "lengths": lengths,
            "texts": texts,
            "global_index": global_idx  # 保存全局索引以便合并时排序
        })
        
        if rank == 0:
            logger.info(f"Processed {global_idx+1} items")
    
    # Save local results to a temporary file
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
    num_leaves = args.num_leaves
    tau = args.tau
    world_size = args.world_size
    
    temp_dir = f"./results/temp/{model_name}/{dataset}"
    final_results = []
    
    for rank in range(world_size):
        temp_file = os.path.join(temp_dir, f"rank_{rank}.json")
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding="utf8") as f:
                final_results.extend(json.load(f))
            # os.remove(temp_file)  # Optional: keep for debugging
        else:
            print(f"Warning: Missing result file for rank {rank}")
            
    final_results.sort(key=lambda x: x["global_index"])
    # Remove temporary index field
    for result in final_results:
        if "global_index" in result:
            del result["global_index"]
    
    # Save final results
    if test_size == -1:
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_all_leaves{num_leaves}_threshold{tau}_ddp_random.json"
    else:
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_{test_size}_leaves{num_leaves}_threshold{tau}_ddp_random.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"Merged results saved to {output_path}")
    # shutil.rmtree(temp_dir) # Optional: clean up temp dir

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--num_leaves", type=int, default=20)
    parser.add_argument("--tau", type=int, default=80)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), 
                       help="使用的GPU数量")
    args = parser.parse_args()
    
    mp.spawn(run_inference, args=(args.world_size, args), nprocs=args.world_size, join=True)
    
    # Merge results from all ranks
    merge_results(args)
    
    logger.info(f"generate results in {time.time() - start_time:.2f} seconds")
    
