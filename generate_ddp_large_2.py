import torch
import datetime
import os
import json
import argparse
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode
from utils import generate_usr_prompt
from importance import get_threshold
from multiprocessing import Process, set_start_method

logger = logging.getLogger(__name__)
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B",
    "gpt-oss-20b": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b"
}

def process_single_item(item_data, tokenizer, model, dataset, sys_prompt, num_leaves, thr, thr_importance):
    """处理单个数据项，不再加载模型，直接用预加载的model/tokenizer"""
    item, global_idx = item_data

    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    usr_prompt = generate_usr_prompt(dataset, item)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    prompt = tokenizer.decode(inputs["input_ids"][0])

    all_leaves = []
    all_tokens = 0

    while len(all_leaves) < num_leaves:
        leaves, new_tokens_cnt = logic_branch_decode(
            tokenizer, model, device, prompt=prompt,
            sample=True, tau=thr, tau_importance=thr_importance,
            branches_m=3, max_leaves=num_leaves - len(all_leaves)
        )
        for leaf in leaves:
            leaf.prob = leaf.prob * len(leaves) / num_leaves
        all_leaves.extend(leaves)
        all_tokens += new_tokens_cnt

    total_prob = sum(leaf.prob for leaf in all_leaves)
    for leaf in all_leaves:
        leaf.prob /= total_prob

    complexity = sum(leaf.prob * leaf.depth for leaf in all_leaves)
    probs = [leaf.prob for leaf in all_leaves]
    texts = [leaf.text for leaf in all_leaves]
    entropies = [-leaf.cum_logprob / leaf.length if leaf.length > 0 else 0.0 for leaf in all_leaves]
    split_positions = [leaf.split_positions for leaf in all_leaves]
    lengths = [leaf.length for leaf in all_leaves]

    result = {
        "original_data": item,
        "num_new_tokens": all_tokens,
        "num_leaves": len(all_leaves),
        "complexity": complexity,
        "probs": probs,
        "entropies": entropies,
        "split_positions": split_positions,
        "lengths": lengths,
        "texts": texts,
        "global_index": global_idx
    }

    return result

def worker_proc(proc_idx, gpu_ids, data_chunk, args, thr, thr_importance, sys_prompt):
    # 关键： 设置本进程可见GPU环境变量，确保启动模型加载时只看到分配给它的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    # 错开加载时间，减少竞争
    time.sleep(proc_idx * 20)
    
    model_name = args.model
    model_dir = model_to_dir[model_name]
    num_leaves = args.num_leaves
    dataset = args.dataset

    print(f"[Proc {proc_idx}] Loading tokenizer and model on GPUs {gpu_ids}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.float16
        # 可以调 max_memory 控制显存分配，按需添加
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for local_idx, (item, global_idx) in enumerate(data_chunk):
        print(f"processing item {local_idx}")
        try:
            res = process_single_item((item, global_idx), tokenizer, model, dataset, sys_prompt, num_leaves, thr, thr_importance)
            results.append(res)
        except Exception as e:
            results.append({"original_data": item, "global_index": global_idx, "status": "failed", "error": str(e)})

    # 释放显存
    del model
    torch.cuda.empty_cache()

    temp_dir = './temp_results'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f'temp_{model_name}_{dataset}_proc{proc_idx}.json')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Proc {proc_idx}] Done, results saved to {temp_path}")

def run_inference_tpdp(args):
    model_name = args.model
    model_dir = model_to_dir[model_name]
    dataset = args.dataset
    test_size = args.test_size
    num_leaves = args.num_leaves
    tau = args.tau
    num_proc = args.num_proc
    gpus_per_proc = 2
    total_gpus = num_proc * gpus_per_proc

    # 加载数据
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]

    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    print("Preloading model weights to cache...")
    # _ = AutoModelForCausalLM.from_pretrained(
    #     model_dir, attn_implementation="eager", device_map="cpu", torch_dtype=torch.float16
    # )
    # del _
    torch.cuda.empty_cache()
    print("Model weights cached. Computing threshold...")

    device = torch.device("cuda:0")
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_dir,
    #     attn_implementation="eager",
    #     device_map="auto",
    #     torch_dtype=torch.float16
    # )
    # thr, thr_importance = get_threshold(tokenizer, model, device, dataset, tau=tau)
    # print(f"Threshold for {dataset}: {thr}, {thr_importance}")
    # del model
    torch.cuda.empty_cache()
    # del tokenizer

    # 数据分片
    items_with_idx = [(item, idx) for idx, item in enumerate(data)]
    chunk_size = (len(items_with_idx) + num_proc - 1) // num_proc
    data_chunks = [items_with_idx[i*chunk_size:(i+1)*chunk_size] for i in range(num_proc)]

    # 分配GPU: 每个进程2张GPU
    gpu_lists = [[i*gpus_per_proc + j for j in range(gpus_per_proc)] for i in range(num_proc)]

    # 启动多进程
    # procs = []
    # for proc_idx in range(num_proc):
    #     p = Process(target=worker_proc, args=(proc_idx, gpu_lists[proc_idx], data_chunks[proc_idx], args, thr, thr_importance, sys_prompt))
    #     p.start()
    #     procs.append(p)
    # for p in procs:
    #     p.join()

    # 合并结果
    results = []
    for proc_idx in range(num_proc):
        temp_path = f'./temp_results/temp_{model_name}_{dataset}_proc{proc_idx}.json'
        with open(temp_path, 'r', encoding='utf-8') as f:
            results.extend(json.load(f))
    results.sort(key=lambda x: x.get("global_index", -1))
    for result in results:
        if "global_index" in result:
            del result["global_index"]

    # 保存最终结果路径
    if test_size == -1:
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_all_leaves{num_leaves}_threshold{tau}_importance80_branches3_ddp_waad.json"
    else:
        output_path = f"./results/{model_name}/{dataset}/logic_tree_results_{test_size}_leaves{num_leaves}_threshold{tau}_importance80_branches3_ddp_waad.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # 使用spawn启动，避免fork造成的锁死
    set_start_method('spawn', force=True)

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--num_leaves", type=int, default=20)
    parser.add_argument("--tau", type=int, default=80)
    parser.add_argument("--num_proc", type=int, default=2, help="进程数，每进程2卡")
    args = parser.parse_args()

    os.makedirs('./logs', exist_ok=True)

    output_path = run_inference_tpdp(args)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds. Results saved to {output_path}")