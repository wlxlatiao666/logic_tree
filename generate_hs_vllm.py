import os
import argparse
import logging
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_usr_prompt, parse_model_answer, get_gt_answer, match_answer
from importance import get_threshold
from vllm import LLM, SamplingParams
from vllm.sampling_params import TreeSearchParams

logger = logging.getLogger(__name__)
model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B",
    "gpt-oss-20b": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b",
    "Llama-3.1-8B-Instruct": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/Llama-3.1-8B-Instruct"
}

def run_hsampling(args):
    model_name = args.model
    model_dir = model_to_dir[model_name]
    dataset = args.dataset
    test_size = args.test_size
    tau = args.tau
    tau_importance = args.tau_importance
    num_branches = args.num_branches
    max_tree_depth = args.max_tree_depth
    device = torch.device(f"cuda:0")
    
    logger.info(f"模型: {model_name}, 数据集: {dataset}, 测试样本数: {test_size}, tau: {tau}, tau_importance: {tau_importance}, num_branches: {num_branches}, max_tree_depth: {max_tree_depth}")  
    
    # 加载数据
    data_path = f"./data/{dataset}/test.json"
    with open(data_path, 'r') as f:
        data = json.load(f)[:test_size] if test_size != -1 else json.load(f)

    with open('./sys_prompt.json', 'r') as f:
        sys_prompts = json.load(f)
    sys_prompt = sys_prompts[dataset]

    all_prompts = []
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, attn_implementation="eager").to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    thr, thr_importance = get_threshold(tokenizer, model, device, dataset, tau=tau, tau_importance=tau_importance)
    del model
    torch.cuda.empty_cache()

    for item in data:
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
        all_prompts.append(prompt)
    
    tree_config = TreeSearchParams(
        enable_tree_search=True,
        entropy_threshold=thr,
        branching_factor=num_branches,
        max_tree_depth=max_tree_depth,
        tau_importance=thr_importance
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        tree_search_params=tree_config
    )

    llm = LLM(
            model=model_dir,
            dtype="float16",
            tensor_parallel_size=1,  # 每个worker使用单GPU
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            trust_remote_code=True
        )
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)

    results = []
    for item, output in zip(data, outputs):
        seq_map = {out.seq_id: out for out in output.outputs}
        leaf_outputs = [out for out in output.outputs if out.is_leaf]
        leaf_texts = []
        for leaf_out in leaf_outputs:
            # Traverse up to collect texts and ids
            path_texts = []
            current = leaf_out
            while current is not None:
                path_texts.append(current.tree_text)
                if current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                    current = seq_map[current.parent_seq_id]
                else:
                    current = None

            # The path gives leaf to root, so we reverse it
            full_text = "".join(reversed(path_texts))
            leaf_texts.append(full_text)

        results.append({
            "original_data": item,
            "texts": leaf_texts
        })

    if test_size == -1:
        output_path = f"./results/{model_name}/{dataset}/hsampling_all_threshold{tau}_branches{num_branches}_depth{max_tree_depth}.json"
    else:
        output_path = f"./results/{model_name}/{dataset}/hsampling_size{test_size}_threshold{tau}_branches{num_branches}_depth{max_tree_depth}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--tau", type=int, default=80)
    parser.add_argument("--tau_importance", type=int, default=80)
    parser.add_argument("--num_branches", type=int, default=3)
    parser.add_argument("--max_tree_depth", type=int, default=3)
    args = parser.parse_args()
    
    run_hsampling(args)
    
