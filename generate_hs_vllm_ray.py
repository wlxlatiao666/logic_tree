import os
import json
import time
import logging
import argparse
import numpy as np
from transformers import AutoTokenizer
from utils import generate_usr_prompt, parse_model_answer, get_gt_answer, match_answer

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

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import TreeSearchParams
    import ray
    _HAS_VLLM_RAY = True
except Exception as e:
    _HAS_VLLM_RAY = False
    print(f"Warning: vllm or ray not available: {e}")


# Number of calibration samples used to estimate thresholds.
_CALIB_SIZE = 50


@ray.remote(num_gpus=1)
class HSWorker:
    """Ray worker，每个worker在一个GPU上运行独立的vLLM实例，执行hierarchical sampling。
    threshold由worker自身用collect_threshold_stats模式校准，无需外部LLM实例。
    """

    def __init__(self, model_dir: str, gpu_id: int, tau_pct: int, tau_importance_pct: int,
                 num_branches: int, max_tree_depth: int):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.tau_pct = tau_pct
        self.tau_importance_pct = tau_importance_pct
        self.num_branches = num_branches
        self.max_tree_depth = max_tree_depth

        # thr/thr_importance will be set by calibrate()
        self.thr = None
        self.thr_importance = None

        self.llm = LLM(
            model=model_dir,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            # trust_remote_code=True
        )
        print(f"HSWorker initialized on GPU {gpu_id}")

    def calibrate(self, calib_prompts: list) -> tuple:
        """用collect_threshold_stats模式跑校准样本，计算并存储thr和thr_importance。
        返回(thr, thr_importance)供主进程记录日志。
        """
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            collect_threshold_stats=True,
        )
        outputs = self.llm.generate(calib_prompts, sampling_params=sampling_params)

        all_entropy, all_importance = [], []
        for req_out in outputs:
            for comp_out in req_out.outputs:
                all_entropy.extend(comp_out.entropy_list)
                imp = [v for v in comp_out.importance_list if v is not None]
                all_importance.extend(imp)

        self.thr = float(np.percentile(all_entropy, self.tau_pct)) if all_entropy else 1.0
        self.thr_importance = float(np.percentile(all_importance, self.tau_importance_pct)) if all_importance else None
        return self.thr, self.thr_importance

    def generate(self, prompts):
        print(f"GPU {self.gpu_id}: thr={self.thr}, thr_importance={self.thr_importance}")
        tree_config = TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=self.thr,
            branching_factor=self.num_branches,
            max_tree_depth=self.max_tree_depth,
            tau_importance=self.thr_importance
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=32768,
            tree_search_params=tree_config
        )
        results = []
        for prompt in prompts:
            leaf_texts = []
            while len(leaf_texts) < 20:
                outputs = self.llm.generate(prompt, sampling_params=sampling_params)
                seq_map = {output.seq_id: output for output in outputs[0].outputs}
                leaf_outputs = [output for output in outputs[0].outputs if output.is_leaf]
                
                for leaf_out in leaf_outputs:
                    # Traverse up to collect texts and ids
                    path_texts = []
                    path_ids = []
                    current = leaf_out
                    while current is not None:
                        path_texts.append(current.tree_text)
                        path_ids.append(list(current.tree_ids))
                        if current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                            current = seq_map[current.parent_seq_id]
                        else:
                            current = None

                    # The path gives leaf to root, so we reverse it
                    full_text = "".join(reversed(path_texts))
                    leaf_texts.append(full_text)
            results.append({"texts": leaf_texts})

        # outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        # for output in outputs:
        #     seq_map = {out.seq_id: out for out in output.outputs}
        #     leaf_outputs = [out for out in output.outputs if out.is_leaf]
        #     leaf_texts = []
        #     for leaf_out in leaf_outputs:
        #         path_texts = []
        #         current = leaf_out
        #         while current is not None:
        #             path_texts.append(current.tree_text)
        #             if current.parent_seq_id is not None and current.parent_seq_id in seq_map:
        #                 current = seq_map[current.parent_seq_id]
        #             else:
        #                 current = None
        #         full_text = "".join(reversed(path_texts))
        #         leaf_texts.append(full_text)
        #     results.append({"texts": leaf_texts})

        return results


def run_hsampling_ray(args):
    if not _HAS_VLLM_RAY:
        raise RuntimeError("vllm or ray is not installed.")

    model_name = args.model
    model_dir = model_to_dir[model_name]
    dataset = args.dataset
    test_size = args.test_size
    tau = args.tau
    tau_importance = args.tau_importance
    num_branches = args.num_branches
    max_tree_depth = args.max_tree_depth
    num_gpus = args.num_gpus

    logger.info(f"模型: {model_name}, 数据集: {dataset}, 测试样本数: {test_size}, "
                f"tau: {tau}, tau_importance: {tau_importance}, "
                f"num_branches: {num_branches}, max_tree_depth: {max_tree_depth}, num_gpus: {num_gpus}")

    # 加载数据
    data_path = f"./data/{dataset}/test.json"
    with open(data_path, 'r') as f:
        data = json.load(f)[:test_size] if test_size != -1 else json.load(f)

    with open('./sys_prompt.json', 'r') as f:
        sys_prompts = json.load(f)
    sys_prompt = sys_prompts[dataset]

    # 构建所有prompts（tokenizer仅用于chat template，不加载模型权重）
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    all_prompts = []
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
        all_prompts.append(tokenizer.decode(inputs["input_ids"][0]))

    # 数据分片
    chunk_size = len(all_prompts) // num_gpus
    remainder = len(all_prompts) % num_gpus
    prompt_chunks, data_chunks = [], []
    start_idx = 0
    for i in range(num_gpus):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        prompt_chunks.append(all_prompts[start_idx:end_idx])
        data_chunks.append(data[start_idx:end_idx])
        start_idx = end_idx

    logger.info(f"数据分片完成: {[len(c) for c in prompt_chunks]}")

    # 初始化Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus)

    # 创建workers
    logger.info("初始化workers...")
    workers = [
        HSWorker.remote(model_dir, gpu_id=i, tau_pct=tau, tau_importance_pct=tau_importance,
                        num_branches=num_branches, max_tree_depth=max_tree_depth)
        for i in range(num_gpus)
    ]

    # 每个worker用自己的校准样本计算threshold（复用同一LLM实例，无需额外显存）
    calib_size = min(_CALIB_SIZE, len(all_prompts))
    logger.info(f"各worker并行校准threshold（校准样本数: {calib_size}）...")
    calib_futures = [workers[i].calibrate.remote(all_prompts[:calib_size]) for i in range(num_gpus)]
    calib_results = ray.get(calib_futures)
    for i, (thr_val, thr_imp_val) in enumerate(calib_results):
        logger.info(f"Worker {i}: thr={thr_val}, thr_importance={thr_imp_val}")

    start_time = time.time()
    logger.info("开始并行生成...")

    futures = [workers[i].generate.remote(prompt_chunks[i]) for i in range(num_gpus)]
    all_results = ray.get(futures)

    generation_time = time.time() - start_time
    logger.info(f"并行生成完成，耗时: {generation_time:.2f}秒 ({generation_time/60:.2f}分钟)")

    # 合并结果
    results = []
    for worker_idx, worker_results in enumerate(all_results):
        for item_idx, item_result in enumerate(worker_results):
            item = data_chunks[worker_idx][item_idx]
            results.append({
                "original_data": item,
                "texts": item_result["texts"]
            })

    # 保存结果
    if test_size == -1:
        output_path = (f"./results/{model_name}/{dataset}/"
                       f"hsampling_all_threshold{tau}_importance{tau_importance}_branches{num_branches}_depth{max_tree_depth}_ray_{num_gpus}gpus.json")
    else:
        output_path = (f"./results/{model_name}/{dataset}/"
                       f"hsampling_size{test_size}_threshold{tau}_importance{tau_importance}_branches{num_branches}_depth{max_tree_depth}_ray_{num_gpus}gpus.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存到: {output_path}")
    ray.shutdown()


if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(model_to_dir.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--tau", type=int, default=80)
    parser.add_argument("--tau_importance", type=int, default=80)
    parser.add_argument("--num_branches", type=int, default=3)
    parser.add_argument("--max_tree_depth", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的GPU数量进行数据并行（默认1）")
    args = parser.parse_args()

    run_hsampling_ray(args)
