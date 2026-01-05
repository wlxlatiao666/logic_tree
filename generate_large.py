import torch
import datetime
import os
import json
import argparse
import time
import logging
import traceback
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode
from utils import generate_usr_prompt
from threshold_relevance import get_threshold
from tqdm import tqdm

logger = logging.getLogger(__name__)
# os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'
model_to_dir = {
    "Qwen2.5-7B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-8B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/inspire/hdd/global_public/public_models/Qwen/Qwen3-14B",
    "gpt-oss-20b": "/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b"
}


def find_free_port():
    """动态查找一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup(rank, world_size, master_port):
    """初始化全局进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=10800))
    torch.cuda.set_device(rank)


def cleanup():
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def run_inference(rank, world_size, args, log_filename, master_port):
    """
    混合并行推理函数
    
    参数说明:
    - model_parallel_size: 每个模型实例使用的GPU数量
    - data_parallel_size: 数据并行的组数 (world_size / model_parallel_size)
    """
    
    model_parallel_size = args.model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    
    # 计算当前进程的组信息
    data_parallel_rank = rank // model_parallel_size  # 数据并行组的编号
    model_parallel_rank = rank % model_parallel_size  # 模型并行组内的编号
    is_group_leader = (model_parallel_rank == 0)      # 是否是组内的leader
    
    # 计算当前组使用的GPU列表
    group_gpu_list = list(range(
        data_parallel_rank * model_parallel_size,
        (data_parallel_rank + 1) * model_parallel_size
    ))
    
    try:
        setup(rank, world_size, master_port)
    except Exception as e:
        print(f"\n✗ GPU {rank}: Failed to initialize process group: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return
    
    # ============ 配置日志 ============
    try:
        if rank == 0:
            fh = logging.FileHandler(log_filename, encoding='utf-8')
        else:
            rank_log_filename = log_filename.replace('.log', f'_rank{rank}.log')
            fh = logging.FileHandler(rank_log_filename, encoding='utf-8')
        
        formatter = logging.Formatter('%(asctime)s - [GPU %(process)d] - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        
        logger.info(f"="*80)
        logger.info(f"GPU {rank}: Starting inference process")
        logger.info(f"GPU {rank}: Data Parallel Rank: {data_parallel_rank}/{data_parallel_size}")
        logger.info(f"GPU {rank}: Model Parallel Rank: {model_parallel_rank}/{model_parallel_size}")
        logger.info(f"GPU {rank}: Group GPUs: {group_gpu_list}")
        logger.info(f"GPU {rank}: Is Group Leader: {is_group_leader}")
        logger.info(f"GPU {rank}: Master port: {master_port}")
        logger.info(f"="*80)
    except Exception as e:
        print(f"\n✗ GPU {rank}: Failed to setup logging: {str(e)}")
        cleanup()
        return
    
    # ============ 创建数据并行进程组 ============
    # 为数据并行组的leader创建一个独立的进程组，用于后续的gather操作
    dp_group = None
    if is_group_leader:
        dp_ranks = [i * model_parallel_size for i in range(data_parallel_size)]
        dp_group = dist.new_group(ranks=dp_ranks)
        logger.info(f"GPU {rank}: Created data parallel group with ranks: {dp_ranks}")
    
    # ============ 加载模型 ============
    model_name = args.model
    model_dir = model_to_dir[model_name]
    
    if rank == 0:
        print(f"\n正在加载模型... / Loading model...")
        print(f"  模型并行: {model_parallel_size} GPUs per model")
        print(f"  数据并行: {data_parallel_size} groups")
        print(f"  GPU分组:")
        for dp_rank in range(data_parallel_size):
            gpu_list = list(range(dp_rank * model_parallel_size, (dp_rank + 1) * model_parallel_size))
            print(f"    组 {dp_rank}: GPUs {gpu_list}")
        print()
    
    logger.info(f"GPU {rank}: Loading model from {model_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        logger.info(f"GPU {rank}: Tokenizer loaded successfully")
        
        # ============ 关键: 使用device_map实现模型分片 ============
        # 方案1: 自动分片到当前组的所有GPU
        device_map_config = "auto" if model_parallel_size > 1 else f"cuda:{rank}"
        
        # 方案2: 手动指定device_map (更精确控制)
        if model_parallel_size > 1:
            # 设置环境变量，限制模型只能看到当前组的GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, group_gpu_list))
            device_map_config = "auto"
            logger.info(f"GPU {rank}: Set CUDA_VISIBLE_DEVICES to {group_gpu_list}")
        else:
            device_map_config = {"": torch.device(f"cuda:{rank}")}
        
        logger.info(f"GPU {rank}: Loading model with device_map: {device_map_config}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device_map_config,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        model.eval()
        logger.info(f"GPU {rank}: Model loaded successfully and set to eval mode")
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"GPU {rank}: Set pad_token to eos_token")
        
        # 打印模型分布情况 (仅group leader)
        if is_group_leader:
            if hasattr(model, 'hf_device_map'):
                logger.info(f"GPU {rank}: Model device map: {model.hf_device_map}")
        
    except Exception as e:
        logger.error(f"GPU {rank}: CRITICAL ERROR - Failed to load model: {str(e)}")
        logger.error(f"GPU {rank}: Traceback:\n{traceback.format_exc()}")
        cleanup()
        return
    
    # ...existing code...
    # ============ 加载数据 ============
    dataset = args.dataset
    test_size = args.test_size
    start_index = args.start_index
    num_leaves = args.num_leaves
    tau = args.tau
    dataset_path = f"./data/{dataset}/test.json"
    
    logger.info(f"GPU {rank}: Loading dataset from {dataset_path}")
    logger.info(f"GPU {rank}: Parameters - start_index={start_index}, test_size={test_size}")
    
    try:
        with open(dataset_path, "r") as f:
            full_data = json.load(f)
        
        logger.info(f"GPU {rank}: Total data size: {len(full_data)}")
        
        if start_index >= len(full_data):
            logger.error(f"GPU {rank}: CRITICAL ERROR - start_index ({start_index}) >= total data size ({len(full_data)})")
            if rank == 0:
                print(f"\n错误: start_index ({start_index}) 超出数据范围 (总数据量: {len(full_data)})\n")
            cleanup()
            return
        
        # 从start_index开始切片
        if test_size == -1:
            data = full_data[start_index:]
            actual_end_index = len(full_data) - 1
        else:
            end_index = min(start_index + test_size, len(full_data))
            data = full_data[start_index:end_index]
            actual_end_index = end_index - 1
        
        logger.info(f"GPU {rank}: Loaded {len(data)} samples from index {start_index} to {actual_end_index}")
        
    except FileNotFoundError as e:
        logger.error(f"GPU {rank}: CRITICAL ERROR - Dataset file not found: {dataset_path}")
        logger.error(f"GPU {rank}: {str(e)}")
        cleanup()
        return
    except json.JSONDecodeError as e:
        logger.error(f"GPU {rank}: CRITICAL ERROR - Failed to parse JSON file: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup()
        return
    except Exception as e:
        logger.error(f"GPU {rank}: CRITICAL ERROR - Failed to load data: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup()
        return
    
    try:
        with open(f"./sys_prompt.json", "r") as f:
            sys_prompt = json.load(f)[dataset]
        logger.info(f"GPU {rank}: System prompt loaded successfully")
    except Exception as e:
        logger.error(f"GPU {rank}: CRITICAL ERROR - Failed to load system prompt: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup()
        return
    
    # ============ 计算阈值 (只在rank 0计算) ============
    # 关键修正：device 应该指向当前组的第一个可见GPU（group_gpu_list[0]），而不是全局rank
    device = torch.device(f"cuda:{group_gpu_list[0]}")
    if rank == 0:
        # device 已经设置为 group_gpu_list[0]
        print("\n" + "="*70)
        print("  计算阈值中... / Calculating threshold...")
        print("="*70)
        try:
            thr, thr_importance = get_threshold(tokenizer, model, device, dataset, tau=tau)
            # thr = 0.627051
            logger.info(f"Threshold for {dataset}: {thr}, {thr_importance}")
            thr_tensor = torch.tensor(thr, dtype=torch.float32).cuda()
            thr_importance_tensor = torch.tensor(thr_importance, dtype=torch.float32).cuda()
            print(f"\n开始推理... / Starting inference...\n")
        except Exception as e:
            logger.error(f"Failed to calculate threshold: {str(e)}")
            logger.error(traceback.format_exc())
            thr_tensor = torch.tensor(0.627051, dtype=torch.float32).cuda()
            thr_importance_tensor = torch.tensor(0.0, dtype=torch.float32).cuda()
            logger.warning(f"Using default threshold: 0.627051")
    else:
        thr_tensor = torch.tensor(0.0, dtype=torch.float32).cuda()
        thr_importance_tensor = torch.tensor(0.0, dtype=torch.float32).cuda()
    
    # 广播阈值到所有进程
    dist.broadcast(thr_tensor, src=0)
    dist.broadcast(thr_importance_tensor, src=0)
    thr = thr_tensor.item()
    thr_importance = thr_importance_tensor.item()
    logger.info(f"GPU {rank}: Threshold synchronized: {thr}, {thr_importance}")
    
    # ============ 数据分片 (只在数据并行组之间分片) ============
    # 关键: 只有组leader参与数据分片，组内其他成员不处理数据
    if is_group_leader:
        chunk_size = len(data) // data_parallel_size
        remainder = len(data) % data_parallel_size
        
        if data_parallel_rank < remainder:
            local_start_idx = data_parallel_rank * (chunk_size + 1)
            local_end_idx = local_start_idx + chunk_size + 1
        else:
            local_start_idx = remainder * (chunk_size + 1) + (data_parallel_rank - remainder) * chunk_size
            local_end_idx = local_start_idx + chunk_size
        
        local_data = data[local_start_idx:local_end_idx]
        
        logger.info(f"GPU {rank}: [Group Leader] Assigned {len(local_data)} samples")
        logger.info(f"GPU {rank}: Local index range: {local_start_idx} to {local_end_idx-1}")
        logger.info(f"GPU {rank}: Global index range: {start_index + local_start_idx} to {start_index + local_end_idx - 1}")
        
        if rank == 0:
            logger.info(f"Total samples to process: {len(data)}, samples per group: ~{chunk_size}")
    else:
        # 非leader进程不处理数据，只是等待
        local_data = []
        local_start_idx = 0
        logger.info(f"GPU {rank}: [Group Member] No data assigned, will assist with model computation")
    
    # ============ 创建临时结果文件 (只有group leader创建) ============
    if is_group_leader:
        temp_dir = "./temp_results"
        os.makedirs(temp_dir, exist_ok=True)
        
        if test_size == -1:
            temp_result_file = os.path.join(temp_dir, 
                f"temp_{model_name}_{dataset}_group{data_parallel_rank}_start{start_index}_leaves{num_leaves}_tau{tau}.jsonl")
        else:
            temp_result_file = os.path.join(temp_dir, 
                f"temp_{model_name}_{dataset}_group{data_parallel_rank}_size{test_size}_start{start_index}_leaves{num_leaves}_tau{tau}.jsonl")
        
        if os.path.exists(temp_result_file):
            os.remove(temp_result_file)
            logger.info(f"GPU {rank}: Removed existing temp file: {temp_result_file}")
        
        logger.info(f"GPU {rank}: Will save results incrementally to: {temp_result_file}")
        
        local_results = []
        success_count = 0
        error_count = 0
    
    # ============ 进度条 (只在rank 0显示) ============
    if rank == 0:
        pbar = tqdm(
            total=len(local_data),
            desc=f"组 0/{data_parallel_size}",
            leave=True,
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    # ============ 本地推理 (只有group leader执行) ============
    if is_group_leader:
        # 注意: 虽然只有leader调用推理函数，但模型计算会自动使用组内所有GPU
        device = torch.device(f"cuda:{rank}")  # leader的设备
        
        for i, item in enumerate(local_data):
            local_idx = local_start_idx + i
            global_idx = start_index + local_idx
            try:
                logger.info(f"GPU {rank}: [Sample {i+1}/{len(local_data)}] Starting to process global index {global_idx}")
                usr_prompt = generate_usr_prompt(dataset, item)
                prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
                logger.debug(f"GPU {rank}: Generated prompt for sample {global_idx}")
                all_leaves = []
                all_tokens = 0
                # 推理 - 模型会自动使用组内所有GPU
                while len(all_leaves) < num_leaves:
                    leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, device, prompt=prompt, 
                                                                sample=True, tau=thr, tau_importance=thr_importance, branches_m=3, max_leaves=num_leaves-len(all_leaves))
                    for leaf in leaves:
                        leaf.prob = leaf.prob * len(leaves) / num_leaves
                    all_leaves.extend(leaves)
                    all_tokens += new_tokens_cnt
                logger.info(f"GPU {rank}: Logic tree decode completed for sample {global_idx} - "
                           f"{len(all_leaves)} leaves, {all_tokens} new tokens")
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
                    "global_index": global_idx,
                    "status": "success",
                    "processed_by_group": data_parallel_rank,
                    "group_gpus": group_gpu_list
                }
                try:
                    with open(temp_result_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    logger.debug(f"GPU {rank}: Saved result for sample {global_idx} to temp file")
                except Exception as save_error:
                    logger.error(f"GPU {rank}: Failed to save result to temp file: {str(save_error)}")
                local_results.append(result)
                success_count += 1
                logger.info(f"GPU {rank}: ✓ Successfully processed sample {global_idx} "
                           f"(complexity: {complexity:.4f})")
            except Exception as e:
                error_count += 1
                error_msg = str(e)
                error_trace = traceback.format_exc()
                logger.error(f"GPU {rank}: ✗ ERROR processing sample {global_idx}")
                logger.error(f"GPU {rank}: Error message: {error_msg}")
                logger.error(f"GPU {rank}: Full traceback:\n{error_trace}")
                result = {
                    "original_data": item,
                    "global_index": global_idx,
                    "status": "failed",
                    "error_message": error_msg,
                    "error_traceback": error_trace,
                    "processed_by_group": data_parallel_rank,
                    "group_gpus": group_gpu_list
                }
                try:
                    with open(temp_result_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    logger.debug(f"GPU {rank}: Saved failed result for sample {global_idx} to temp file")
                except Exception as save_error:
                    logger.error(f"GPU {rank}: Failed to save failed result to temp file: {str(save_error)}")
                local_results.append(result)
                if rank == 0:
                    print(f"\n⚠ 警告: 样本 {global_idx} 处理失败，已跳过并记录错误\n")
            if rank == 0:
                pbar.update(1)
        if rank == 0:
            pbar.close()
            print(f"\n{'组 0 完成!':<20} 其他组也在后台完成，等待同步...\n")
        logger.info(f"GPU {rank}: Local processing completed - Success: {success_count}, Errors: {error_count}")
        logger.info(f"GPU {rank}: Success rate: {success_count}/{len(local_data)} ({100*success_count/len(local_data):.2f}%)")
        logger.info(f"GPU {rank}: All results have been saved to: {temp_result_file}")
    dist.barrier()
    
    # ============ 收集结果 (只在group leaders之间收集) ============
    if is_group_leader:
        logger.info(f"GPU {rank}: Gathering results from all groups...")
        
        # 只有group leaders参与gather
        all_results = [None] * data_parallel_size
        dist.gather_object(
            local_results, 
            all_results if rank == 0 else None, 
            dst=0,
            group=dp_group
        )
        
        # ============ 主进程合并结果 ============
        if rank == 0:
            print("="*70)
            print("  合并结果中... / Merging results...")
            print("="*70)
            
            logger.info("Merging results from all groups...")
            
            final_results = []
            total_success = 0
            total_errors = 0
            
            for group_idx, group_results in enumerate(all_results):
                logger.info(f"Received {len(group_results)} results from Group {group_idx}")
                final_results.extend(group_results)
            
            # 按global_index排序
            final_results.sort(key=lambda x: x["global_index"])
            logger.info(f"Sorted {len(final_results)} results by global_index")
            
            # 统计
            for result in final_results:
                if result["status"] == "success":
                    total_success += 1
                else:
                    total_errors += 1
            
            logger.info(f"Final statistics - Total: {len(final_results)}, "
                       f"Success: {total_success}, Errors: {total_errors}")
            
            # 保存最终结果
            if test_size == -1:
                output_path = f"./results/{model_name}/{dataset}/logic_tree_results_all_leaves{num_leaves}_threshold{tau}_start{start_index}_end{actual_end_index}_hybrid_parallel.json"
            else:
                output_path = f"./results/{model_name}/{dataset}/logic_tree_results_{test_size}_leaves{num_leaves}_threshold{tau}_start{start_index}_end{actual_end_index}_hybrid_parallel.json"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                error_samples = [r for r in final_results if r["status"] == "failed"]
                
                with open(output_path, 'w', encoding="utf8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Results saved to {output_path}")
                
                if error_samples:
                    error_path = output_path.replace('.json', '_ERRORS.json')
                    with open(error_path, 'w', encoding="utf8") as f:
                        json.dump(error_samples, f, indent=2, ensure_ascii=False)
                    logger.warning(f"⚠ Error samples saved separately to {error_path}")
                
                print(f"\n✓ 结果已保存 / Results saved to:\n  {output_path}")
                print(f"\n统计信息 / Statistics:")
                print(f"  总样本数 (Total)    : {len(final_results)}")
                print(f"  成功 (Success)      : {total_success} ({100*total_success/len(final_results):.2f}%)")
                print(f"  失败 (Failed)       : {total_errors} ({100*total_errors/len(final_results):.2f}%)")
                
                if total_errors > 0:
                    print(f"\n⚠ 注意: 有 {total_errors} 个样本处理失败")
                    print(f"  详细错误信息已保存到: {error_path}")
                    print(f"  请查看日志文件了解完整错误堆栈")
                
                print(f"\n临时文件保存在: {temp_dir}")
                print(f"  如果需要恢复数据，请保留临时文件")
                print(f"  如果确认无需恢复，可手动删除临时文件以节省空间")
                print()
                
            except Exception as e:
                logger.error(f"CRITICAL ERROR - Failed to save results: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"\n✗ 保存结果失败 / Failed to save results: {str(e)}")
                print(f"  但临时文件已保存在: {temp_dir}")
                print(f"  可以从临时文件手动恢复数据\n")
    
    logger.info(f"GPU {rank}: Cleaning up and exiting...")
    cleanup()


if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Logic Tree Inference with Hybrid Parallelism (Model + Data Parallel)")
    parser.add_argument("--model", type=str, required=True, choices=model_to_dir.keys(),
                       help="Model name to use for inference")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--test_size", type=int, default=-1, 
                       help="Number of samples to process (-1 for all samples from start_index)")
    parser.add_argument("--start_index", type=int, default=0,
                       help="Starting index in the dataset (default: 0)")
    parser.add_argument("--num_leaves", type=int, default=20,
                       help="Maximum number of leaves in logic tree")
    parser.add_argument("--tau", type=int, default=80,
                       help="Threshold parameter (percentage)")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), 
                       help="Total number of GPUs to use")
    parser.add_argument("--model_parallel_size", type=int, default=1,
                       help="Number of GPUs per model instance (model parallelism)")
    args = parser.parse_args()
    
    # 验证参数
    if args.world_size % args.model_parallel_size != 0:
        print(f"\n✗ 错误: world_size ({args.world_size}) 必须是 model_parallel_size ({args.model_parallel_size}) 的整数倍")
        print(f"  例如: 4个GPU, model_parallel_size可以是 1, 2, 或 4")
        exit(1)
    
    data_parallel_size = args.world_size // args.model_parallel_size
    
    # 生成日志文件名
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logs_dir = './logs'
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")
    
    # 创建主日志文件
    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"逻辑树推理程序 (混合并行) / Logic Tree Inference Program (Hybrid Parallel)\n")
            f.write(f"{'='*80}\n")
            f.write(f"启动时间: {datetime.datetime.now()}\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"数据集: {args.dataset}\n")
            f.write(f"起始索引: {args.start_index}\n")
            f.write(f"样本数: {args.test_size if args.test_size != -1 else 'All from start_index'}\n")
            f.write(f"总GPU数: {args.world_size}\n")
            f.write(f"模型并行大小: {args.model_parallel_size} GPUs/model\n")
            f.write(f"数据并行大小: {data_parallel_size} groups\n")
            f.write(f"{'='*80}\n\n")
        print(f"✓ 日志文件已创建: {log_filename}")
    except Exception as e:
        print(f"✗ 创建日志文件失败: {str(e)}")
        print(f"  日志路径: {log_filename}")
        exit(1)
    
    # 查找可用端口
    try:
        master_port = find_free_port()
        print(f"✓ 找到可用端口: {master_port}")
    except Exception as e:
        print(f"✗ 查找可用端口失败: {str(e)}")
        print("  使用默认端口: 12355")
        master_port = 12355
    
    print("\n" + "="*70)
    print("  逻辑树推理程序 (混合并行) / Logic Tree Inference (Hybrid Parallel)")
    print("="*70)
    print(f"  模型 (Model)         : {args.model}")
    print(f"  数据集 (Dataset)     : {args.dataset}")
    print(f"  起始索引 (Start)     : {args.start_index}")
    print(f"  样本数 (Samples)     : {args.test_size if args.test_size != -1 else 'All from start_index'}")
    print(f"  叶子数 (Leaves)      : {args.num_leaves}")
    print(f"  阈值τ (Tau)         : {args.tau}%")
    print(f"  总GPU数 (Total GPUs) : {args.world_size}")
    print(f"  模型并行 (MP Size)   : {args.model_parallel_size} GPUs/model")
    print(f"  数据并行 (DP Size)   : {data_parallel_size} groups")
    print(f"  主端口 (Master Port) : {master_port}")
    print(f"  日志文件 (Log)       : {log_filename}")
    print("="*70)
    print(f"\n并行策略说明:")
    print(f"  • 模型并行: 每{args.model_parallel_size}个GPU共享一个模型实例")
    print(f"  • 数据并行: 共有{data_parallel_size}组模型并行处理数据")
    print(f"  • GPU分组:")
    for dp_rank in range(data_parallel_size):
        gpu_start = dp_rank * args.model_parallel_size
        gpu_end = (dp_rank + 1) * args.model_parallel_size - 1
        print(f"    组 {dp_rank}: GPU {gpu_start}-{gpu_end} (共{args.model_parallel_size}个GPU)")
    print(f"\n其他特性:")
    print(f"  • 只显示组0的进度条，其他组在后台并行工作")
    print(f"  • 每个GPU都有独立的日志文件")
    print(f"  • 单个样本失败不会中断整个流程")
    print(f"  • 失败样本会被标记并单独保存到 _ERRORS.json 文件")
    print(f"  • 实时保存: 每处理完1个样本就保存到临时文件")
    print(f"  • 预计加速比: ~{data_parallel_size}x (数据并行)\n")
    
    # 启动推理进程
    try:
        mp.spawn(run_inference, args=(args.world_size, args, log_filename, master_port), 
                 nprocs=args.world_size, join=True)
    except Exception as e:
        error_msg = f"\n{'='*70}\n✗ 推理过程出错 / Inference failed\n{'='*70}\n"
        error_msg += f"错误信息: {str(e)}\n"
        error_msg += f"详细堆栈:\n{traceback.format_exc()}\n"
        error_msg += f"{'='*70}\n"
        
        print(error_msg)
        
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(error_msg)
        except:
            pass
        
        print(f"\n请查看日志文件了解详情: {log_filename}")
        print(f"⚠ 注意: 即使程序崩溃，已处理的结果仍保存在 ./temp_results/ 目录")
        print(f"  可以从临时文件手动恢复数据\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n" + "="*70)
    print(f"  ✓ 推理完成! / Inference completed!")
    print(f"  总耗时 (Total time)  : {elapsed_time:.2f} 秒")
    print(f"  平均速度 (Avg/group) : {elapsed_time/data_parallel_size:.2f} 秒/组")
    print("="*70 + "\n")