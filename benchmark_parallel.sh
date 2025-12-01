#!/bin/bash

# vLLM并行推理性能测试脚本
# 用于对比不同并行方案的性能

set -e

# 配置参数
MODEL="Qwen2.5-7B-Instruct"
DATASET="gsm8k"
TEST_SIZE=20  # 使用小数据集进行测试
SAMPLES=5

echo "=========================================="
echo "vLLM 并行推理性能测试"
echo "=========================================="
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "测试数据量: $TEST_SIZE"
echo "每个样本数: $SAMPLES"
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p ./logs
mkdir -p ./results

# 测试1：原版单GPU
echo "[测试1] 原版单GPU (baseline)"
echo "命令: python generate_multi_vllm.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --device 0"
echo "开始时间: $(date)"
START_TIME=$(date +%s)

python generate_multi_vllm.py \
    --model $MODEL \
    --dataset $DATASET \
    --test_size $TEST_SIZE \
    --samples $SAMPLES \
    --device 0 2>&1 | tee ./logs/test1_original.log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "完成时间: $(date)"
echo "耗时: ${DURATION}秒"
echo ""
echo "=========================================="
echo ""

# 测试2：张量并行版本（单GPU，批量生成优化）
echo "[测试2] 张量并行版本 - 单GPU批量生成"
echo "命令: python generate_multi_vllm_parallel.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --tensor_parallel_size 1"
echo "开始时间: $(date)"
START_TIME=$(date +%s)

python generate_multi_vllm_parallel.py \
    --model $MODEL \
    --dataset $DATASET \
    --test_size $TEST_SIZE \
    --samples $SAMPLES \
    --tensor_parallel_size 1 2>&1 | tee ./logs/test2_parallel_tp1.log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "完成时间: $(date)"
echo "耗时: ${DURATION}秒"
echo ""
echo "=========================================="
echo ""

# 检查GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU"
echo ""

if [ $NUM_GPUS -ge 2 ]; then
    # 测试3：张量并行版本（2 GPU）
    echo "[测试3] 张量并行版本 - 2 GPU"
    echo "命令: python generate_multi_vllm_parallel.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --tensor_parallel_size 2"
    echo "开始时间: $(date)"
    START_TIME=$(date +%s)

    python generate_multi_vllm_parallel.py \
        --model $MODEL \
        --dataset $DATASET \
        --test_size $TEST_SIZE \
        --samples $SAMPLES \
        --tensor_parallel_size 2 2>&1 | tee ./logs/test3_parallel_tp2.log

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "完成时间: $(date)"
    echo "耗时: ${DURATION}秒"
    echo ""
    echo "=========================================="
    echo ""

    # 测试4：数据并行版本（2 GPU）
    echo "[测试4] 数据并行版本 - 2 GPU (Ray)"
    echo "命令: python generate_multi_vllm_ray.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --num_gpus 2"
    echo "开始时间: $(date)"
    START_TIME=$(date +%s)

    python generate_multi_vllm_ray.py \
        --model $MODEL \
        --dataset $DATASET \
        --test_size $TEST_SIZE \
        --samples $SAMPLES \
        --num_gpus 2 2>&1 | tee ./logs/test4_ray_2gpus.log

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "完成时间: $(date)"
    echo "耗时: ${DURATION}秒"
    echo ""
    echo "=========================================="
    echo ""
fi

if [ $NUM_GPUS -ge 4 ]; then
    # 测试5：张量并行版本（4 GPU）
    echo "[测试5] 张量并行版本 - 4 GPU"
    echo "命令: python generate_multi_vllm_parallel.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --tensor_parallel_size 4"
    echo "开始时间: $(date)"
    START_TIME=$(date +%s)

    python generate_multi_vllm_parallel.py \
        --model $MODEL \
        --dataset $DATASET \
        --test_size $TEST_SIZE \
        --samples $SAMPLES \
        --tensor_parallel_size 4 2>&1 | tee ./logs/test5_parallel_tp4.log

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "完成时间: $(date)"
    echo "耗时: ${DURATION}秒"
    echo ""
    echo "=========================================="
    echo ""

    # 测试6：数据并行版本（4 GPU）
    echo "[测试6] 数据并行版本 - 4 GPU (Ray)"
    echo "命令: python generate_multi_vllm_ray.py --model $MODEL --dataset $DATASET --test_size $TEST_SIZE --samples $SAMPLES --num_gpus 4"
    echo "开始时间: $(date)"
    START_TIME=$(date +%s)

    python generate_multi_vllm_ray.py \
        --model $MODEL \
        --dataset $DATASET \
        --test_size $TEST_SIZE \
        --samples $SAMPLES \
        --num_gpus 4 2>&1 | tee ./logs/test6_ray_4gpus.log

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "完成时间: $(date)"
    echo "耗时: ${DURATION}秒"
    echo ""
    echo "=========================================="
    echo ""
fi

# 生成性能报告
echo ""
echo "=========================================="
echo "性能测试完成！"
echo "=========================================="
echo ""
echo "正在生成性能报告..."

python - <<EOF
import re
import os
from datetime import datetime

# 读取日志文件并提取耗时
def extract_time(log_file):
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 查找总耗时
    match = re.search(r'总耗时:\s*([\d.]+)秒', content)
    if match:
        return float(match.group(1))
    return None

# 测试配置
tests = [
    ("原版单GPU", "./logs/test1_original.log"),
    ("张量并行-1GPU", "./logs/test2_parallel_tp1.log"),
    ("张量并行-2GPU", "./logs/test3_parallel_tp2.log"),
    ("数据并行-2GPU", "./logs/test4_ray_2gpus.log"),
    ("张量并行-4GPU", "./logs/test5_parallel_tp4.log"),
    ("数据并行-4GPU", "./logs/test6_ray_4gpus.log"),
]

print("\n性能对比报告")
print("=" * 60)
print(f"{'测试方案':<20} {'耗时(秒)':<15} {'加速比':<10}")
print("-" * 60)

baseline_time = None
results = []

for name, log_file in tests:
    time = extract_time(log_file)
    if time is not None:
        if baseline_time is None:
            baseline_time = time
            speedup = 1.0
        else:
            speedup = baseline_time / time
        
        results.append((name, time, speedup))
        print(f"{name:<20} {time:<15.2f} {speedup:<10.2f}x")

print("=" * 60)

# 保存报告
report_file = f"./logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, 'w') as f:
    f.write("vLLM并行推理性能测试报告\n")
    f.write("=" * 60 + "\n")
    f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型: $MODEL\n")
    f.write(f"数据集: $DATASET\n")
    f.write(f"测试数据量: $TEST_SIZE\n")
    f.write(f"每个样本数: $SAMPLES\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"{'测试方案':<20} {'耗时(秒)':<15} {'加速比':<10}\n")
    f.write("-" * 60 + "\n")
    for name, time, speedup in results:
        f.write(f"{name:<20} {time:<15.2f} {speedup:<10.2f}x\n")
    f.write("=" * 60 + "\n")

print(f"\n报告已保存到: {report_file}")
EOF

echo ""
echo "所有测试完成！"
echo "详细日志保存在 ./logs/ 目录"
echo "结果文件保存在 ./results/ 目录"
