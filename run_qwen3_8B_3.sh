CUDA_VISIBLE_DEVICES=2
python generate_multi.py --model Qwen3-8B --dataset csqa --samples 20
python generate.py --model Qwen3-8B --dataset csqa --num_leaves 20
python generate_multi.py --model Qwen3-8B --dataset arc-e --samples 20
python generate.py --model Qwen3-8B --dataset arc-e --num_leaves 20
python generate_multi.py --model Qwen3-8B --dataset arc-c --samples 20
python generate.py --model Qwen3-8B --dataset arc-c --num_leaves 20

