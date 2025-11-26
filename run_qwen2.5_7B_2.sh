CUDA_VISIBLE_DEVICES=3
python generate_multi.py --model Qwen2.5-7B-Instruct --dataset csqa --samples 20 --device 3
python generate.py --model Qwen2.5-7B-Instruct --dataset csqa --num_leaves 20 --device 3
python generate_multi.py --model Qwen2.5-7B-Instruct --dataset arc-e --samples 20 --device 3
python generate.py --model Qwen2.5-7B-Instruct --dataset arc-e --num_leaves 20 --device 3
python generate_multi.py --model Qwen2.5-7B-Instruct --dataset arc-c --samples 20 --device 3
python generate.py --model Qwen2.5-7B-Instruct --dataset arc-c --num_leaves 20 --device 3

