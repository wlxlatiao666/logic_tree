python generate.py --model Qwen3-8B --dataset gpqa-diamond --num_leaves 20 --device 1
python generate.py --model Qwen3-8B --dataset gpqa-main --num_leaves 20 --device 1
python generate_multi.py --model Qwen3-8B --dataset gpqa-diamond --samples 20 --device 1
python generate_multi.py --model Qwen3-8B --dataset gpqa-main --samples 20 --device 1