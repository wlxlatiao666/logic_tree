python avg_tokens.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathqa --sc_file generated_answers_20samples_ray_8gpus.json
python pass_curve.py --model Qwen3-8B --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp.json
python pass_curve.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathqa --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp.json
