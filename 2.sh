python metrics.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathqa --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathcloze --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset svamp --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset math500 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset aime24 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset aime25 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset gpqa-main --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset gpqa-diamond --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset scibench --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen2.5-7B-Instruct --dataset csqa --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathqa --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset gaokao-mathcloze --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset svamp --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset math500 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset aime24 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset aime25 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset gpqa-main --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset gpqa-diamond --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen2.5-7B-Instruct --dataset csqa --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python metrics.py --model Qwen3-8B --dataset gaokao-mathqa --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset gaokao-mathcloze --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset svamp --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset math500 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset aime24 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset aime25 --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset gpqa-main --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset gpqa-diamond --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset scibench --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python metrics.py --model Qwen3-8B --dataset csqa --file_name logic_tree_results_all_leaves20_threshold80_ddp.json
python evaluate.py --model Qwen3-8B --dataset gaokao-mathqa --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset gaokao-mathcloze --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset svamp --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset math500 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset aime24 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset aime25 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset gpqa-main --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset gpqa-diamond --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json
python evaluate.py --model Qwen3-8B --dataset csqa --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_metrics.json

