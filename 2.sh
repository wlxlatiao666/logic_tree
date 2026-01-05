# python metrics.py --model Qwen2.5-7B-Instruct --dataset math500 --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-7B-Instruct --dataset math500 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-7B-Instruct --dataset gpqa-main --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-7B-Instruct --dataset gpqa-main --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-7B-Instruct --dataset gpqa-diamond --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-7B-Instruct --dataset gpqa-diamond --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-7B-Instruct --dataset scibench --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-7B-Instruct --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-14B-Instruct --dataset math500 --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-14B-Instruct --dataset math500 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-14B-Instruct --dataset gpqa-main --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-14B-Instruct --dataset gpqa-main --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-14B-Instruct --dataset gpqa-diamond --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-14B-Instruct --dataset gpqa-diamond --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-14B-Instruct --dataset scibench --file_name logic_tree_results_all_leaves20_threshold80_ddp_relevance.json
# python evaluate.py --model Qwen2.5-14B-Instruct --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_ddp_relevance_metrics.json
# python metrics.py --model Qwen2.5-32B-Instruct --dataset math500 --file_name logic_tree_results_all_leaves20_threshold80_tpdp.json
# python evaluate.py --model Qwen2.5-32B-Instruct --dataset math500 --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_tpdp_metrics.json
python metrics.py --model Qwen2.5-32B-Instruct --dataset gpqa-main --file_name logic_tree_results_all_leaves20_threshold80_tpdp.json
python evaluate.py --model Qwen2.5-32B-Instruct --dataset gpqa-main --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_tpdp_metrics.json
# python metrics.py --model Qwen2.5-32B-Instruct --dataset gpqa-diamond --file_name logic_tree_results_all_leaves20_threshold80_tpdp.json
# python evaluate.py --model Qwen2.5-32B-Instruct --dataset gpqa-diamond --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_tpdp_metrics.json
python metrics.py --model Qwen2.5-32B-Instruct --dataset scibench --file_name logic_tree_results_all_leaves20_threshold80_tpdp.json
python evaluate.py --model Qwen2.5-32B-Instruct --dataset scibench --sc_file generated_answers_20samples_ray_8gpus.json --lt_file logic_tree_results_all_leaves20_threshold80_tpdp_metrics.json
