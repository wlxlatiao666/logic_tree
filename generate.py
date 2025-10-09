import torch
import json
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from logic_tree_decode import logic_branch_decode

def generate_usr_prompt(dataset: str, item: dict) -> str:
    if dataset == "gsm8k":
        usr_prompt = item["question"]
    elif dataset == "reclor":
        usr_prompt = "Context: " + item['context'] + "\nQuestion: " + item['question'] + \
            "\nA. " + item['answers'][0] + \
            "\nB. " + item['answers'][1] + \
            "\nC. " + item['answers'][2] + \
            "\nD. " + item['answers'][3]
        print("usr_prompt: ", usr_prompt)
    else:
        raise ValueError(f"dataset {dataset} not supported")
    return usr_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()

    # load model
    MODEL_NAME = "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load system prompt
    dataset = args.dataset
    test_size = args.test_size
    dataset_path = f"./data/{dataset}/test.json"
    with open(dataset_path, "r") as f:
        if test_size == -1:
            data = json.load(f)
        else:
            data = json.load(f)[:test_size]
    with open(f"./sys_prompt.json", "r") as f:
        sys_prompt = json.load(f)[dataset]

    # generate
    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        usr_prompt = generate_usr_prompt(dataset, item)
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"

        root, leaves, new_tokens_cnt = logic_branch_decode(tokenizer, model, prompt=prompt, sample=True, M=3)

        complexity = sum(leaf.prob * leaf.depth for leaf in leaves)
        probs = [leaf.prob for leaf in leaves]
        texts = [leaf.text for leaf in leaves]
        entropies = [-leaf.cum_logprob / leaf.length if leaf.length > 0 else 0.0 for leaf in leaves]
        results.append({
            "original_data": item,
            "num_new_tokens": new_tokens_cnt,
            "num_leaves": len(leaves),
            "complexity": complexity,
            "probs": probs,
            "entropies": entropies,
            "texts": texts
        })
    end_time = time.time()
    duration = end_time - start_time
    print(f"generate {len(results)} results in {duration:.2f} seconds({duration/60:.2f} minutes)")

    if test_size == -1:
        output_path = f"./results/{dataset}/logic_tree_results_all.json"
    else:
        output_path = f"./results/{dataset}/logic_tree_results_{test_size}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)