from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import sys
import json
import torch
from logic_tree_decode import logic_branch_decode

sys.stdout = open('output.log', 'w', encoding='utf-8')
# set_seed(42)

MODEL_NAME = "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print("eos_token: ", tokenizer.eos_token)

input_path = '/mnt/public/gpfs-jd/code/weilongxuan/uncertainty/data/gsm8k/test.json'
output_path = './uncertainty_results_7B_gsm8k.json'

with open(input_path, 'r') as f:
    test_data = json.load(f)  # 取前100条

with open('sys_prompt.txt', 'r') as f:
    system_prompt = f.read().strip()

results = []
for i, item in enumerate(test_data):
    # 构建Qwen2.5格式prompt
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
    # 调用解码逻辑
    root, leaves, U, complexity = logic_branch_decode(tokenizer, model, prompt=prompt)
        
    # 获取贪婪解码结果（取第一个leaf）
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    greedy_output = model.generate(
        inputs.input_ids,
        max_length=1024,
        num_beams=1,
        do_sample=False
    )
    greedy_answer = tokenizer.decode(greedy_output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
    max_leaf = max(leaves, key=lambda x: x.prob)
    single_entropy = -max_leaf.cum_logprob / max_leaf.length
    # 记录结果
    results.append({
        "question": item['question'],
        "gt_answer": item['answer'],
        "complexity": complexity,
        "logic_uncertainty": U,
        "single_uncertainty": float(single_entropy),
        "greedy_answer": greedy_answer
    })
    print(f"Processed {i+1}/100 | U={U:.2f}")

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"\nResults saved to {output_path}")