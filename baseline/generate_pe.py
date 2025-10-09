import json
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

import sys

sys.stdout = open(f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w', encoding='utf-8')

start_time = time.time()

# 配置参数
NUM_SAMPLES = 5
MODEL_NAME = '/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct'
INPUT_FILE = '../data/gsm8k/test.json'
OUTPUT_FILE = f'../results_gsm8k/generated_answers_{NUM_SAMPLES}samples1.json'

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map='auto', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', trust_remote_code=True)

with open('../sys_prompt.txt', 'r') as f:
    system_prompt = f.read().strip()

# 读取输入数据
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

results = []
for item in data:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": item['question']}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    sampled_answers = []
    sampled_entropies = []
    for _ in range(NUM_SAMPLES):
        outputs = model.generate(
            **inputs,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.90,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = outputs.sequences
        generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        sampled_answers.append(generated_text)

        # 计算对数概率
        neg_logprobs = []
        input_length = inputs["input_ids"].shape[1]

        for step_idx, step_scores in enumerate(outputs.scores):
            # 获取当前步骤生成的token ID
            token_id = generated_ids[0, input_length + step_idx]
            
            # 计算对数概率
            step_logprobs = F.log_softmax(step_scores[0], dim=-1)
            logprob = step_logprobs[token_id].item()
            neg_logprobs.append(-logprob)

        # 计算平均对数概率
        avg_logprob = sum(neg_logprobs) / len(neg_logprobs) if neg_logprobs else 0.0

        sampled_entropies.append(avg_logprob)
    
    # 生成greedy解码答案
    greedy_output = model.generate(
        **inputs,
        max_length=1024,
        num_beams=1,
        do_sample=False
    )
    most_likely_answer = tokenizer.decode(greedy_output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    results.append({
        "question": item['question'],
        "gt_answer": item['answer'],
        "sampled_answers": sampled_answers,
        "sampled_entropies": sampled_entropies,
        "greedy_answer": most_likely_answer
    })

# 保存结果
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

end_time = time.time()
duration = end_time - start_time
print(f"\n[运行统计] 总耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")