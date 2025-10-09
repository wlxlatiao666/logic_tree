from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import torch.nn.functional as F

input_path = '/mnt/public/gpfs-jd/code/weilongxuan/uncertainty/data/gsm8k/test.json'
output_path = '../results/generated_answers_greedy.json'

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct",
    device_map='auto',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct",
    device_map='auto',
    trust_remote_code=True
)

with open(input_path, 'r') as f:
    test_data = json.load(f)  # 取前100条

with open('../sys_prompt.txt', 'r') as f:
    system_prompt = f.read().strip()

results = []
for item in test_data:
    query = item['question']
    dialog = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(
        dialog,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # 执行贪婪解码
    model.eval()
    outputs = model.generate(
        **inputs,
        max_length=1024,
        do_sample=False,  # 关闭采样
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 提取生成结果
    generated_ids = outputs.sequences
    generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

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

    results.append({
        "question": item['question'],
        "gt_answer": item['answer'],
        "greedy_answer": generated_text,
        "avg_logprob": avg_logprob
    })

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)