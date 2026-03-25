# Load model directly
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct")
model = model.to("cuda")
while True:
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Who are you?"},
	]
	inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_dict=True,
		return_tensors="pt",
	).to(model.device)

	prompt = tokenizer.decode(inputs["input_ids"][0])
# print(prompt)
	outputs = model.generate(**inputs, max_new_tokens=40)
	print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
	# time.sleep(1800)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))