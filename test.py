# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/gpt-oss-20b")
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
print(prompt)
outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))