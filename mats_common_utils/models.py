import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device}")

model_name = "Qwen/Qwen3-0.6B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the difference between dense and sparse models in a simple way."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=256
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n✅ Response from model:")
print(response)
