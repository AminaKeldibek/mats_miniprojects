import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device}")

# FIXED: Changed to Qwen2.5-14B-Instruct
model_name = "Qwen/Qwen2.5-14B-Instruct"

# Configuration for 4-bit quantization to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"--> Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"--> Loading model: {model_name} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

print("--> Model loaded successfully! Running inference...")

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
