from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the original base model
model_name = "t5-small"
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
# Load the LoRA adapter and apply it to the base model
peft_model = PeftModel.from_pretrained(base_model, "./math_encoder_lora/checkpoint-150")

# Put the model in evaluation mode (important for inference)
peft_model.eval()

# You can move the model to your Mac's GPU if you have one
# device = torch.device("mps")
# peft_model.to(device)

def ask_question(question, model, tokenizer):
    """Takes a question string and returns the model's decoded output."""
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    # input_ids = input_ids.to(device) # Move to device if using one

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=50)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# Now we can test a whole list of questions easily
test_questions = [
    "What is 6 - 5?",  # Expected answer: one
    "What is 8 / 4?",  # Expected answer: two
    "What is 3 + 4?",  # Expected answer: seven
]

print("--- Testing Model ---")
for question in test_questions:
    output = ask_question(question, peft_model, tokenizer)
    print(f"Input: {question}")
    print(f"Output: {output}\n")