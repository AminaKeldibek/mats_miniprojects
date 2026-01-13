from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Tuple

class GPTStudyModel:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize a GPT-2 model and tokenizer from Hugging Face.
        
        Args:
            model_name (str): The name of the GPT-2 model variant to load.
                             Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        """
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Loaded {model_name} on {self.device}")
    
    def generate_text(self, prompt, max_length=10, temperature=1.0, do_sample=False):
        """
        Generate text using the loaded GPT-2 model.
        
        Args:
            prompt (str): Input text prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text
        """
    
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Generate with safer parameters
        with torch.no_grad():
            gen_kwargs = {
                **inputs,
                "max_length": max_length,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            # Only add temperature if sampling is enabled
            if do_sample:
                gen_kwargs["temperature"] = temperature

            outputs = self.model.generate(**gen_kwargs)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text


    def generate_with_saving_hook(self, layer: int, prompt: str, max_length=100, temperature=1.0, do_sample=True):
        """
        Generate text while capturing activations from a specific layer.
        
        Args:
            layer (int): Layer index to capture activations from
            prompt (str): Input text prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            list: List of captured activations
        """
        def saving_hook(module, input, output: Tuple[torch.Tensor]):
            captured_activations.append(output[0].detach())

        captured_activations = []
        
        target_layer = self.model.h[layer]
        hook_handle = target_layer.register_forward_hook(saving_hook)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hook_handle.remove()

        return captured_activations


