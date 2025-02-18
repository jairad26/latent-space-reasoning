import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from typing import Optional, Any, Union
import os
from transformers import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput
import torch.nn.functional as F
import math
from custom_model import modify_model_for_intermediates
import time
def setup_model():
    model = AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Modify the model to capture intermediates
    model = modify_model_for_intermediates(model)
    return model, tokenizer, device

def generate_with_memory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    num_steps: int = 16,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate with the model
    outputs = model.generate(
        input_ids,
        generation_config=generation_config,
        tokenizer=tokenizer,  # Pass tokenizer to the model
        num_steps=num_steps,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Get pattern statistics
    pattern_stats = model.store.get_pattern_statistics()
    
    return {
        'text': generated_text,
        'pattern_stats': pattern_stats,
    }
    
def print_model_structure(model):
    """Print the model's structure to find the recurrent block"""
    for name, module in model.named_modules():
        print(f"Module name: {name}")
        print(f"Module type: {type(module)}")
        print("-" * 50)

# Add this to your mem_huginn.py
def inspect_model(model):
    print("Model structure:")
    print_model_structure(model)
    
    # Also print direct attributes
    print("\nDirect attributes:")
    for attr in dir(model):
        if not attr.startswith('_'):  # Skip private attributes
            print(attr)

def main():
    # Setup
    model, tokenizer, device = setup_model()
    
    # inspect_model(model)
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement:",
        "2+2=",
    ]
    
    # Generate for each prompt
    time_start = time.time()
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = generate_with_memory(model, tokenizer, prompt, device)
        print(f"Generated: {result['text']}")
        print(f"Pattern Statistics: {result['pattern_stats']}")
        
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    main()