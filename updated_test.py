import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from huginn_modified import modify_model_for_intermediates, analyze_intermediate_states

# Load and modify model
model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
model = modify_model_for_intermediates(model)

# Analyze states
logits, states = analyze_intermediate_states(model, tokenizer, "The capital of Westphalia is", num_steps=16)

print(logits.shape)
print(states.shape)