import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

device = torch.device("mps" if torch.mps.is_available() else "cpu")

model.eval()
model.to(device)

input_ids = tokenizer.encode("The capital of Westphalia is", return_tensors="pt", add_special_tokens=True).to(device)
outputs = model.generate(input_ids, tokenizer=tokenizer, num_steps=16)

print("raw output: ", model(input_ids, num_steps=16, use_cache=True))

print("raw output logits: ", model(input_ids, num_steps=16, use_cache=True).logits.shape)
print("raw output latent: ", model(input_ids, num_steps=16, use_cache=True).latent_states.shape)

print("=========================")

print("latent output: ", tokenizer.decode(outputs[0], skip_special_tokens=True))


