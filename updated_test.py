import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huginn_modified import modify_model_for_intermediates, analyze_intermediate_states
import chromadb

# Load and modify model
model = AutoModelForCausalLM.from_pretrained(
    "tomg-group-umd/huginn-0125", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
model = modify_model_for_intermediates(model)

# Initialize Chroma client
client = chromadb.PersistentClient("./huginn_db")

text = "The capital of Westphalia is"
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.eval()
model.to(device)

# Analyze states using the modified forward (with return_intermediates)
logits, states, lam_vector, attention_weights = analyze_intermediate_states(client, model, tokenizer, text, num_steps=16)
print("Logits shape:", logits.shape)  # Expected shape: (batch_size, seq_len, vocab_size)
print("States shape:", states.shape)    # Expected shape: (batch_size, num_steps, seq_len, hidden_size)
print("Lam vector shape:", lam_vector.shape)
print("Attention weights shape:", attention_weights.shape)

# Use the model's generate method to produce additional tokens.
# With our updated new_forward, generate will use the original forward.
input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
outputs = model.generate(input_ids, num_steps=16)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated output text:", output_text)

# Optional checks
assert torch.any(logits != 0), "Logits are all zeros. There might be an issue with the model."
first_token_states = states[0, :, 0, :]  # Shape: (num_steps, hidden_size)
norm_diffs = torch.norm(first_token_states[1:] - first_token_states[:-1], dim=1)
print("Norm differences between consecutive states for the first token:")
print(norm_diffs)
assert torch.any(norm_diffs > 1e-3), "Latent states are not evolving as expected."
