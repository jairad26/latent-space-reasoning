import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huginn_modified import modify_model_for_intermediates, analyze_intermediate_states
import chromadb
from memory import MemorySystem

# Create/clear debug log file
with open('debuglog.txt', 'w') as f:
    f.write("=== Starting New Generation Session ===\n")

# Load and modify model
model = AutoModelForCausalLM.from_pretrained(
    "tomg-group-umd/huginn-0125", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
model = modify_model_for_intermediates(model)

# Initialize Chroma client and memory system
client = chromadb.PersistentClient("./huginn_db")
memory = MemorySystem(client)

# Set up device and model
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.eval()
model.to(device)

input_ids = tokenizer.encode("The capital of Westphalia is", return_tensors="pt", add_special_tokens=True).to(device)
outputs = model.generate_minimal(
    input_ids,
    num_steps=16,
    continuous_compute=True
)

generated_text = tokenizer.decode(outputs[0])
with open('debuglog.txt', 'a') as f:
    f.write(f"\nGenerated text: {generated_text}\n")
