import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huginn_modified import modify_model_for_intermediates, analyze_intermediate_states
import chromadb
from memory import MemorySystem

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

def test_model_output(text: str, num_steps: int = 16):
    """Test the model with intermediate state analysis and generate readable output."""
    print(f"\nTesting with input: '{text}'")
    
    # Get analysis of intermediate states
    logits, states, lam_vector, attention_weights = analyze_intermediate_states(
        client, model, tokenizer, text, num_steps=num_steps, memory=memory
    )
    
    # Generate continuation using the model
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=50,  # Adjust as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print("\nModel Output:")
    print(f"Complete generated text: {generated_text}")
    
    # Print some analysis of the intermediate states
    print("\nIntermediate State Analysis:")
    print(f"Number of steps analyzed: {states.shape[1]}")
    print(f"Sequence length: {states.shape[2]}")
    print(f"Hidden state dimension: {states.shape[3]}")
    
    # Analyze state changes for each token
    for token_idx in range(input_ids.shape[1]):
        token = tokenizer.decode(input_ids[0, token_idx])
        token_states = states[0, :, token_idx, :]
        state_changes = torch.norm(token_states[1:] - token_states[:-1], dim=1)
        
        print(f"\nToken {token_idx} ('{token}')")
        print("Average state change:", state_changes.mean().item())
        print("Max state change:", state_changes.max().item())
    
    return generated_text, states

if __name__ == "__main__":
    # Test with different prompts
    test_cases = [
        "2 plus 2 is",
    ]
    
    for test_case in test_cases:
        generated_text, _ = test_model_output(test_case)