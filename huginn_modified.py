import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lam import LearnedAttentionAggregator
from utils import precompute_freqs_cis
import chromadb
from memory import MemorySystem
from typing import Optional
import os
def modify_model_for_intermediates(model):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.lam = LearnedAttentionAggregator(model.config.hidden_size).to(device)
    if os.path.exists("lam_state.pt"):
        model.lam.load_state()
    else:
        model.lam.save_state()
    original_forward = model.forward
    
    def new_forward(self, input_ids, text: Optional[str] = None, num_steps=32, return_intermediates=False, use_cache=False, return_lam=False, memory: Optional[MemorySystem] = None, **kwargs):
        # If caching is requested (as in generation), use the original forward method.
        if use_cache:
            # Remove our extra keyword arguments before calling the original forward.
            kwargs.pop("num_steps", None)
            kwargs.pop("return_intermediates", None)
            kwargs.pop("return_lam", None)
            kwargs.pop("memory", None)
            return original_forward(input_ids, **kwargs)

        # Otherwise, run our modified forward code.
        # Get the initial embeddings
        hidden_states = self.transformer.wte(input_ids)

        # Generate RoPE cache
        batch_size, seq_len = input_ids.shape
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            theta=50000.0,  # Base from the paper
            condense_ratio=1
        ).to(input_ids.device)

        # Apply prelude layers
        for i, block in enumerate(self.transformer.prelude):
            hidden_states = block(hidden_states, freqs_cis=freqs_cis, step_idx=i)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
                
        # Apply learned attention aggregator
        lam_vector, attention_weights = self.lam(hidden_states)
        
        if memory:
            query_text, collection_name = memory.find_similar_query(lam_vector)
            if query_text:
                cached_states = memory.get_cached_states(collection_name)
                print(f"Found cached states for query: {query_text}")
                print(f"Collection name: {collection_name}")
                print(f"Cached states: {cached_states}")

        # Initialize random state with correct variance
        random_state = torch.randn(
            batch_size, seq_len, self.config.hidden_size,
            device=input_ids.device
        ) * (2/5)**0.5

        # Concatenate embeddings with state for adapter
        adapter_input = torch.cat([hidden_states, random_state], dim=-1)
        current_state = self.transformer.adapter(adapter_input)

        all_states = [] if return_intermediates else None

        # Run recurrent steps through core_block
        for step in range(num_steps):
            adapter_input = torch.cat([hidden_states, current_state], dim=-1)
            next_state = self.transformer.adapter(adapter_input)
            for i, block in enumerate(self.transformer.core_block):
                next_state = block(next_state, freqs_cis=freqs_cis, step_idx=step*len(self.transformer.core_block) + i)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
            current_state = next_state
            if return_intermediates:
                all_states.append(current_state.detach().clone())

        # Apply coda layers
        for i, block in enumerate(self.transformer.coda):
            current_state = block(current_state, freqs_cis=freqs_cis, step_idx=num_steps*len(self.transformer.core_block) + i)
            if isinstance(current_state, tuple):
                current_state = current_state[0]

        # Final layer norm and project to vocabulary
        current_state = self.transformer.ln_f(current_state)
        logits = self.lm_head(current_state)
        all_states = torch.stack(all_states, dim=1)
        
        if memory:
            memory.cache_query(text, lam_vector, all_states)

        if return_intermediates and return_lam:
            return logits, all_states, lam_vector, attention_weights
        elif return_intermediates:
            return logits, all_states
        elif return_lam:
            return logits, lam_vector, attention_weights
        else:
            return logits

    model.forward = new_forward.__get__(model)
    return model

def analyze_intermediate_states(client, model, tokenizer, text, num_steps=16, memory: Optional[MemorySystem] = None):
    # Set up device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare input
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Run model with intermediate states
    with torch.no_grad():
        logits, all_states, lam_vector, attention_weights = model(input_ids, text=text, num_steps=num_steps, return_intermediates=True, return_lam=True, memory=memory)
    
    # Print shapes and analysis
    print(f"Input shape: {input_ids.shape}")
    print(f"All states shape: {all_states.shape}")
    print(f"Final logits shape: {logits.shape}")
    print(f"Lam vector shape: {lam_vector.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Analyze state changes
    for token_idx in range(input_ids.shape[1]):
        token = tokenizer.decode(input_ids[0, token_idx])
        token_states = all_states[0, :, token_idx, :]
        state_changes = torch.norm(token_states[1:] - token_states[:-1], dim=1)
        
        print(f"\nToken {token_idx} ('{token}')")
        print("State changes between steps:")
        for step, change in enumerate(state_changes):
            print(f"Step {step+1} -> {step+2}: {change.item():.4f}")
    
    return logits, all_states, lam_vector, attention_weights

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    
    model = modify_model_for_intermediates(model)
    
    text = "The capital of Westphalia is"
    memory = MemorySystem(chromadb.PersistentClient("./huginn_db"))
    logits, states = analyze_intermediate_states(model, tokenizer, text, num_steps=16, memory=memory)