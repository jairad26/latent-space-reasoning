import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    """Precompute the frequency cache for RoPE."""
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)

def modify_model_for_intermediates(model):
    original_forward = model.forward
    
    def new_forward(self, input_ids, num_steps=32, return_intermediates=False, use_cache=False, **kwargs):
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
        
        # Apply prelude
        for i, block in enumerate(self.transformer.prelude):
            hidden_states = block(hidden_states, freqs_cis=freqs_cis, step_idx=i)

            # Extract only the tensor (ignore None)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0] 
            if isinstance(hidden_states, tuple):
                print(f"Tuple contents: {[type(h) for h in hidden_states]}")

        
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
            # Concatenate current embeddings with state
            adapter_input = torch.cat([hidden_states, current_state], dim=-1)
            next_state = self.transformer.adapter(adapter_input)
            
            # Apply core blocks
            for i, block in enumerate(self.transformer.core_block):
                next_state = block(next_state, freqs_cis=freqs_cis, step_idx=step*len(self.transformer.core_block) + i)
                
                # Extract only the tensor (ignore None)
                if isinstance(next_state, tuple):
                    next_state = next_state[0] 
            
            current_state = next_state
            
            if return_intermediates:
                all_states.append(current_state.detach().clone())
        
        # Apply coda blocks
        for i, block in enumerate(self.transformer.coda):
            current_state = block(current_state, freqs_cis=freqs_cis, step_idx=num_steps*len(self.transformer.core_block) + i)
            
            # Extract only the tensor (ignore None)
            if isinstance(current_state, tuple):
                current_state = current_state[0] 
            
        # Final layer norm
        current_state = self.transformer.ln_f(current_state)
        
        # Project to vocabulary
        logits = self.lm_head(current_state)
        
        if return_intermediates:
            all_states = torch.stack(all_states, dim=1)
            return logits, all_states
        return logits

    # Replace the forward method
    model.forward = new_forward.__get__(model)
    return model

def analyze_intermediate_states(model, tokenizer, text, num_steps=16):
    # Set up device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare input
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Run model with intermediate states
    with torch.no_grad():
        logits, all_states = model(input_ids, num_steps=num_steps, return_intermediates=True)
    
    # Print shapes and analysis
    print(f"Input shape: {input_ids.shape}")
    print(f"All states shape: {all_states.shape}")
    print(f"Final logits shape: {logits.shape}")
    
    # Analyze state changes
    for token_idx in range(input_ids.shape[1]):
        token = tokenizer.decode(input_ids[0, token_idx])
        token_states = all_states[0, :, token_idx, :]
        state_changes = torch.norm(token_states[1:] - token_states[:-1], dim=1)
        
        print(f"\nToken {token_idx} ('{token}')")
        print("State changes between steps:")
        for step, change in enumerate(state_changes):
            print(f"Step {step+1} -> {step+2}: {change.item():.4f}")
    
    return logits, all_states

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    
    model = modify_model_for_intermediates(model)
    
    text = "The capital of Westphalia is"
    logits, states = analyze_intermediate_states(model, tokenizer, text, num_steps=16)