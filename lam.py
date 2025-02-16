import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from utils import precompute_freqs_cis

class LearnedAttentionAggregator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # A learnable query vector to compute attention scores over the token embeddings
        self.query = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        # Compute attention scores for each token by a dot product with the query vector
        attn_scores = torch.matmul(x, self.query)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len)
        # Compute the weighted sum over tokens to produce a single vector per example
        aggregated = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # (batch_size, hidden_size)
        return aggregated, attn_weights
    
def get_aggregated_vector(model, tokenizer, text, aggregator):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    
    with torch.no_grad():
        # Get initial token embeddings from the model's embedding layer
        hidden_states = model.transformer.wte(input_ids)  # (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = hidden_states.shape
        
        # Precompute RoPE frequencies (if your prelude layers use them)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            theta=50000.0,
            condense_ratio=1
        ).to(device)
        
        # Process through the prelude layers
        for i, block in enumerate(model.transformer.prelude):
            hidden_states = block(hidden_states, freqs_cis=freqs_cis, step_idx=i)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        
        # Now hidden_states holds the post-prelude representations for every token.
        # Apply the learned attention aggregator to get a single vector.
        aggregated_vector, attention_weights = aggregator(hidden_states)  # (batch_size, hidden_size)
    return aggregated_vector, attention_weights