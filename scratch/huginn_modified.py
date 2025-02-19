import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lam import LearnedAttentionAggregator
from utils import precompute_freqs_cis
import chromadb
from memory import MemorySystem
from typing import Optional, Any, Union
import os
from state_transfer import StateTransferModule
from transformers import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput
import torch.nn.functional as F
from huginn_dynamic_cache import HuginnDynamicCache
import math

def modify_model_for_intermediates(model, residual_scale=0.3, adapter_scale=1.0):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.lam = LearnedAttentionAggregator(model.config.hidden_size).to(device)
    original_forward = model.forward

    @torch.no_grad()
    def new_generate_minimal(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,
        tokenizer=None,
        streamer=None,
        continuous_compute: bool = False,
        num_steps: int = 16,
        cache_kwargs: dict = {},
        **model_kwargs,
    ) -> Union[torch.Tensor, dict[str, Any]]:
        with open('debuglog.txt', 'a') as f:
            f.write("USING NEW GENERATE MINIMAL\n")
        if generation_config is None:
            generation_config = self.generation_config

        # Default sampling parameters.
        if not hasattr(generation_config, 'do_sample') or generation_config.do_sample is None:
            generation_config.do_sample = True
        if not hasattr(generation_config, 'temperature') or generation_config.temperature is None:
            generation_config.temperature = 0.7
        if not hasattr(generation_config, 'top_k') or generation_config.top_k is None:
            generation_config.top_k = 50
        if not hasattr(generation_config, 'top_p') or generation_config.top_p is None:
            generation_config.top_p = 0.95
        if not hasattr(generation_config, 'repetition_penalty') or generation_config.repetition_penalty is None:
            generation_config.repetition_penalty = 1.2

        model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
        model_kwargs["use_cache"] = False
        model_kwargs["return_intermediates"] = True
        model_kwargs["num_steps"] = num_steps
        model_kwargs["return_lam"] = True
        model_kwargs["memory"] = None
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)

        if continuous_compute:
            embedded_inputs, _, _ = self.embed_inputs(input_ids)
            model_kwargs["input_states"] = self.initialize_state(embedded_inputs)

        # Dynamic Temperature Annealing
        initial_temperature = generation_config.temperature
        annealing_rate = 0.01  # Adjust as needed
        min_temperature = 0.2

        for i in range(generation_config.max_length - input_ids.shape[1]):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)
            next_token_logits = outputs.logits[0, -1, :]

            # Repetition penalty
            if generation_config.repetition_penalty != 1.0:
                generated_tokens = set(input_ids.squeeze().tolist())
                for token_id in generated_tokens:
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= generation_config.repetition_penalty
                    else:
                        next_token_logits[token_id] /= generation_config.repetition_penalty

            if continuous_compute:
                current_last_latent = outputs.latent_states[:, -1:, :]

            # Apply temperature annealing
            temperature = max(initial_temperature * math.exp(-annealing_rate * i), min_temperature)
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Gen] Temperature: {temperature:.4f}\n")

            if generation_config.do_sample:
                next_token_logits = next_token_logits / temperature #generation_config.temperature
                probs = F.softmax(next_token_logits, dim=-1)
                if generation_config.top_k:
                    top_k_probs, _ = torch.topk(probs, generation_config.top_k)
                    probs[probs < top_k_probs[-1]] = 0
                if generation_config.top_p:
                    sorted_probs = torch.sort(probs, descending=True)[0]
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    probs[cumsum > generation_config.top_p] = 0
                if generation_config.min_p:
                    probs[probs < generation_config.min_p * probs.max()] = 0
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token[None, :]], dim=-1)
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Gen] New token: {next_token.item()}  Current sequence length: {input_ids.shape[1]}\n")

            if streamer:
                streamer.put(next_token.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if continuous_compute:
                model_kwargs["input_states"] = current_last_latent

            if stop_tokens is not None and (stop_tokens == next_token.item()).any():
                break

        if streamer:
            streamer.end()

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=None,
                logits=None,
                attentions=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    def new_forward(self, input_ids, text: Optional[str] = None, num_steps=32, 
                return_intermediates=False, use_cache=False, return_lam=False, 
                memory: Optional[MemorySystem] = None, input_states=None, **kwargs):
        if use_cache:
            kwargs.pop("num_steps", None)
            kwargs.pop("return_intermediates", None)
            kwargs.pop("return_lam", None)
            kwargs.pop("memory", None)
            kwargs.pop("input_states", None)
            return original_forward(input_ids, **kwargs)

        # Get initial embeddings and apply scaling.
        hidden_states = self.transformer.wte(input_ids)
        batch_size, seq_len = input_ids.shape
        hidden_dim = self.config.hidden_size
        scale_factor = 1.0 / math.sqrt(hidden_dim)
        hidden_states = hidden_states * scale_factor
        with open('debuglog.txt', 'a') as f:
            f.write(f"[Init] Hidden states norm stats: mean={torch.mean(torch.norm(hidden_states, dim=-1)).item():.4f}\n")
            f.write(f"[Init] Hidden state stats: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}, min={hidden_states.min().item():.4f}, max={hidden_states.max().item():.4f}\n")

        # Compute target norm from initial hidden_states.
        with torch.no_grad():
            target_norm = torch.mean(torch.norm(hidden_states, dim=-1)).item()
        with open('debuglog.txt', 'a') as f:
            f.write(f"[Init] Target norm set to: {target_norm:.4f}\n")

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            theta=50000.0,
            condense_ratio=1
        ).to(input_ids.device)

        # Initialize current state.
        if input_states is not None:
            if input_states.dim() == 4:
                current_state = input_states[:, -1]
            else:
                current_state = input_states
            if current_state.size(1) != hidden_states.size(1):
                last_state = current_state[:, -1:]
                pad_length = hidden_states.size(1) - 1
                padding = torch.zeros((current_state.size(0), pad_length, current_state.size(2)),
                                    device=current_state.device, dtype=current_state.dtype)
                current_state = torch.cat([padding, last_state], dim=1)
        else:
            random_state = torch.randn(batch_size, seq_len, hidden_dim, device=input_ids.device) * 0.01 # smaller initialization
            adapter_input = torch.cat([hidden_states, random_state], dim=-1)
            current_state = self.transformer.adapter(adapter_input)
            current_state = current_state * scale_factor
        with open('debuglog.txt', 'a') as f:
            f.write(f"[Init] Initial state norm: {torch.norm(current_state, dim=-1).mean().item():.4f}\n")

        # Get aggregated vector.
        lam_vector, attention_weights = self.lam(hidden_states)
        cached_states = None
        if memory:
            query_text, collection_name = memory.find_similar_query(lam_vector)
            if query_text:
                cached_states = memory.get_cached_states(collection_name)
                with open('debuglog.txt', 'a') as f:
                    f.write(f"[Memory] Found cached states for query: {query_text} in collection: {collection_name}\n")
            else:
                with open('debuglog.txt', 'a') as f:
                    f.write("[Memory] No similar query found in memory.\n")

        all_states = [] if return_intermediates else None

        # Recurrent loop with weighted residual updates and state normalization.
        for step in range(num_steps):
            with open('debuglog.txt', 'a') as f:
                f.write(f"\n[Step {step+1}] Starting recurrent update...\n")
            adapter_input = torch.cat([hidden_states, current_state], dim=-1)
            delta = self.transformer.adapter(adapter_input)
            # Log raw adapter delta stats before scaling.
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Step {step+1}] Raw adapter delta stats: mean={delta.mean().item():.4f}, std={delta.std().item():.4f}, min={delta.min().item():.4f}, max={delta.max().item():.4f}\n")
            delta = delta * scale_factor * adapter_scale # adapter scaling
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Step {step+1}] Adapter output (delta) norm: {torch.norm(delta, dim=-1).mean().item():.4f}\n")
            new_state = current_state + residual_scale * delta  # weighted residual update
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Step {step+1}] After adapter update, state norm: {torch.norm(new_state, dim=-1).mean().item():.4f}\n")
                f.write(f"[Step {step+1}] New state stats: mean={new_state.mean().item():.4f}, std={new_state.std().item():.4f}, min={new_state.min().item():.4f}, max={new_state.max().item():.4f}\n")

            if cached_states:
                cached_next_state, confidence = self.state_transfer.transfer_computation_pattern(
                    new_state, cached_states, similarity_threshold=0.8)
                if confidence > 0.8:
                    new_state = cached_next_state
                    with open('debuglog.txt', 'a') as f:
                        f.write(f"[Step {step+1}] Using cached state computation, norm: {torch.norm(new_state, dim=-1).mean().item():.4f}, Confidence: {confidence:.4f}\n")
                else:
                    with open('debuglog.txt', 'a') as f:
                        f.write(f"[Step {step+1}] Transfer computation pattern failed. Transfer computation score {confidence:.4f}\n")
                    for i, block in enumerate(self.transformer.core_block):
                        delta_block = block(new_state, freqs_cis=freqs_cis, step_idx=step * len(self.transformer.core_block) + i)
                        if isinstance(delta_block, tuple):
                            delta_block = delta_block[0]
                        delta_block = delta_block * scale_factor
                        new_state = new_state + residual_scale * delta_block
                        with open('debuglog.txt', 'a') as f:
                            f.write(f"[Step {step+1} Block {i}] After block update, state norm: {torch.norm(new_state, dim=-1).mean().item():.4f}\n")
                            f.write(f"[Step {step+1} Block {i}] New state stats: mean={new_state.mean().item():.4f}, std={new_state.std().item():.4f}\n")
            else:
                for i, block in enumerate(self.transformer.core_block):
                    delta_block = block(new_state, freqs_cis=freqs_cis, step_idx=step * len(self.transformer.core_block) + i)
                    if isinstance(delta_block, tuple):
                        delta_block = delta_block[0]
                    delta_block = delta_block * scale_factor
                    new_state = new_state + residual_scale * delta_block
                    with open('debuglog.txt', 'a') as f:
                        f.write(f"[Step {step+1} Block {i}] After block update, state norm: {torch.norm(new_state, dim=-1).mean().item():.4f}\n")
                        f.write(f"[Step {step+1} Block {i}] New state stats: mean={new_state.mean().item():.4f}, std={new_state.std().item():.4f}\n")

            # Normalize new_state to the target norm.
            norm_new = torch.norm(new_state, dim=-1, keepdim=True) + 1e-8
            new_state = new_state / norm_new * target_norm
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Step {step+1}] After normalization, state norm (should be ~{target_norm:.4f}): {torch.norm(new_state, dim=-1).mean().item():.4f}\n")
            current_state = new_state
            if return_intermediates:
                all_states.append(current_state.detach().clone())

        # Apply coda layers with scaling.
        for i, block in enumerate(self.transformer.coda):
            current_state = block(current_state, freqs_cis=freqs_cis, step_idx=num_steps * len(self.transformer.core_block) + i)
            if isinstance(current_state, tuple):
                current_state = current_state[0]
            with open('debuglog.txt', 'a') as f:
                f.write(f"[Coda Block {i}] State norm after coda: {torch.norm(current_state, dim=-1).mean().item():.4f}\n")
                f.write(f"[Coda Block {i}] State stats: mean={current_state.mean().item():.4f}, std={current_state.std().item():.4f}\n")
                
        
        current_state = current_state / (torch.norm(current_state, dim=-1, keepdim=True) + 1e-8) * target_norm

        current_state = self.transformer.ln_f(current_state)
        logits = self.lm_head(current_state)
        with open('debuglog.txt', 'a') as f:
            f.write(f"[Final] Final state norm: {torch.norm(current_state, dim=-1).mean().item():.4f}\n")
            f.write(f"[Final] Logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, min={logits.min().item():.4f}, max={logits.max().item():.4f}\n")
        if return_intermediates:
            all_states = torch.stack(all_states, dim=1)
            if memory:
                memory.cache_query(text, lam_vector, all_states)
            
            class OutputWithLatentStates(tuple):
                def __new__(cls, logits, states, lam_vector=None, attention_weights=None):
                    return super().__new__(cls, (logits, states, lam_vector, attention_weights))
                def __init__(self, logits, states, lam_vector=None, attention_weights=None):
                    self.logits = logits
                    self.latent_states = states
                    self.lam_vector = lam_vector
                    self.attention_weights = attention_weights
            
            return OutputWithLatentStates(logits, all_states, lam_vector, attention_weights)
        
        return logits

    model.forward = new_forward.__get__(model)
    model.generate_minimal = new_generate_minimal.__get__(model)
    return model

def analyze_intermediate_states(model, tokenizer, text, num_steps=16, memory: Optional[MemorySystem] = None):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    with open('debuglog.txt', 'a') as f:
        f.write(f"[Analyze] Input shape: {input_ids.shape}\n")
    with torch.no_grad():
        outputs = model(input_ids, text=text, num_steps=num_steps, 
                        return_intermediates=True, return_lam=True, memory=memory)
        logits, all_states, lam_vector, attention_weights = (outputs.logits, 
                                                             outputs.latent_states, 
                                                             outputs.lam_vector, 
                                                             outputs.attention_weights)
    with open('debuglog.txt', 'a') as f:
        f.write(f"[Analyze] Final logits shape: {logits.shape}\n")
        f.write(f"[Analyze] All states shape: {all_states.shape}\n")
        f.write(f"[Analyze] Lam vector shape: {lam_vector.shape}\n")
        f.write(f"[Analyze] Attention weights shape: {attention_weights.shape}\n")
    for token_idx in range(input_ids.shape[1]):
        token = tokenizer.decode(input_ids[0, token_idx])
        token_states = all_states[0, :, token_idx, :]
        state_changes = torch.norm(token_states[1:] - token_states[:-1], dim=1)
        with open('debuglog.txt', 'a') as f:
            f.write(f"\n[Analyze] Token {token_idx} ('{token}')\n")
        for step, change in enumerate(state_changes):
            with open('debuglog.txt', 'a') as f:
                f.write(f"  Step {step+1} -> {step+2}: {change.item():.4f}\n")
    return logits, all_states, lam_vector, attention_weights

if __name__ == "__main__":
    # Clear the debug log
    with open('debuglog.txt', 'w') as f:
        f.write("=== Starting New Generation Session ===\n")

    model = AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    # Use a residual_scale (e.g., 0.3) and normalization to help keep the state norms stable.
    adapter_scale = 10.0
    model = modify_model_for_intermediates(model, residual_scale=0.3, adapter_scale=adapter_scale)
    text = "The capital of Westphalia is"
    memory = MemorySystem(chromadb.PersistentClient("./huginn_db"))
    logits, states = analyze_intermediate_states(model, tokenizer, text, num_steps=16, memory=memory)
    
    #Generate
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        max_length=50  # Adjust as needed
    )
    generated_ids = model.generate_minimal(input_ids, generation_config=generation_config, tokenizer=tokenizer)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")