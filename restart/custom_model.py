import torch
from typing import Optional, List, Dict, Any
from transformers.generation.utils import Cache
import numpy as np
from lss import LatentStateStore, LatentStateMetadata

def modify_model_for_intermediates(model):
    original_forward = model.forward
    model.store = LatentStateStore()
    
    def new_forward(self,
                   input_ids: Optional[torch.Tensor] = None,
                   input_embeds: Optional[torch.Tensor] = None,
                   input_states: Optional[torch.Tensor] = None,
                   attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.Tensor] = None,
                   labels: Optional[torch.Tensor] = None,
                   num_steps: Optional[int] = None,
                   past_key_values: Optional[Cache] = None,
                   tokenizer = None,
                   output_details: Dict[str, bool] = {
                       "return_logits": True,
                       "return_latents": True,
                       "return_attention": False,
                       "return_head": False,
                       "return_stats": False,
                   },
                   use_cache: bool = False,
                   cache_position: Optional[torch.Tensor] = None,
                   **kwargs) -> Any:
        
        original_hooks = []
        latent_states = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                state = output[0]
            else:
                state = output
            latent_states.append(state.detach().cpu())
            # print(f"Hook captured state with shape: {state.shape}")

        # Hook into the core_block
        if hasattr(self.transformer, 'core_block'):
            # print("Found core_block")
            for idx, block in enumerate(self.transformer.core_block):
                handle = block.register_forward_hook(hook_fn)
                original_hooks.append(handle)
                # print(f"Registered hook for core_block.{idx}")
        else:
            print("Warning: Could not find core_block")
            
        # Prepare forward arguments
        forward_kwargs = {
            'input_ids': input_ids,
            'input_embeds': input_embeds,
            'input_states': input_states,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels,
            'num_steps': num_steps,
            'past_key_values': past_key_values,
            'output_details': output_details,
            'use_cache': use_cache,
            'cache_position': cache_position,
            **kwargs
        }
        forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}
        
        # Run forward pass
        output = original_forward(**forward_kwargs)
        
        # Remove hooks
        for hook in original_hooks:
            hook.remove()
            
        # Process captured states
        if latent_states:
            # Reshape states to match expectations (iterations x batch x seq x hidden)
            # The model has 4 core blocks that repeat num_steps times
            num_blocks = 4  # From the model structure
            if num_steps is None:
                num_steps = len(latent_states) // num_blocks
            
            reshaped_states = []
            for step in range(num_steps):
                step_states = latent_states[step * num_blocks:(step + 1) * num_blocks]
                # Take the last block's output for each iteration
                reshaped_states.append(step_states[-1])
            
            all_states = torch.stack(reshaped_states, dim=0)
            
            # Calculate statistics
            mean_state = all_states.mean(dim=2)  # Average over sequence length
            final_state = all_states[-1]  # Last iteration
            
            # Calculate trajectory statistics regardless of output_details
            state_diff = torch.norm(all_states[1:] - all_states[:-1], dim=-1)
            convergence_rate = state_diff.mean(dim=-1)
            
            # Add to output
            output.latent_trajectory = all_states
            output.mean_latent_state = mean_state
            output.final_latent_state = final_state
            
            if output_details.get("return_stats", False):
                output.convergence_rate = convergence_rate
                
            # Create metadata
            metadata = LatentStateMetadata(
                token_ids=input_ids[0].tolist() if input_ids is not None else None,
                text=tokenizer.decode(input_ids[0]) if tokenizer and input_ids is not None else None,
                convergence_rate=float(convergence_rate.mean().item()),
                num_iterations=num_steps
            )
            
            # Process for storage
            latent_data = {
                'trajectory': all_states.numpy(),
                'final_state': final_state.numpy(),
                'mean_state': mean_state.numpy(),
                'convergence_rate': float(convergence_rate.mean().item())
            }
            
            # Store in database
            try:
                model.store.store_latent_states(latent_data, metadata)
            except Exception as e:
                print(f"Failed to store latent states: {e}")
        
        return output
    
    # Replace forward method
    model.forward = new_forward.__get__(model, model.__class__)
    return model

