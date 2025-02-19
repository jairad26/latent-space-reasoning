# import torch
# import torch.nn.functional as F
# from typing import Optional, Dict, List, Tuple, Union, Any
# from memory import MemorySystem
# from transformers import GenerationConfig
# from transformers.generation.utils import GenerateDecoderOnlyOutput

# from huginn_modified import HuginnDynamicCache

# @torch.no_grad()
# def generate_minimal(
#     self,
#     input_ids: torch.LongTensor,
#     generation_config: Optional[GenerationConfig] = None,
#     tokenizer=None,
#     streamer=None,
#     continuous_compute: bool = False,
#     cache_kwargs: dict = {},
#     **model_kwargs,
# ) -> Union[torch.Tensor, dict[str, Any]]:
#     """Minimal single-sequence generation using our forward implementation."""
#     # Setup
#     if generation_config is None:
#         generation_config = self.generation_config
#     model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
#     model_kwargs["use_cache"] = True
#     model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
#     stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)

#     if continuous_compute:
#         embedded_inputs, _, _ = self.embed_inputs(input_ids)
#         model_kwargs["input_states"] = self.initialize_state(embedded_inputs)

#     # Generate tokens
#     for _ in range(generation_config.max_length - input_ids.shape[1]):
#         # Forward pass
#         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#         outputs = self(**model_inputs)
#         next_token_logits = outputs.logits[0, -1, :]
        
#         if continuous_compute:
#             current_last_latent = outputs.latent_states[:, -1:, :]

#         # Sample or select next token
#         if generation_config.do_sample:
#             if generation_config.temperature:
#                 next_token_logits = next_token_logits / generation_config.temperature

#             probs = F.softmax(next_token_logits, dim=-1)

#             # Apply top_k
#             if generation_config.top_k:
#                 top_k_probs, _ = torch.topk(probs, generation_config.top_k)
#                 probs[probs < top_k_probs[-1]] = 0
#             # Apply top_p
#             if generation_config.top_p:
#                 sorted_probs = torch.sort(probs, descending=True)[0]
#                 cumsum = torch.cumsum(sorted_probs, dim=-1)
#                 probs[cumsum > generation_config.top_p] = 0
#             # Apply min_p
#             if generation_config.min_p:
#                 probs[probs < generation_config.min_p * probs.max()] = 0

#             probs = probs / probs.sum()
#             next_token = torch.multinomial(probs, num_samples=1)
#         else:
#             next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

#         input_ids = torch.cat([input_ids, next_token[None, :]], dim=-1)

#         if streamer:
#             streamer.put(next_token.cpu())

#         # Update model kwargs
#         model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
#         if continuous_compute:
#             model_kwargs["input_states"] = current_last_latent

#         # Check if we hit a stop token
#         if stop_tokens is not None and next_token in stop_tokens:
#             break

#     if streamer:
#         streamer.end()

#     if generation_config.return_dict_in_generate:
#         return GenerateDecoderOnlyOutput(
#             sequences=input_ids,
#             scores=None,
#             logits=None,
#             attentions=None,
#             hidden_states=None,
#             past_key_values=model_kwargs.get("past_key_values"),
#         )
#     return input_ids