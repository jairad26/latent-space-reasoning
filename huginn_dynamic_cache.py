import torch
from transformers.generation.utils import DynamicCache
from typing import Optional

class HuginnDynamicCache(DynamicCache):
    def __init__(self, lookup_strategy: str = "full") -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.value_cache: dict[int, dict[int, torch.Tensor]] = {}
        # structure: cache[index_of_layer_or_recurrent_step][index_in_sequence]
        # the cache is held uncoalesced because certain recurrent steps may be missing for some sequence ids if using
        # per-token adaptive compute. In those cases, the "lookup_strategy" determines how to proceed
        # Also, It is critical that the head indices do not overlap with the recurrent iteration indices
        self.lookup_strategy = lookup_strategy

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx: int,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        if "compress-" in self.lookup_strategy and step_idx > 1:  # hardcode for current model!
            compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
            if "compress-s" in self.lookup_strategy:
                new_step_idx = (step_idx - 2) % compression_stage + 2
            else:
                new_step_idx = (step_idx - 2) // compression_stage + 2
            # @ print(step_idx, new_step_idx, compression_stage)
            step_idx = new_step_idx
        # Init
        if step_idx not in self.key_cache:
            self.key_cache[step_idx] = {}
            self.value_cache[step_idx] = {}
        # Update the number of seen tokens, we assume that step_idx=0 (first prelude) is always hit
        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # Add entries to cache
        for idx, entry in enumerate(key_states.unbind(dim=-2)):
            if "compress-" not in self.lookup_strategy:
                assert step_idx < 0 or self._seen_tokens - key_states.shape[-2] + idx not in self.key_cache[step_idx]
            # print(f"Overwrote cache entry for step_idx {step_idx}") # likely the head
            self.key_cache[step_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
        for idx, entry in enumerate(value_states.unbind(dim=-2)):
            self.value_cache[step_idx][self._seen_tokens - value_states.shape[-2] + idx] = entry

        # Materialize past state based on lookup strategy:
        if len(self.key_cache[step_idx]) == self._seen_tokens or self.lookup_strategy == "full":
            # All entries are present, materialize cache as normal
            return (
                torch.stack(list(self.key_cache[step_idx].values()), dim=-2),
                torch.stack(list(self.value_cache[step_idx].values()), dim=-2),
            )
        else:  # some entries where not previously computed
            # if lookup_strategy.startswith("latest"):
            #     latest_keys = []
            #     latest_values = []
            #     for token_pos in range(self._seen_tokens):
            #         # Find the latest step that has this token position
            #         max_step = max((s for s in range(step_idx + 1) if token_pos in self.key_cache[s]), default=None)
            #         if max_step is None:
            #             raise ValueError(f"No cache entry found for token position {token_pos}")
            #         latest_keys.append(self.key_cache[max_step][token_pos])
            #         latest_values.append(self.value_cache[max_step][token_pos])
            #     return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            if lookup_strategy.startswith("latest-m4"):
                latest_keys = []
                latest_values = []
                for token_pos in range(self._seen_tokens):
                    # For steps >= 2, use modulo 4
                    if step_idx >= 2:
                        # Find valid steps for this token position
                        valid_steps = [s for s in range(step_idx + 1) if token_pos in self.key_cache[s]]
                        max_step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4])
                    else:
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    if max_step is None:
                        raise ValueError(f"No cache entry found for token position {token_pos}")
                    latest_keys.append(self.key_cache[max_step][token_pos])
                    latest_values.append(self.value_cache[max_step][token_pos])
                return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            elif lookup_strategy.startswith("skip"):
                existing_keys = []
                existing_values = []
                for token_pos in range(self._seen_tokens):
                    if token_pos in self.key_cache[step_idx]:
                        existing_keys.append(self.key_cache[step_idx][token_pos])
                        existing_values.append(self.value_cache[step_idx][token_pos])
                return torch.stack(existing_keys, dim=-2), torch.stack(existing_values, dim=-2)
            elif lookup_strategy.startswith("randomized"):  # sanity check
                rand_keys = []
                rand_values = []
                for token_pos in range(self._seen_tokens):
                    if step_idx < 2:  # For prelude steps
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    else:  # Get all steps from same block position
                        curr_modulo = (step_idx - 2) % 4 + 2
                        valid_steps = [
                            s
                            for s in range(2, step_idx + 1)
                            if (s - 2) % 4 + 2 == curr_modulo and token_pos in self.key_cache[s]
                        ]
                        max_step = valid_steps[torch.randint(len(valid_steps), (1,))]
                    rand_keys.append(self.key_cache[max_step][token_pos])
                    rand_values.append(self.value_cache[max_step][token_pos])
                return torch.stack(rand_keys, dim=-2), torch.stack(rand_values, dim=-2)
            else:
                raise ValueError(f"Unknown lookup strategy: {lookup_strategy}")

    def reset(self) -> None:
        """Reset the cache state."""
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()

    def get_seq_length(self, step_idx: int = 0) -> int:
        return self._seen_tokens

    def get_memory_usage(self) -> float:
        total_bytes = 0
        # For each recurrent step/layer index
        for step_idx in self.key_cache:
            # Get the sequence cache for this step
            key_seq_cache = self.key_cache[step_idx]
            for seq_idx in key_seq_cache:
                key_tensor = key_seq_cache[seq_idx]
                # Add memory for of key tensors, assuming value is the same
                total_bytes += key_tensor.nelement() * key_tensor.element_size()
        return total_bytes * 2 / (1024 * 1024)