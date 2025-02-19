import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Tuple, List

class StateTransferModule:
    def __init__(self, model_hidden_size: int):
        self.hidden_size = model_hidden_size
        
    def analyze_state_transitions(self, cached_states: Dict[str, any]) -> List[torch.Tensor]:
        """
        Analyze the state transitions from cached computations to extract transformation matrices.
        
        Args:
            cached_states: Dictionary containing states and metadata from the cache
            
        Returns:
            List of transformation matrices for each step
        """
        states_data = cached_states['states']
        
        # Group states by step
        max_steps = max(meta['step'] for meta in states_data['metadatas'])
        step_states = [[] for _ in range(max_steps + 1)]
        
        for i, metadata in enumerate(states_data['metadatas']):
            step = metadata['step']
            embedding = torch.tensor(states_data['embeddings'][i])
            step_states[step].append(embedding)
            
        # Convert to tensors and calculate transition matrices
        transition_matrices = []
        for i in range(len(step_states) - 1):
            current = torch.stack(step_states[i])
            next_state = torch.stack(step_states[i + 1])
            
            # Use pseudo-inverse to find the transformation matrix
            # that best maps current to next_state
            transition = torch.pinverse(current) @ next_state
            transition_matrices.append(transition)
            
        return transition_matrices
    
    def apply_state_transformations(self, 
                                  current_states: torch.Tensor,
                                  cached_transformations: List[torch.Tensor],
                                  adaptation_rate: float = 0.5) -> torch.Tensor:
        """
        Apply cached transformation patterns to current states with adaptation.
        
        Args:
            current_states: Current intermediate states (batch_size, seq_len, hidden_size)
            cached_transformations: List of transformation matrices from cache
            adaptation_rate: How much to blend between cached and original computations
            
        Returns:
            Modified states incorporating cached computational patterns
        """
        batch_size, seq_len, _ = current_states.shape
        modified_states = current_states.clone()
        
        for transform_matrix in cached_transformations:
            # Apply cached transformation
            cached_computation = modified_states @ transform_matrix
            
            # Blend with original computation trajectory
            modified_states = (adaptation_rate * cached_computation + 
                             (1 - adaptation_rate) * modified_states)
            
            # Optional: Add noise for robustness
            noise = torch.randn_like(modified_states) * 0.01
            modified_states = modified_states + noise
            
            # Normalize to prevent exponential growth/decay
            modified_states = modified_states / torch.norm(modified_states, dim=-1, keepdim=True)
            
        return modified_states
    
    def transfer_computation_pattern(self,
                                   current_states: torch.Tensor,
                                   cached_states: Dict[str, any],
                                   similarity_threshold: float = 0.8) -> Tuple[torch.Tensor, float]:
        """
        Transfer computation patterns from cached states to current input.
        
        Args:
            current_states: Current intermediate states
            cached_states: Dictionary containing cached states and metadata
            similarity_threshold: Minimum similarity required for transfer
            
        Returns:
            Tuple of (modified_states, confidence_score)
        """
        
        # Extract transformation patterns
        cached_transforms = self.analyze_state_transitions(cached_states)
        
        # Reshape current states to match cache format
        # current_states is [batch, seq_len, hidden_size]
        current_flat = current_states.squeeze(0)  # Remove batch dim -> [seq_len, hidden_size]
        
        # Get cached embeddings for the current step
        cached_embeddings = torch.tensor(np.array(cached_states['states']['embeddings']))
        # Reshape cached to [num_steps, seq_len, hidden_size]
        seq_len = current_states.size(1)
        num_steps = len(cached_states['states']['metadatas']) // seq_len
        cached_reshaped = cached_embeddings.view(num_steps, seq_len, -1)
        
        # Calculate patterns for comparison
        current_pattern = self.extract_computation_pattern(current_flat)
        # Use first step of cached states for comparison
        cached_first_step = cached_reshaped[0]  # [seq_len, hidden_size]
        cached_pattern = self.extract_computation_pattern(cached_first_step)
        
        # Calculate similarity
        similarity = F.cosine_similarity(current_pattern, cached_pattern, dim=0).mean()
        
        # If similarity is too low, return original states
        if similarity < similarity_threshold:
            return current_states, similarity.item()
            
        # Apply transformations with adaptive rate based on similarity
        adaptation_rate = similarity.item()
        modified_states = self.apply_state_transformations(
            current_states, 
            cached_transforms,
            adaptation_rate
        )
        
        return modified_states, similarity.item()
    
    def extract_computation_pattern(self, states: torch.Tensor) -> torch.Tensor:
        """
        Extract a representation of the computation pattern from states.
        
        Args:
            states: State tensors to analyze
            
        Returns:
            Pattern representation tensor
        """
        # If states has 3 dimensions, remove batch dimension
        if states.dim() == 3:
            states = states.squeeze(0)
            
        # Cast to float32 before moving to MPS
        U, S, V = torch.linalg.svd(states)
        U = U.to(dtype=torch.float32).to("mps")
        S = S.to(dtype=torch.float32).to("mps")
        V = V.to(dtype=torch.float32).to("mps")
        
        # Determine number of components based on explained variance
        explained_variance = (S ** 2) / (S ** 2).sum()
        cumulative_variance = torch.cumsum(explained_variance, dim=0)
        # Keep components that explain 95% of variance
        n_components = torch.where(cumulative_variance > 0.95)[0][0].item() + 1
        
        pattern = V[:, :n_components]
        return pattern