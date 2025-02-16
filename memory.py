import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import hashlib
import chromadb 
from utils import _generate_collection_name

class MemorySystem:
    def __init__(self, client: chromadb.PersistentClient, similarity_threshold=0.85):
        """
        Initialize the caching system with:
        - lam_collection: stores LAM vectors and maps to computation collection names
        - dynamic computation collections: created per query
        
        Args:
            client: ChromaDB client
            similarity_threshold: threshold for considering LAM vectors similar
        """
        self.client = client
        self.similarity_threshold = similarity_threshold
        
        # Main collection for LAM vectors
        self.lam_collection = client.get_or_create_collection(
            name="lam_vectors",
            metadata={"description": "Store LAM vectors and computation collection mappings"}
        )
        
    def find_similar_query(self, lam_vector: torch.Tensor) -> Tuple[Optional[str], Optional[str]]:
        """
        Search for a similar query using the LAM vector.
        
        Returns:
            Tuple of (query_text if found else None, collection_name if found else None)
        """
        lam_vector_list = lam_vector.cpu().numpy().reshape(-1).tolist()
        
        results = self.lam_collection.query(
            query_embeddings=[lam_vector_list],
            n_results=1
        )
        
        
        if results and results['distances'][0]:
            similarity = 1 - results['distances'][0][0]
            if similarity >= self.similarity_threshold:
                query_text = results['documents'][0][0]
                collection_name = results['metadatas'][0][0]['collection_name']
                return query_text, collection_name
        
        return None, None
    
    def cache_query(self, 
                   text: str, 
                   lam_vector: torch.Tensor, 
                   all_states: torch.Tensor) -> str:
        """
        Cache a new query by:
        1. Creating a new collection for its states
        2. Storing the LAM vector with a reference to the collection
        
        Returns:
            collection_name: Name of the created collection
        """
        collection_name = _generate_collection_name(text)
        
        # Create new collection for this query's states
        states_collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"original_query": text}
        )
        
        # Store states in the new collection
        # We'll store each step's states separately to make retrieval more flexible
        batch_size, num_steps, seq_len, hidden_size = all_states.shape
        
        for step in range(num_steps):
            step_states = all_states[0, step].cpu().numpy()  # [seq_len, hidden_size]
            
            # Store each token's state vector for this step
            for token_idx in range(seq_len):
                states_collection.add(
                    embeddings=[step_states[token_idx].tolist()],
                    documents=[f"Step {step}, Token {token_idx}"],
                    ids=[f"step_{step}_token_{token_idx}"],
                    metadatas=[{
                        "step": step,
                        "token_idx": token_idx,
                        "state_norm": float(np.linalg.norm(step_states[token_idx]))
                    }]
                )
        
        # Store LAM vector with reference to the new collection
        self.lam_collection.add(
            embeddings=[lam_vector.cpu().numpy().reshape(-1).tolist()],
            documents=[text],
            ids=[collection_name],
            metadatas=[{
                "collection_name": collection_name,
                "num_steps": num_steps,
                "seq_length": seq_len,
                "hidden_size": hidden_size
            }]
        )
        
        return collection_name
    
    def get_cached_states(self, collection_name: str, step: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve states from a cached computation collection.
        
        Args:
            collection_name: Name of the collection to retrieve from
            step: Optional specific step to retrieve (None for all steps)
            
        Returns:
            Dictionary containing the retrieved states and metadata
        """
        states_collection = self.client.get_collection(collection_name)
        
        # Get collection metadata
        metadata = states_collection.metadata
        
        # Query based on step if specified
        if step is not None:
            results = states_collection.get(
                where={"step": step}
            )
        else:
            results = states_collection.get()
        
        return {
            "metadata": metadata,
            "states": results
        }
    