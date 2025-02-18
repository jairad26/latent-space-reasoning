import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
import json
from dataclasses import dataclass
import uuid

@dataclass
class LatentStateMetadata:
    """Metadata for stored latent states"""
    token_ids: Optional[List[int]] = None
    text: Optional[str] = None
    convergence_rate: Optional[float] = None
    num_iterations: Optional[int] = None
    pattern_type: Optional[str] = None  # 'orbit', 'slider', 'convergent'

class LatentStateStore:
    def __init__(self, collection_name: str = "latent_states", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _process_trajectory(self, 
                            trajectory: np.ndarray,
                            sliding_window: int = 3) -> Dict[str, np.ndarray]:
        """
        More sensitive pattern detection in latent trajectories.
        """
        trajectory_2d = trajectory.mean(axis=(1, 2))
        diffs = np.diff(trajectory_2d, axis=0)
        
        # Orbital pattern detection
        correlation_matrix = np.corrcoef(diffs)
        if correlation_matrix.shape[0] > sliding_window * 2:
            # Look for repeating patterns in shorter segments
            periodic_scores = []
            for i in range(0, len(diffs) - sliding_window * 2, sliding_window):
                segment = correlation_matrix[i:i+sliding_window, i+sliding_window:i+sliding_window*2]
                periodic_scores.append(np.max(np.abs(segment)))
            periodic_score = np.mean(periodic_scores) if periodic_scores else 0
        else:
            periodic_score = 0
        
        # Slider pattern detection
        direction_consistency = np.mean(np.abs(diffs.sum(axis=0))) / (np.mean(np.abs(diffs)) + 1e-6)
        
        # Convergence detection
        convergence_rate = np.mean(np.linalg.norm(diffs, axis=1))
        final_movement = np.linalg.norm(diffs[-1]) if len(diffs) > 0 else 0
        
        # Pattern classification with more nuanced thresholds
        if periodic_score > 0.6:  # Lowered threshold for orbital patterns
            pattern_type = 'orbit'
        elif direction_consistency > 0.5:  # Lowered threshold for sliders
            if convergence_rate < 0.1 and final_movement < 0.05:
                pattern_type = 'convergent'
            else:
                pattern_type = 'slider'
        else:
            if convergence_rate < 0.1:
                pattern_type = 'convergent'
            else:
                pattern_type = 'complex'  # New category for mixed patterns
                
        return {
            'pattern_type': pattern_type,
            'periodic_score': float(periodic_score),
            'direction_consistency': float(direction_consistency),
            'convergence_rate': float(convergence_rate),
            'final_movement': float(final_movement)
        }

    def store_latent_states(self,
                           latent_data: Dict[str, np.ndarray],
                           metadata: LatentStateMetadata) -> str:
        doc_id = str(uuid.uuid4())
        pattern_info = self._process_trajectory(latent_data['trajectory'])
        final_state = latent_data['final_state'].mean(axis=(0, 1))
        
        # Convert numpy arrays to lists and handle serialization
        final_state_list = final_state.tolist()
        trajectory_list = latent_data['trajectory'].tolist()
        
        # Prepare metadata with proper serialization
        meta_dict = {
            "token_ids": json.dumps(metadata.token_ids if metadata.token_ids else []),
            "text": metadata.text if metadata.text else "",
            "convergence_rate": float(metadata.convergence_rate) if metadata.convergence_rate is not None else 0.0,
            "num_iterations": metadata.num_iterations if metadata.num_iterations is not None else 0,
            "pattern_type": pattern_info['pattern_type'],
            "periodic_score": float(pattern_info['periodic_score']),
            "direction_consistency": float(pattern_info['direction_consistency']),
            "trajectory": json.dumps(trajectory_list),  # Store the full trajectory as serialized JSON
            "mean_state": json.dumps(latent_data['mean_state'].tolist())
        }
        
        try:
            self.collection.add(
                embeddings=[final_state_list],
                metadatas=[meta_dict],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            print(f"Storage error details: {meta_dict}")
            raise e

    def find_similar_states(self,
                          query_state: np.ndarray,
                          pattern_type: Optional[str] = None,
                          n_results: int = 5) -> List[Dict]:
        if query_state.ndim > 1:
            query_state = query_state.mean(axis=tuple(range(query_state.ndim-1)))
            
        where_clause = {}
        if pattern_type:
            where_clause["pattern_type"] = pattern_type
            
        results = self.collection.query(
            query_embeddings=[query_state.flatten().tolist()],
            where=where_clause if where_clause else None,
            n_results=n_results
        )
        
        similar_states = []
        if len(results['ids']) > 0:  # Check if we got any results
            for idx, (id_, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Deserialize stored data
                metadata['distance'] = distance
                metadata['token_ids'] = json.loads(metadata['token_ids'])
                metadata['trajectory'] = json.loads(metadata['trajectory'])
                metadata['mean_state'] = json.loads(metadata['mean_state'])
                similar_states.append(metadata)
            
        return similar_states

    def get_pattern_statistics(self) -> Dict[str, int]:
        results = self.collection.get()
        if not results['metadatas']:
            return {}
            
        pattern_counts = {}
        for metadata in results['metadatas']:
            pattern_type = metadata['pattern_type']
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
        return pattern_counts
    
def example_usage():
    """Example of how to use the LatentStateStore"""
    # Initialize store
    store = LatentStateStore()
    
    # Example latent data (you would get this from your model)
    latent_data = {
        'trajectory': np.random.randn(10, 128),  # 10 iterations, 128-dim hidden state
        'final_state': np.random.randn(128),
        'mean_state': np.random.randn(128)
    }
    
    # Example metadata
    metadata = LatentStateMetadata(
        token_ids=[1, 2, 3, 4],
        text="example text",
        convergence_rate=0.95,
        num_iterations=10
    )
    
    # Store the states
    doc_id = store.store_latent_states(latent_data, metadata)
    
    # Find similar states
    similar_states = store.find_similar_states(
        query_state=latent_data['final_state'],
        pattern_type='orbit'
    )
    
    # Get pattern statistics
    pattern_stats = store.get_pattern_statistics()
    print(f"Pattern statistics: {pattern_stats}")

if __name__ == "__main__":
    example_usage()