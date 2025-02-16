import torch
import hashlib
from datetime import datetime

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    """Precompute the frequency cache for RoPE."""
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)
    
def _generate_collection_name(text: str) -> str:
    """Generate a unique collection name for a query."""
    # Create a hash of the text to use as part of collection name
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"comp_{text_hash}_{timestamp}"