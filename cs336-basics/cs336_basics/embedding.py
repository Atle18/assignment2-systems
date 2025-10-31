from typing import Optional
import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embedding = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        
        return self.embedding[token_ids]