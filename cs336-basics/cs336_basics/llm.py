import torch
from .rmsnorm import RMSNorm
from .transformer import Transformer
from .embedding import Embedding
from .linear import Linear


class LLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta=None):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformers = torch.nn.ModuleList([
            Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model=d_model)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        
        for transformer in self.transformers:
            x = transformer(x)
        
        x = self.rmsnorm(x)
        logits = self.linear(x)
        return logits
    