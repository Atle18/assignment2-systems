from typing import Optional
import torch
import math
from .softmax import softmax
from .rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    # q: [batch_size, ..., seq_len, d_k], k: [batch_size, ..., seq_len, d_k], v: [batch_size, ..., seq_len, d_v]
    d_k = q.shape[-1]
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn = softmax(scores, dim=-1)
    out = attn @ v
    return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads
        
        self.Wq = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = torch.nn.Parameter(torch.empty(d_model, d_model))
        
        std = math.sqrt(2.0 / (d_model + d_model))
        torch.nn.init.trunc_normal_(self.Wq, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.Wk, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.Wv, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.Wo, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]=None):
        # x: [... seq_len, d_model]
        seq_len = x.shape[-2]
        Q = x @ self.Wq.transpose(-1, -2)
        K = x @ self.Wk.transpose(-1, -2)
        V = x @ self.Wv.transpose(-1, -2)
        
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        K = K.view(*K.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        V = V.view(*V.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        
        if self.theta is not None and self.max_seq_len is not None and token_positions is not None:
            rope = RotaryPositionalEmbedding(theta=self.theta, d_k=self.head_dim, max_seq_len=self.max_seq_len)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = ~causal_mask
        attn = scaled_dot_product_attention(Q, K, V, causal_mask)
        
        attn = attn.transpose(-2, -3).contiguous()
        attn = attn.view(*attn.shape[:-2], self.d_model)
        out = attn @ self.Wo.transpose(-1, -2)
        return out
    