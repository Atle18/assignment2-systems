from typing import Optional
import torch
from .attention import MultiHeadAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.rmsnorm1 = RMSNorm(d_model=self.d_model)
        self.rmsnorm2 = RMSNorm(d_model=self.d_model)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, theta=self.theta, max_seq_len=self.max_seq_len)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)
        
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]=None):        
        if token_positions is None:
            seq_length = x.shape[-2]
            token_positions = torch.arange(seq_length, device=x.device).expand(*x.shape[:-2], seq_length)
        
        input = x
        x = self.rmsnorm1(x)
        x = self.mha(x, token_positions)
        x = input + x
        
        ffn_in = x
        x = self.rmsnorm2(x)
        x = self.ffn(x)
        ffn_out = ffn_in + x        
        return ffn_out