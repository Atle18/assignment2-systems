import torch
import math

class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, seq_len, d_model = Q.shape
        Bc, Br = 16, 16
        Tr = seq_len // Br
        Tc = seq_len // Bc
        device = Q.device
        m = torch.full([batch_size, seq_len], float("-inf"), device=device)
        l = torch.zeros([batch_size, seq_len], device=device)
        L = torch.zeros([batch_size, seq_len], device=device)
        O = torch.zeros(Q.shape, device=device)
        for i in range(Tr):
            q_i = Q[:, i*Br:(i+1)*Br, :]
            
            for j in range(Tc):
                k_j = K[:, j*Bc:(j+1)*Bc, :]
                S_ij = q_i @ k_j.transpose(-1, -2) / math.sqrt(d_model)
                m_new = torch.max(m[:, i*Br:(i+1)*Br], torch.max(S_ij, dim=-1)[0])
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
                l[:, i*Br:(i+1)*Br] = torch.exp(m[:, i*Br:(i+1)*Br] - m_new) * l[:, i*Br:(i+1)*Br] + torch.sum(P_ij, dim=-1)
                O[:, i*Br:(i+1)*Br, :] = torch.exp(m[:, i*Br:(i+1)*Br] - m_new).unsqueeze(-1) * O[:, i*Br:(i+1)*Br, :] + P_ij @ V[:, j*Bc:(j+1)*Bc, :]
                m[:, i*Br:(i+1)*Br] = m_new
            
            O[:, i*Br:(i+1)*Br, :] = O[:, i*Br:(i+1)*Br, :] / l[:, i*Br:(i+1)*Br].unsqueeze(-1)
            L[:, i*Br:(i+1)*Br] = m[:, i*Br:(i+1)*Br] + torch.log(l[:, i*Br:(i+1)*Br])
    
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.br = Br
        ctx.bc = Bc
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError


def flash_attention_2_pytorch(Q, K, V):
    return FlashAttention2Pytorch.apply(Q, K, V)

Q = torch.randn([4, 64, 64], device="cuda")
K = torch.randn([4, 64, 64], device="cuda")
V = torch.randn([4, 64, 64], device="cuda")
O = flash_attention_2_pytorch(Q, K, V)