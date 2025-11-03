import torch
import math
import triton
import triton.language as tl

@triton.jit
def fa2_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(q, tl.trans(k_j)) * scale
        if IS_CAUSAL:
            q_offset = query_tile_index * Q_TILE_SIZE
            k_offset = j * K_TILE_SIZE
            causal_mask = tl.arange(0, Q_TILE_SIZE)[:, None] + q_offset >= tl.arange(0, K_TILE_SIZE)[None, :] + k_offset
            S = tl.where(causal_mask, S, -1e6)
        m_new = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_new[:, None])
        l = tl.exp(m - m_new) * l + tl.sum(P, axis=1)
        P_cast = P.to(v_j.dtype)
        acc = tl.exp(m - m_new)[:, None] * O
        O = tl.dot(P_cast, v_j, acc=acc)
        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O = O / l[:, None]
    L = m + tl.log(l)

    tl.store(O_block_ptr, O, boundary_check=(0,1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, seq_len, d_model = Q.shape
        Bc, Br = 16, 16
        Tr = (seq_len + Br - 1) // Br
        device = Q.device
        L = torch.zeros([batch_size, seq_len], device=device)
        O = torch.zeros(Q.shape, device=device)

        scale = 1.0 / math.sqrt(d_model)
        
        fa2_fwd_kernel[(Tr, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len, seq_len,
            scale,
            D=d_model,
            Q_TILE_SIZE=Br,
            K_TILE_SIZE=Bc,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.br = Br
        ctx.bc = Bc
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError


def flash_attention_2_triton(Q, K, V):
    return FlashAttention2Triton.apply(Q, K, V)

Q = torch.randn([4, 64, 64], device="cuda")
K = torch.randn([4, 64, 64], device="cuda")
V = torch.randn([4, 64, 64], device="cuda")
O = flash_attention_2_triton(Q, K, V)