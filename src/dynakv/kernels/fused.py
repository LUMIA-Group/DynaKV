import torch
import triton
import triton.language as tl
import math

def get_configs():
    configs = []
    block_l_list = [16]
    block_r_list = [64]
    num_warps_list = [4, 8]
    num_stages_list = [2, 3, 4]

    for block_l in block_l_list:
        for block_r in block_r_list:
            for num_warps in num_warps_list:
                for num_stages in num_stages_list:
                    if block_l * block_r * 4 > 64 * 1024: 
                        continue
                    configs.append(
                        triton.Config(
                            {'BLOCK_SIZE_L': block_l, 'BLOCK_SIZE_R': block_r},
                            num_stages=num_stages, 
                            num_warps=num_warps
                        )
                    )
    return configs

@triton.jit
def _fused_flash_decode_sparse_kernel(
    Q_ptr, K_flat_ptr, K_up_ptr, V_flat_ptr, V_up_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kflat_b, stride_vflat_b,
    stride_k_up_h, stride_k_up_r, stride_k_up_d,
    stride_v_up_h, stride_v_up_r, stride_v_up_d,
    stride_out_b, stride_out_h, stride_out_d,
    L_k_ptr, stride_lk_b, stride_lk_s,
    O_k_ptr, stride_ok_b, stride_ok_s,
    L_v_ptr, stride_lv_b, stride_lv_s,
    O_v_ptr, stride_ov_b, stride_ov_s,
    seq_len, RANK_TOTAL, log_theta, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_S: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)


    offs_d_0 = tl.arange(0, HEAD_DIM // 2)
    offs_d_1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)
    offs_d = tl.arange(0, HEAD_DIM)

    q_0 = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d_0 * stride_qd)
    q_1 = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d_1 * stride_qd)
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    sm_scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    k_up_base = K_up_ptr + pid_h * stride_k_up_h
    v_up_base = V_up_ptr + pid_h * stride_v_up_h



    freqs = (tl.arange(0, HEAD_DIM // 2) * 2).to(tl.float32) / tl.cast(HEAD_DIM, tl.float32)
    freqs = tl.exp(-freqs * log_theta)


    for s0 in range(0, seq_len, BLOCK_SIZE_S):
        offs_s = s0 + tl.arange(0, BLOCK_SIZE_S)
        mask_s = offs_s < seq_len

        cur_k_lens = tl.load(L_k_ptr + pid_b * stride_lk_b + offs_s * stride_lk_s, mask=mask_s, other=0)
        cur_k_offs = tl.load(O_k_ptr + pid_b * stride_ok_b + offs_s * stride_ok_s, mask=mask_s, other=0)
        

        k_block_0 = tl.zeros((BLOCK_SIZE_S, HEAD_DIM // 2), dtype=tl.float32)
        k_block_1 = tl.zeros((BLOCK_SIZE_S, HEAD_DIM // 2), dtype=tl.float32)
        
        for r0 in range(0, RANK_TOTAL, BLOCK_SIZE_R):
            offs_r = r0 + tl.arange(0, BLOCK_SIZE_R)
            mask_r = offs_r < RANK_TOTAL
            
            mask_xk = mask_s[:, None] & (offs_r[None, :] < cur_k_lens[:, None])

            xk_ptrs = K_flat_ptr + pid_b * stride_kflat_b + cur_k_offs[:, None] + offs_r[None, :]
            xk = tl.load(xk_ptrs, mask=mask_xk, other=0.0)
            

            k_up_chunk_0 = tl.load(k_up_base + offs_r[:, None] * stride_k_up_r + offs_d_0[None, :] * stride_k_up_d, mask=mask_r[:, None], other=0.0)
            k_up_chunk_1 = tl.load(k_up_base + offs_r[:, None] * stride_k_up_r + offs_d_1[None, :] * stride_k_up_d, mask=mask_r[:, None], other=0.0)
            
            k_block_0 += tl.dot(xk, k_up_chunk_0)
            k_block_1 += tl.dot(xk, k_up_chunk_1)
        

        angles = offs_s[:, None] * freqs[None, :]
        cos = tl.cos(angles)
        sin = tl.sin(angles)

        k_rope_0 = k_block_0 * cos - k_block_1 * sin
        k_rope_1 = k_block_1 * cos + k_block_0 * sin


        scores = (tl.sum(q_0[None, :] * k_rope_0, axis=1) + tl.sum(q_1[None, :] * k_rope_1, axis=1)) * sm_scale
        scores = tl.where(mask_s & (cur_k_lens > 0), scores, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(scores - m_ij)
        l_ij = l_i * alpha + tl.sum(p)

        acc = acc * alpha


        cur_v_lens = tl.load(L_v_ptr + pid_b * stride_lv_b + offs_s * stride_lv_s, mask=mask_s, other=0)
        cur_v_offs = tl.load(O_v_ptr + pid_b * stride_ov_b + offs_s * stride_ov_s, mask=mask_s, other=0)
        
        v_block = tl.zeros((BLOCK_SIZE_S, HEAD_DIM), dtype=tl.float32)
        for r0 in range(0, RANK_TOTAL, BLOCK_SIZE_R):
            offs_r = r0 + tl.arange(0, BLOCK_SIZE_R)
            mask_r = offs_r < RANK_TOTAL
            
            mask_xv = mask_s[:, None] & (offs_r[None, :] < cur_v_lens[:, None])

            xv_ptrs = V_flat_ptr + pid_b * stride_vflat_b + cur_v_offs[:, None] + offs_r[None, :]
            xv = tl.load(xv_ptrs, mask=mask_xv, other=0.0)
            
            v_up_ptrs = v_up_base + offs_r[:, None] * stride_v_up_r + offs_d[None, :] * stride_v_up_d
            v_up_chunk = tl.load(v_up_ptrs, mask=mask_r[:, None], other=0.0)
            
            v_block += tl.dot(xv, v_up_chunk)
            
        acc += tl.sum(p[:, None] * v_block, axis=0)

        m_i = m_ij
        l_i = l_ij


    acc = acc / l_i
    out_ptrs = Out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + offs_d * stride_out_d
    tl.store(out_ptrs, acc.to(Q_ptr.dtype.element_ty))

def fused_attention_dispatch(q, k_flat, k_up, v_flat, v_up, l_k, o_k, l_v, o_v, theta):
    q = q.squeeze(2) 
    batch, num_heads, head_dim = q.shape
    rank_total = k_up.shape[1]
    seq_len = l_k.shape[1]
    
    log_theta = math.log(theta)

    out = torch.empty((batch, num_heads, head_dim), dtype=q.dtype, device=q.device)
    
    grid = (batch, num_heads)
    _fused_flash_decode_sparse_kernel[grid](
        q, k_flat, k_up, v_flat, v_up, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_flat.stride(0), v_flat.stride(0),
        k_up.stride(0), k_up.stride(1), k_up.stride(2),
        v_up.stride(0), v_up.stride(1), v_up.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        l_k, l_k.stride(0), l_k.stride(1),
        o_k, o_k.stride(0), o_k.stride(1),
        l_v, l_v.stride(0), l_v.stride(1),
        o_v, o_v.stride(0), o_v.stride(1),
        seq_len, rank_total, log_theta,
        HEAD_DIM=head_dim, 
        BLOCK_SIZE_S=64, BLOCK_SIZE_R=64
    )
    return out
