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
def get_freq_multi_tokens(
    starting_idx,
    theta: tl.constexpr,
    NB_TOKENS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):

    offs_d = tl.arange(0, BLOCK_SIZE_D // 2) * 2
    freqs = offs_d.to(tl.float32) / BLOCK_SIZE_D
    

    log_theta = math.log(float(theta))
    freqs = tl.exp(-freqs * log_theta)
    
    pos = starting_idx + tl.arange(0, NB_TOKENS)
    angles = pos[:, None] * freqs[None, :]
    
    return tl.cos(angles), tl.sin(angles)

@triton.autotune(
    configs=get_configs(),
    key=[], 
)
@triton.jit
def _abx_sparse_fwd(
    a_ptr, b_ptr, x_flat_ptr, out_ptr,
    stride_ab, stride_ah, stride_ad,
    stride_bh, stride_br, stride_bd,
    stride_ob, stride_oh, stride_os,
    stride_xflat_b,
    L_vec_ptr, stride_l_b, stride_l_s,
    Offsets_ptr, stride_o_b, stride_o_s,
    R, D, seq_len, 
    head_dim: tl.constexpr, 
    rank_per_group: tl.constexpr, 
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_l = tl.program_id(axis=2)
    
    HEAD_GROUPS_ID = pid_h // HEADS_PER_GROUP 
    
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    
    offs_ls = (pid_l * BLOCK_SIZE_L) + offs_l
    mask_l = offs_ls < seq_len
    

    l_ptrs = L_vec_ptr + pid_b * stride_l_b + offs_ls * stride_l_s
    o_ptrs = Offsets_ptr + pid_b * stride_o_b + offs_ls * stride_o_s
    
    cur_lens = tl.load(l_ptrs, mask=mask_l, other=0)
    cur_offsets = tl.load(o_ptrs, mask=mask_l, other=0)
    
    max_len_in_block = tl.max(cur_lens, axis=0)
    safe_max_len = tl.minimum(max_len_in_block, rank_per_group)
    num_loop_k = tl.cdiv(safe_max_len, BLOCK_SIZE_R)
    

    A_ptrs = a_ptr + pid_b * stride_ab + pid_h * stride_ah + (offs_d[None, :] * stride_ad)
    
    B_ptr_head = b_ptr + HEAD_GROUPS_ID * stride_bh
    B_ptr_k_start = B_ptr_head + (offs_r[:, None] * stride_br + offs_d[None, :] * stride_bd)
    
    O_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + (offs_ls * stride_os)

    xb_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    mask_d0 = offs_d < head_dim
    mask_d1 = (offs_d + BLOCK_SIZE_D) < head_dim
    mask_seq_valid = mask_l[:, None]
    stride_b_k = BLOCK_SIZE_R * stride_br
    
    for k in range(num_loop_k):
        base_r = k * BLOCK_SIZE_R
        current_r_indices = base_r + offs_r
        
        mask_rank_valid = current_r_indices[None, :] < cur_lens[:, None]
        mask_x = mask_rank_valid & mask_seq_valid
        

        x_ptrs = x_flat_ptr + pid_b * stride_xflat_b + cur_offsets[:, None] + current_r_indices[None, :]
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        mask_r = current_r_indices < R
        mask_b = mask_r[:, None]
        cur_b_ptrs = B_ptr_k_start + (k * stride_b_k)
        
        b_0 = tl.load(cur_b_ptrs, mask=mask_b & mask_d0[None, :], other=0.0)
        b_1 = tl.load(cur_b_ptrs + BLOCK_SIZE_D * stride_bd, mask=mask_b & mask_d1[None, :], other=0.0)
        
        xb_0 = tl.dot(x, b_0, xb_0)
        xb_1 = tl.dot(x, b_1, xb_1)

    start_block = pid_l * BLOCK_SIZE_L
    cos, sin = get_freq_multi_tokens(
        starting_idx=start_block, 
        theta=THETA, 
        NB_TOKENS=BLOCK_SIZE_L,
        BLOCK_SIZE_D=BLOCK_SIZE_D * 2 
    )
    
    xb_0 = xb_0.to(tl.float32)
    xb_1 = xb_1.to(tl.float32)
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)

    xb_rope_0 = xb_0 * cos - xb_1 * sin
    xb_rope_1 = xb_1 * cos + xb_0 * sin
    
    a_0 = tl.load(A_ptrs, mask=mask_d0[None, :], other=0.0).to(tl.float32)
    a_1 = tl.load(A_ptrs + BLOCK_SIZE_D * stride_ad, mask=mask_d1[None, :], other=0.0).to(tl.float32)
    
    abx_0 = tl.sum(a_0 * xb_rope_0, 1)
    abx_1 = tl.sum(a_1 * xb_rope_1, 1)
    abx = abx_0 + abx_1
    
    has_content = cur_lens > 0
    abx = tl.where(has_content, abx, -1.0e4)
    abx = abx * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))
    
    tl.store(O_ptrs, abx.to(out_ptr.dtype.element_ty), mask=mask_l)

def abx_sparse(a: torch.Tensor, b: torch.Tensor, x_flat: torch.Tensor, l_vec: torch.Tensor, offsets: torch.Tensor = None, theta: float = 10000.0) -> torch.Tensor:
    batch, num_heads_total, head_dim = a.shape
    b_batch, seq_len = l_vec.shape
    rank_per_head_groups = b.shape[1]

    if offsets is None:
        raise ValueError("offsets are required for the variable-length key cache")

    heads_per_group = num_heads_total // b.shape[0]
    
    out = torch.empty((batch, num_heads_total, seq_len), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_D = 64 
    

    grid = lambda META: (batch, num_heads_total, triton.cdiv(seq_len, META["BLOCK_SIZE_L"]))
    
    _abx_sparse_fwd[grid](
        a, b, x_flat, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        x_flat.stride(0),
        l_vec, l_vec.stride(0), l_vec.stride(1),
        offsets, offsets.stride(0), offsets.stride(1),
        rank_per_head_groups, head_dim, seq_len,
        head_dim, rank_per_head_groups,
        BLOCK_SIZE_D = BLOCK_SIZE_D,
        HEADS_PER_GROUP = heads_per_group, 
        THETA = theta,
    )
    return out

def get_fused_proj_configs():
    configs = []

    block_s_list = [16, 32, 64] 
    num_warps_list = [4, 8]
    
    for bs in block_s_list:
        for nw in num_warps_list:
            configs.append(
                triton.Config(
                    {'BLOCK_SIZE_S': bs},
                    num_warps=nw,
                )
            )
    return configs

@triton.autotune(
    configs=get_fused_proj_configs(),
    key=['HEAD_DIM', 'HIDDEN_SIZE', 'RANK_TOTAL'],
)
@triton.jit
def _avx_proj_fwd(
    a_ptr, x_flat_ptr, out_ptr,
    v_up_ptr, w_o_ptr, 
    stride_az, stride_ah, stride_as,
    stride_oz, stride_oh, 
    stride_xflat_b,
    stride_vz, stride_vr, stride_vd, 
    stride_wo_out, stride_wo_in, 
    L_vec_ptr, stride_l_b, stride_l_s,
    Offsets_ptr, stride_o_b, stride_o_s,
    seq_len,
    RANK_TOTAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr, 
    BLOCK_SIZE_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    a_ptr_base = a_ptr + pid_b * stride_az + pid_h * stride_ah
    l_ptr_base = L_vec_ptr + pid_b * stride_l_b
    o_ptr_base = Offsets_ptr + pid_b * stride_o_b
    
    kv_head_idx = pid_h // GROUP_SIZE
    v_ptr_base = v_up_ptr + kv_head_idx * stride_vz

    m_global = -float('inf')
    l_global = 0.0
    
    acc_r = tl.zeros((BLOCK_SIZE_R,), dtype=tl.float32)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    mask_r = offs_r < RANK_TOTAL

    for s0 in range(0, seq_len, BLOCK_SIZE_S):
        offs_s = s0 + tl.arange(0, BLOCK_SIZE_S)
        mask_s = offs_s < seq_len
        
        scores = tl.load(a_ptr_base + offs_s * stride_as, mask=mask_s, other=-float('inf'))
        
        m_new = tl.maximum(m_global, tl.max(scores, axis=0))
        alpha = tl.exp(m_global - m_new)
        p = tl.exp(scores - m_new)
        p_masked = tl.where(mask_s, p, 0.0)

        cur_lens = tl.load(l_ptr_base + offs_s * stride_l_s, mask=mask_s, other=0)
        cur_offs = tl.load(o_ptr_base + offs_s * stride_o_s, mask=mask_s, other=0)

        mask_x = mask_s[:, None] & mask_r[None, :] & (offs_r[None, :] < cur_lens[:, None])

        x_ptrs = x_flat_ptr + pid_b * stride_xflat_b + cur_offs[:, None] + offs_r[None, :]
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        acc_r = acc_r * alpha + tl.sum(p_masked[:, None] * x, axis=0)

        m_global = m_new
        l_global = l_global * alpha + tl.sum(p_masked, axis=0)

    acc_r = acc_r / tl.maximum(l_global, 1e-6)

    offs_d = tl.arange(0, HEAD_DIM)
    v_ptrs = v_ptr_base + offs_r[:, None] * stride_vr + offs_d[None, :] * stride_vd
    
    v_chunk = tl.load(v_ptrs, mask=mask_r[:, None], other=0.0)
    temp_head = tl.sum(acc_r[:, None] * v_chunk, axis=0)

    offs_d_out = tl.arange(0, HEAD_DIM)
    out_ptrs = out_ptr + pid_b * stride_oz + pid_h * stride_oh + offs_d_out
    tl.store(out_ptrs, temp_head.to(a_ptr.dtype.element_ty))

def avx(
    a: torch.Tensor,
    x_flat: torch.Tensor,
    v_up: torch.Tensor,
    l_vec: torch.Tensor,
    offsets: torch.Tensor = None
) -> torch.Tensor:
    
    if a.ndim == 4:
        a = a.squeeze(2)

    batch, num_heads, seq_len = a.shape
    num_kv_heads, rank_total, head_dim = v_up.shape
    
    import triton
    b_r = triton.next_power_of_2(rank_total) 

    if offsets is None:
        raise ValueError("offsets are required for the variable-length value cache")

    out = torch.empty((batch, num_heads, head_dim), dtype=a.dtype, device=a.device)
    
    grid = lambda META: (batch, num_heads)
    
    _avx_proj_fwd[grid](
        a, x_flat, out, 
        v_up, None, 
        a.stride(0), a.stride(1), a.stride(2),
        out.stride(0), out.stride(1), 
        x_flat.stride(0),
        v_up.stride(0), v_up.stride(1), v_up.stride(2), 
        0, 0, 
        l_vec, l_vec.stride(0), l_vec.stride(1),
        offsets, offsets.stride(0), offsets.stride(1),
        seq_len,
        RANK_TOTAL=rank_total,
        HEAD_DIM=head_dim,
        HIDDEN_SIZE=0,
        GROUP_SIZE=num_heads // num_kv_heads,
        BLOCK_SIZE_R=b_r
    )
    return out
