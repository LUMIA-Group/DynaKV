import torch
from transformers.cache_utils import DynamicCache
from typing import Optional
import triton
import triton.language as tl

@triton.jit
def update_cache_decode_kernel(
    states_ptr, lens_ptr,
    flat_buf_ptr, stride_flat_b,
    lens_buf_ptr, offs_buf_ptr, flat_ptr_gpu,
    seq_ptr,
    stride_lens_b, stride_lens_s,
    stride_offs_b, stride_offs_s,
    MAX_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    

    length = tl.load(lens_ptr + pid) 
    offset = tl.atomic_add(flat_ptr_gpu + pid, length)
    
    l_ptr = lens_buf_ptr + pid * stride_lens_b + seq_ptr * stride_lens_s
    o_ptr = offs_buf_ptr + pid * stride_offs_b + seq_ptr * stride_offs_s
    tl.store(l_ptr, length)
    tl.store(o_ptr, offset)
    

    for i in range(0, MAX_DIM, BLOCK_SIZE):
        offs_d = i + tl.arange(0, BLOCK_SIZE)
        mask = offs_d < length
        val = tl.load(states_ptr + pid * MAX_DIM + offs_d, mask=mask, other=0.0)
        tl.store(flat_buf_ptr + pid * stride_flat_b + offset + offs_d, val, mask=mask)

def simple_decompress(flat_values, lengths, max_dim):
    B, S = lengths.shape
    device = flat_values.device
    

    states = torch.zeros((B, S, max_dim), dtype=flat_values.dtype, device=device)
    
    for b in range(B):
        lens_b = lengths[b]
        
        range_tensor = torch.arange(max_dim, device=device).unsqueeze(0)
        mask = range_tensor < lens_b.unsqueeze(1)
        

        total_valid = torch.sum(lens_b).item()
        

        valid_states = flat_values[b, :total_valid]
        states[b][mask] = valid_states
        
    return states

class CustomDynamicCache(DynamicCache):
    def __init__(self, config=None, max_batch_size=None, max_cache_len=None, dtype=None, device=None, layer_device_map=None, **kwargs):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self.device = device
        self.layer_device_map = layer_device_map
        
        self.key_cache_sp = []
        self.value_cache_sp = []
    
    def get_origin_states(self, max_dim, layer_idx):
        c_k = self.key_cache_sp[layer_idx]
        c_v = self.value_cache_sp[layer_idx]
        
        flat_k, lens_k, offs_k, ptr_k_gpu, seq_ptr, _ = c_k
        flat_v, lens_v, offs_v, ptr_v_gpu, _, _ = c_v


        k_states = simple_decompress(flat_k, lens_k[:, :seq_ptr], max_dim)
        v_states = simple_decompress(flat_v, lens_v[:, :seq_ptr], max_dim)
        
        return k_states, v_states

    def reset(self):
        self.key_cache_sp = []
        self.value_cache_sp = []

    def get_seq_length_sp(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache_sp) <= layer_idx: return 0
        if self.key_cache_sp[layer_idx] is None: return 0
        return self.key_cache_sp[layer_idx][4]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Expose the compressed sequence length through the HF Cache API."""
        return self.get_seq_length_sp(layer_idx)

    def _update_layer_cache(self, cache_list, layer_idx, new_states, new_len, tensor_d, is_decode=False):
        if len(cache_list) <= layer_idx:
            for _ in range(len(cache_list), layer_idx + 1):
                cache_list.append(None)

        current_cache = cache_list[layer_idx]
        B, new_S = new_len.shape
        

        if isinstance(tensor_d, torch.Tensor):
            _MAX_DIM = int(tensor_d.max().item())
        else:
            _MAX_DIM = int(tensor_d)

        if current_cache is None:

            init_seq = new_S
            

            actual_compressed_flat = int(new_len.sum(dim=1).max().item())

            init_flat = actual_compressed_flat
            

            flat_buf = new_states.new_zeros(B, init_flat)
            lens_buf = new_len.new_zeros(B, init_seq).to(torch.int32)
            offs_buf = new_len.new_zeros(B, init_seq).to(torch.int32)
            


            flat_ptr_gpu = torch.zeros(B, dtype=torch.int32, device=new_states.device)
            max_flat_ptr_cpu = [0] * B
            seq_ptr = 0

            cache_list[layer_idx] = [flat_buf, lens_buf, offs_buf, flat_ptr_gpu, seq_ptr, max_flat_ptr_cpu]

        cache = cache_list[layer_idx]
        flat_buf, lens_buf, offs_buf, flat_ptr_gpu, seq_ptr, max_flat_ptr_cpu = cache

        EXPAND_SEQ_STEP = 1024
        if seq_ptr + new_S > lens_buf.shape[1]:
            new_seq_cap = max(lens_buf.shape[1] + EXPAND_SEQ_STEP, seq_ptr + new_S)
            new_lens = lens_buf.new_zeros(B, new_seq_cap)
            new_lens[:, :seq_ptr].copy_(lens_buf[:, :seq_ptr])
            cache[1] = new_lens
            
            new_offs = offs_buf.new_zeros(B, new_seq_cap)
            new_offs[:, :seq_ptr].copy_(offs_buf[:, :seq_ptr])
            cache[2] = new_offs
            lens_buf, offs_buf = cache[1], cache[2]


        if is_decode:

            max_added_flat = new_S * _MAX_DIM 
        else:

            max_added_flat = int(new_len.sum(dim=1).max().item())
        

        if max(max_flat_ptr_cpu) + max_added_flat > flat_buf.shape[1]:

            real_ptrs = flat_ptr_gpu.cpu().tolist() 
            cache[5] = real_ptrs 
            
            if max(real_ptrs) + max_added_flat > flat_buf.shape[1]:
                expand_flat_step = EXPAND_SEQ_STEP * (_MAX_DIM // 2)
                new_cap = max(flat_buf.shape[1] + expand_flat_step, max(real_ptrs) + max_added_flat)
                new_flat_buf = flat_buf.new_zeros(B, int(new_cap))
                for b in range(B):
                    new_flat_buf[b, :real_ptrs[b]].copy_(flat_buf[b, :real_ptrs[b]])
                cache[0] = new_flat_buf
                flat_buf = cache[0]

        if is_decode:
            grid = (B,)
            update_cache_decode_kernel[grid](
                new_states, new_len,
                flat_buf, flat_buf.stride(0),
                lens_buf, offs_buf, flat_ptr_gpu,
                seq_ptr,
                lens_buf.stride(0), lens_buf.stride(1),
                offs_buf.stride(0), offs_buf.stride(1),
                MAX_DIM=_MAX_DIM,
                BLOCK_SIZE=64
            )
        else:

            for b in range(B):
                lens_b = new_len[b]
                states_b = new_states[b]
                

                cum_lens = torch.cumsum(lens_b, dim=0)
                start_ptr = flat_ptr_gpu[b].item()
                
                zero_tensor = torch.zeros(1, dtype=lens_b.dtype, device=lens_b.device)
                offs_b = start_ptr + torch.cat([zero_tensor, cum_lens[:-1]])
                
                lens_buf[b, seq_ptr:seq_ptr + new_S] = lens_b
                offs_buf[b, seq_ptr:seq_ptr + new_S] = offs_b
                

                range_tensor = torch.arange(_MAX_DIM, device=states_b.device).unsqueeze(0)
                mask = range_tensor < lens_b.unsqueeze(1)
                valid_states = states_b[mask].to(flat_buf.dtype)
                
                total_valid = valid_states.numel()

                flat_buf[b, start_ptr : start_ptr + total_valid] = valid_states
                
                flat_ptr_gpu[b] += total_valid
            
            cache[5] = flat_ptr_gpu.cpu().tolist()
            
        cache[4] = seq_ptr + new_S
        if is_decode:

            cache[5] = [ptr + max_added_flat for ptr in cache[5]]
            



    def update_sp(self, key_states, key_len, value_states, value_len, max_dim, layer_idx, tensor_d, k_scalar=None, v_scalar=None):
        is_decode = (key_states.shape[1] == 1)

        self._update_layer_cache(self.key_cache_sp, layer_idx, key_states, key_len, tensor_d, is_decode=is_decode)
        self._update_layer_cache(self.value_cache_sp, layer_idx, value_states, value_len, tensor_d, is_decode=is_decode)

        c_k = self.key_cache_sp[layer_idx]
        c_v = self.value_cache_sp[layer_idx]
        return (c_k[0], c_v[0], c_k[1][:, :c_k[4]], c_v[1][:, :c_v[4]], c_k[2][:, :c_k[4]], c_v[2][:, :c_v[4]])
