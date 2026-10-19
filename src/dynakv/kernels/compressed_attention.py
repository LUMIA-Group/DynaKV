import torch
import torch.nn as nn
from typing import Optional, Callable
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward, rotate_half, LlamaAttention, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
import os
from .router import compressed_decode_attention

pca_path_llama3 = os.environ.get("DYNAKV_PCA_DIR", "")
pca_path_qwen3 = os.environ.get("DYNAKV_PCA_DIR", "")
upper_ratio = 1

def apply_rotary_pos_emb_(x, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class KVProcessor:
    def __init__(self, dim, pca_path):
        self.dim = dim

        self.Wk = nn.Parameter(torch.randn(dim, dim, dtype=self.dtype))  
        self.bk = nn.Parameter(torch.randn(1, dim, dtype=self.dtype))  

        self.Wv = nn.Parameter(torch.randn(dim, dim, dtype=self.dtype))  
        self.bv = nn.Parameter(torch.randn(1, dim, dtype=self.dtype))  

        def load_basis(suffix):
            path = os.path.join(pca_path, f"L{self.layer_idx}_{suffix}.pt") if pca_path else ""
            if path and os.path.isfile(path):
                value = torch.load(path, map_location="cpu", weights_only=True)
                return (value[0] if value.ndim == 3 else value).to(self.dtype)
            return torch.eye(dim, dtype=self.dtype)

        self.k_down = nn.Parameter(load_basis("k_Ut"))
        self.k_up = nn.Parameter(load_basis("k_U"))
        self.v_down = nn.Parameter(load_basis("v_Ut"))
        self.v_up = nn.Parameter(load_basis("v_U"))


    def calc_mask(self, states, typee):
        if typee == 0:
            W = self.Wk
            b = self.bk
        else:
            W = self.Wv
            b = self.bv
        
        linear_all = torch.nn.functional.linear(states, W.t(), b.view(-1))
            
        mask_all = torch.cumsum(torch.softmax(linear_all, dim=-1), dim=-1)
        mask_all = torch.flip(mask_all, dims=[-1]) 

        mask_soft_all = mask_all.half()
        mask_hard_all = (mask_soft_all >= 0.1).to(mask_soft_all.dtype)
        
        if self.training:
            mask_ret_all = mask_soft_all
        else:
            mask_ret_all = mask_hard_all

        return mask_soft_all, mask_hard_all, mask_ret_all.mean()


    def key_value_forward(self, states, typee):
        if states.shape[1] == 0:
            return states, torch.empty(states.shape[0], 0, device=states.device, dtype=torch.long), torch.tensor(0.0, device=states.device, dtype=states.dtype)
        
        mask_soft_all, mask_hard_all, mask_rate = self.calc_mask(states, typee)

        lengths = mask_hard_all.sum(dim=-1).long()

        states_soft = states*mask_soft_all
        states_hard = states*mask_hard_all
        return states_soft, states_hard, lengths, mask_rate
    
    def kv_process_fused(self, key_states, value_states):


        B, S, D = key_states.shape
        flat_k = key_states.view(-1, D)
        flat_v = value_states.view(-1, D)
        

        logits_k = torch.addmm(self.bk.view(-1), flat_k, self.Wk)
        logits_v = torch.addmm(self.bv.view(-1), flat_v, self.Wv)
        

        logits_all = torch.cat([logits_k, logits_v], dim=0)
        

        mask_all = torch.softmax(logits_all, dim=-1).cumsum(dim=-1).flip(dims=[-1])
        mask_hard = (mask_all >= 0.1)
        

        mask_k, mask_v = mask_hard.chunk(2, dim=0)
        

        mask_k = mask_k.view(B, S, D)
        mask_v = mask_v.view(B, S, D)

        mask_k[..., int(self.low_rank):] = 0
        mask_v[..., int(self.low_rank):] = 0


        len_k = mask_k.sum(dim=-1, dtype=torch.int32)
        len_v = mask_v.sum(dim=-1, dtype=torch.int32)
        
        k_out = key_states * mask_k
        v_out = value_states * mask_v
        
        return k_out, v_out, len_k, len_v, 1, 1

    def kv_process(self, key_states, value_states, past_len=0):
        key_states_hard, value_states_hard, key_lengths, value_lengths, key_mask_rate, value_mask_rate = self.kv_process_fused(key_states, value_states)
        mask_rate = (key_mask_rate + value_mask_rate) / 2
        return key_states, key_states_hard, key_lengths, value_states, value_states_hard, value_lengths, mask_rate

    def kv_down_up_decoding(self, key_all, value_all, hidden_shape, device, dtype):
        key_states_down, key_states_down_hard, key_lengths, value_states_down, value_states_down_hard, value_lengths, _ = self.kv_process(key_all, value_all)
        return key_states_down_hard, key_lengths, value_states_down_hard, value_lengths
    
    def kv_down_up(self, key_all, value_all, hidden_shape, past_key_value, device, dtype, return_compressed=False):

        if past_key_value is None:
            past_len = 0
        else:
            past_len = past_key_value.get_seq_length(self.layer_idx)

        key_states = key_all
        value_states = value_all

        key_states_down, key_states_down_hard, key_lengths, value_states_down, value_states_down_hard, value_lengths, mask_rate = self.kv_process(key_states, value_states, past_len)

        if past_key_value is not None:
            d_down = self.k_down.shape[0] 
            if return_compressed:
                return key_states_down_hard, key_lengths, value_states_down_hard, value_lengths, mask_rate





            



            



            


            


            past_key_value.update_sp(key_states, key_lengths, value_states, value_lengths, d_down, self.layer_idx, self.tensor_d)
            
            key_states_down, value_states_down = past_key_value.get_origin_states(d_down, self.layer_idx)

            key_all = torch.einsum('bsd,nd->bsn', key_states_down, self.k_up)
            value_all = torch.einsum('bsd,nd->bsn', value_states_down, self.v_up)


            hidden_shape = (key_all.shape[0], key_all.shape[1], -1, self.head_dim)

            key_states = key_all.view(hidden_shape).transpose(1, 2)
            value_states = value_all.view(hidden_shape).transpose(1, 2)

        else:

            if self.training:
                key_states_down = key_states_down.to(self.dtype)
                value_states_down = value_states_down.to(self.dtype)
            else:
                key_states_down = key_states_down_hard.to(self.dtype)
                value_states_down = value_states_down_hard.to(self.dtype)

            key_all = torch.einsum('bsd,nd->bsn', key_states_down, self.k_up)
            value_all = torch.einsum('bsd,nd->bsn', value_states_down, self.v_up)
            
            key_states = key_all.view(hidden_shape).transpose(1, 2)
            value_states = value_all.view(hidden_shape).transpose(1, 2)
        
        return key_states, value_states, mask_rate

    def calc_cos_sin(self, states):
        seq_len = states.size(-2)
        device = states.device

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = position_ids.expand(states.size(0), -1)  # [b, seq_len]

        return self.rotary_emb(states, position_ids)

class CustomAttention(LlamaAttention, KVProcessor): 
    def __init__(self, config: LlamaConfig, layer_idx: int, original_attn: LlamaAttention | None = None):
        LlamaAttention.__init__(self, config, layer_idx)

        if original_attn is not None:
            self.__dict__.update(original_attn.__dict__)
        self.dtype = self.q_proj.weight.dtype

        KVProcessor.__init__(self, self.config.num_key_value_heads * self.head_dim, pca_path_llama3)

        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.current_theta = getattr(self.config, "rope_theta", 10000.0)
        self.d = self.config.num_key_value_heads * self.head_dim
        self.merged = 0
        self.B = None
        self.tensor_d = None
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

        self.cuda_graph = None
        self.static_k = None
        self.static_v = None
        self.static_k_out = None
        self.static_v_out = None
        self.static_len_k = None
        self.static_len_v = None
    
        self.low_rank = self.d * upper_ratio
    def capture_graph(self, batch_size, hidden_dim, device, dtype):

        self.static_k = torch.zeros((batch_size, 1, hidden_dim), device=device, dtype=dtype)
        self.static_v = torch.zeros((batch_size, 1, hidden_dim), device=device, dtype=dtype)
        

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.kv_down_up_decoding(self.static_k, self.static_v, None, device, dtype)
        torch.cuda.current_stream().wait_stream(s)


        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):


            self.static_k_out, self.static_len_k, self.static_v_out, self.static_len_v = \
                self.kv_down_up_decoding(self.static_k, self.static_v, None, device, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        if self.merged == 0:
            self.merged = 1
            with torch.no_grad():

                self.k_proj.weight.data = torch.matmul(self.k_down, self.k_proj.weight.data)
                self.v_proj.weight.data = torch.matmul(self.v_down, self.v_proj.weight.data)
                

                if self.k_proj.bias is not None:
                    self.k_proj.bias.data = torch.matmul(self.k_down, self.k_proj.bias.data)
                    
                if self.v_proj.bias is not None:
                    self.v_proj.bias.data = torch.matmul(self.v_down, self.v_proj.bias.data)


                k_up_sliced = self.k_up[:, :int(self.low_rank)]
                v_up_sliced = self.v_up[:, :int(self.low_rank)]
                

                self.B = k_up_sliced.view(self.config.num_key_value_heads, self.head_dim, int(self.low_rank)).transpose(1, 2).contiguous()
                self.B = self.B.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=0).contiguous()
                
                self.v_up_reshaped = v_up_sliced.view(self.config.num_key_value_heads, self.head_dim, int(self.low_rank)).transpose(1, 2).contiguous()
                self.v_up_reshaped = self.v_up_reshaped.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=0).contiguous()                
            
                self.tensor_d = torch.full((hidden_states.shape[0], 1), int(self.low_rank), device=hidden_states.device, dtype=torch.int32)    
                

                self.k_down.data = torch.eye(self.k_down.shape[0], dtype=self.dtype, device=self.k_down.device)
                self.v_down.data = torch.eye(self.v_down.shape[0], dtype=self.dtype, device=self.v_down.device)
                

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        batch_size, seq_len_q, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_all   = self.k_proj(hidden_states)
        value_all = self.v_proj(hidden_states)


        cos, sin = position_embeddings
        

        is_decoding = past_key_value is not None and seq_len_q == 1 

        device = hidden_states.device
        dtype = hidden_states.dtype
        mask_rate = -1
        if is_decoding:
            bsz = query_states.shape[0]
            query_states = apply_rotary_pos_emb_(query_states, cos, sin)
            device = query_states.device
            dtype  = query_states.dtype
            
            q_fused = query_states.contiguous()
            

            if self.cuda_graph is None or self.static_k.shape[0] != bsz:
                self.capture_graph(bsz, key_all.shape[-1], device, dtype)
            
            self.static_k.copy_(key_all)
            self.static_v.copy_(value_all)

            self.cuda_graph.replay()
            

            flat_key, flat_value, l_vec_k, l_vec_v, offsets_k, offsets_v = past_key_value.update_sp(
                self.static_k_out, self.static_len_k, 
                self.static_v_out, self.static_len_v, 
                self.d, self.layer_idx, int(self.low_rank)
            )
            cache_length = past_key_value.get_seq_length_sp(self.layer_idx)
            attn_output_sram = compressed_decode_attention(
                q=q_fused, 
                k_flat=flat_key, 
                k_up=self.B,  
                v_flat=flat_value, 
                v_up=self.v_up_reshaped, 
                key_lengths=l_vec_k,
                key_offsets=offsets_k,
                value_lengths=l_vec_v,
                value_offsets=offsets_v,
                theta=self.current_theta,
                cache_length=cache_length,
            )
            

            num_heads = self.config.num_attention_heads
            attn_output = attn_output_sram.view(bsz, 1, num_heads * self.head_dim).contiguous()
            

            attn_output = self.o_proj(attn_output)
            
            return attn_output, None, mask_rate

        else:
            query_states = apply_rotary_pos_emb_(query_states, cos, sin)

            key_states, value_states, mask_rate = self.kv_down_up(key_all, value_all, hidden_shape, past_key_value, device, dtype)

            if past_key_value is not None:
                cos, sin = self.calc_cos_sin(key_states)

            key_states = apply_rotary_pos_emb_(key_states, cos, sin)

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights, mask_rate

class CustomQwen3Attention(Qwen3Attention, KVProcessor):
    def __init__(self, config: LlamaConfig, layer_idx: int, original_attn: Qwen3Attention | None = None):
        Qwen3Attention.__init__(self, config, layer_idx)

        if original_attn is not None:
            self.__dict__.update(original_attn.__dict__)
        self.dtype = self.q_proj.weight.dtype

        KVProcessor.__init__(self, self.config.num_key_value_heads * self.head_dim, pca_path_qwen3)

        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb_(query_states, cos, sin)

        key_all   = self.k_proj(hidden_states)
        value_all = self.v_proj(hidden_states)

        device = query_states.device
        dtype  = query_states.dtype
        key_states, value_states, mask_rate = self.kv_down_up(key_all, value_all, hidden_shape, past_key_value, device, dtype)

        key_states = self.k_norm(key_states)

        if past_key_value is not None:
            cos, sin = self.calc_cos_sin(key_states)

        key_states = apply_rotary_pos_emb_(key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, mask_rate
