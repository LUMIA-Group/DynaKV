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
import json
import warnings

prev_size = 0


enable_streamingLLM = False
reserve_token_num = 1024

def apply_rotary_pos_emb_(x, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class KVProcessor:
    def __init__(self, dim):
        self.dump = 1
        self.dim = dim

        self.Wk = nn.Parameter(torch.randn(dim, dim, dtype=self.dtype))  
        self.bk = nn.Parameter(torch.randn(1, dim, dtype=self.dtype))  

        self.Wv = nn.Parameter(torch.randn(dim, dim, dtype=self.dtype))  
        self.bv = nn.Parameter(torch.randn(1, dim, dtype=self.dtype))  

        pca_dir = os.environ.get("DYNAKV_PCA_DIR")

        def load_basis(suffix):
            if pca_dir:
                path = os.path.join(pca_dir, f"L{self.layer_idx}_{suffix}.pt")
                if os.path.isfile(path):
                    value = torch.load(path, map_location="cpu", weights_only=True)
                    if value.ndim == 3:
                        value = value[0]
                    return value.to(self.dtype)
            # Checkpoints already contain these parameters. Identity is a safe
            # constructor default before from_pretrained loads their values.
            if self.layer_idx == 0 and not getattr(self.config, "_dynakv_identity_warned", False):
                warnings.warn(
                    "DYNAKV_PCA_DIR is unset or incomplete; initializing the PCA "
                    "basis as identity. This is expected while loading a trained "
                    "DynaKV checkpoint, but not when converting a base model."
                )
                self.config._dynakv_identity_warned = True
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
        
        linear_all = (
            torch.einsum("bsd,de->bse", states, W) + 
            b.view(1, 1, self.dim)
        )
            
        mask_all = torch.cumsum(torch.softmax(linear_all, dim=-1), dim=-1)
        mask_all = torch.flip(mask_all, dims=[-1]) 

        mask_soft_all = mask_all.half()
        mask_hard_all = (mask_soft_all >= 0.1).to(mask_soft_all.dtype)
        
        if self.dump == 0 and typee == 0:
            with open("mask_dump_key.jsonl", "a") as f:
                data_to_write = {
                "layer_idx": self.layer_idx,
                "mask_hard": mask_hard_all.detach().cpu().tolist()
                }
                f.write(json.dumps(data_to_write) + "\n")

        if self.dump == 0 and typee == 1:
            with open("mask_dump_value.jsonl", "a") as f:
                data_to_write = {
                "layer_idx": self.layer_idx,
                "mask_hard": mask_hard_all.detach().cpu().tolist()
                }
                f.write(json.dumps(data_to_write) + "\n")

        if self.training:
            mask_return = mask_soft_all
        else:
            mask_return = mask_hard_all
        return mask_soft_all, mask_hard_all, mask_return.mean()


    def key_value_forward(self, states, typee):
        if states.shape[1] == 0:
            return states, states, torch.tensor(0.0, device=states.device, dtype=states.dtype)

        mask_soft_all, mask_hard_all, mask_rate = self.calc_mask(states, typee)

        states_soft = states*mask_soft_all
        states_hard = states*mask_hard_all
        return states_soft, states_hard, mask_rate

    def kv_process(self, key_states, value_states, past_len):
        if (past_len >= prev_size):
            key_states  , key_states_hard   ,key_mask_rate   = self.key_value_forward(key_states  , 0)
            value_states, value_states_hard ,value_mask_rate = self.key_value_forward(value_states, 1)
        else:
            reserve_num = prev_size - past_len
            key_prev = key_states[:, :reserve_num, :]
            key_recent = key_states[:, reserve_num:, :]

            value_prev = value_states[:, :reserve_num, :]
            value_recent = value_states[:, reserve_num:, :]

            key_recent_soft  , key_recent_hard  , key_mask_rate   = self.key_value_forward(key_recent  , 0)
            value_recent_soft, value_recent_hard, value_mask_rate = self.key_value_forward(value_recent, 1)

            key_states      = torch.cat([key_prev, key_recent_soft], dim=1)
            key_states_hard = torch.cat([key_prev, key_recent_hard], dim=1)
            value_states      = torch.cat([value_prev, value_recent_soft], dim=1)
            value_states_hard = torch.cat([value_prev, value_recent_hard], dim=1)

        return key_states, key_states_hard, value_states, value_states_hard, (key_mask_rate + value_mask_rate) / 2

    def kv_down_up(self, key_all, value_all, hidden_shape, past_key_value, device, dtype):

        if enable_streamingLLM and (past_key_value is not None):
            key_states2 = key_all.view(hidden_shape).transpose(1, 2)
            value_states2 = value_all.view(hidden_shape).transpose(1, 2)
            key_states2, value_states2 = past_key_value.update2(key_states2, value_states2, self.layer_idx)

        if past_key_value is None:
            past_len = 0
        else:
            past_len = past_key_value.get_seq_length(self.layer_idx)

        key_states_down = torch.einsum('bsd,nd->bsn', key_all, self.k_down)
        value_states_down = torch.einsum('bsd,nd->bsn', value_all, self.v_down)
        
        mask_rate = torch.tensor(0.0, device=device, dtype=dtype)
        key_states_down, key_states_down_hard, value_states_down, value_states_down_hard, mask_rate = self.kv_process(key_states_down, value_states_down, past_len)
        
        if past_key_value is not None:
            key_states_down, value_states_down = past_key_value.update(key_states_down_hard, value_states_down_hard, self.layer_idx)
            
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
        
        if enable_streamingLLM and (past_key_value is not None):
            sequence_length = key_states.shape[-2]
            reserve = min(reserve_token_num, sequence_length)
            key_states = torch.cat([key_states[:, :, :-reserve, :], key_states2[:, :, -reserve:, :]], dim=2)
            value_states = torch.cat([value_states[:, :, :-reserve, :], value_states2[:, :, -reserve:, :]], dim=2)
        
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

        KVProcessor.__init__(self, self.config.num_key_value_heads * self.head_dim)

        self.rotary_emb = LlamaRotaryEmbedding(config)

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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb_(query_states, cos, sin)

        key_all   = self.k_proj(hidden_states)
        value_all = self.v_proj(hidden_states)

        device = query_states.device
        dtype  = query_states.dtype
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

        KVProcessor.__init__(self, self.config.num_key_value_heads * self.head_dim)

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
