import torch
from typing import Any, Optional
from transformers.cache_utils import DynamicCache

class CustomDynamicCache(DynamicCache):

    def __init__(self, config=None, max_batch_size=None, max_cache_len=None, dtype=None, device=None, layer_device_map=None, **kwargs):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self.device = device
        self.layer_device_map = layer_device_map


        self.key_cache2: list[torch.Tensor] = []
        self.value_cache2: list[torch.Tensor] = []

    def reset(self):
        self._seen_tokens = 0
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.key_cache2 = []
        self.value_cache2 = []
    
    def update2(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache2) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache2), layer_idx):
                    self.key_cache2.append(torch.tensor([]))
                    self.value_cache2.append(torch.tensor([]))
                self.key_cache2.append(key_states)
                self.value_cache2.append(value_states)
            elif (
                not self.key_cache2[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache2[layer_idx] = key_states
                self.value_cache2[layer_idx] = value_states
            else:
                self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
                self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)
                

                if len(self.key_cache) <= layer_idx:
                    self.key_cache2[layer_idx] = key_states
                    self.value_cache2[layer_idx] = value_states
        
        return self.key_cache2[layer_idx], self.value_cache2[layer_idx]
