"""DynaKV: token-wise adaptive KV-cache compression."""

__version__ = "0.1.0"


def get_model_class(model_type: str):
    """Return the training/evaluation model class for a supported backbone."""
    normalized = model_type.lower().replace("_", "-")
    if normalized in {"llama", "llama3", "llama-3"}:
        from .custom_llama import CustomLlamaForCausalLM

        return CustomLlamaForCausalLM
    if normalized in {"qwen", "qwen3", "qwen-3"}:
        from .custom_qwen3 import CustomQwen3ForCausalLM

        return CustomQwen3ForCausalLM
    raise ValueError(f"Unsupported model type: {model_type!r}; choose llama or qwen.")
