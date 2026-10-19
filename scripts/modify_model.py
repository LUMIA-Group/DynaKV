#!/usr/bin/env python3
"""Replace attention blocks and initialize DynaKV from PCA bases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynakv.custom_attn import CustomAttention, CustomQwen3Attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local base-model path")
    parser.add_argument("--model-type", required=True, choices=["llama", "qwen"])
    parser.add_argument("--pca-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    os.environ["DYNAKV_PCA_DIR"] = str(args.pca_dir.resolve())
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    attention_class = CustomAttention if args.model_type == "llama" else CustomQwen3Attention

    expected = [
        args.pca_dir / f"L{index}_{kind}_{direction}.pt"
        for index in range(len(model.model.layers))
        for kind in ("k", "v")
        for direction in ("U", "Ut")
    ]
    missing_pca = [str(path) for path in expected if not path.is_file()]
    if missing_pca:
        preview = "\n".join(missing_pca[:8])
        raise FileNotFoundError(
            f"PCA directory is incomplete ({len(missing_pca)} files missing). First entries:\n{preview}"
        )

    for index, layer in enumerate(model.model.layers):
        original = layer.self_attn
        replacement = attention_class(model.config, index)
        missing, unexpected = replacement.load_state_dict(original.state_dict(), strict=False)
        unexpected = [name for name in unexpected if name not in {"rotary_emb.inv_freq"}]
        if unexpected:
            raise RuntimeError(f"Unexpected base attention weights at layer {index}: {unexpected}")
        replacement.to(device=original.q_proj.weight.device, dtype=dtype)
        layer.self_attn = replacement
        print(f"converted layer {index:02d}; new parameters: {', '.join(missing)}")

    model.config.dynakv_model_type = args.model_type
    model.config.dynakv_format_version = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model, use_fast=True).save_pretrained(args.output_dir)
    print(f"Initialized checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
