#!/usr/bin/env python3
"""Collect pre-RoPE K/V activations and build per-layer PCA bases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Moments:
    count: int
    total: torch.Tensor
    gram: torch.Tensor

    @classmethod
    def empty(cls, dim: int, device: str) -> "Moments":
        return cls(
            0,
            torch.zeros(dim, dtype=torch.float32, device=device),
            torch.zeros(dim, dim, dtype=torch.float32, device=device),
        )

    def update(self, values: torch.Tensor) -> None:
        x = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float32)
        self.count += x.shape[0]
        self.total += x.sum(dim=0)
        self.gram += x.T @ x

    def basis(self) -> torch.Tensor:
        if self.count < 2:
            raise RuntimeError("At least two activation rows are required for PCA")
        covariance = (self.gram - torch.outer(self.total, self.total) / self.count) / (self.count - 1)
        _, vectors = torch.linalg.eigh(covariance)
        return vectors.flip(dims=(1,)).to(device="cpu", dtype=torch.float32).contiguous()


def get_text(dataset_path: Path | None) -> list[str]:
    if dataset_path and dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
    else:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    return [text for text in dataset["text"] if text and not text.isspace()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikitext-path", type=Path, default=Path("data/wikitext2"))
    parser.add_argument("--tokens", type=int, default=10_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = model.model.layers
    dim = layers[0].self_attn.k_proj.out_features
    moments = {
        (i, kind): Moments.empty(dim, args.device)
        for i in range(len(layers))
        for kind in ("k", "v")
    }
    handles = []

    def hook(layer: int, kind: str):
        def collect(_module, _inputs, output):
            moments[(layer, kind)].update(output)
        return collect

    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook(i, "v")))

    texts = get_text(args.wikitext_path)
    token_ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    target = min(args.tokens, token_ids.numel())
    try:
        with torch.inference_mode():
            for start in range(0, target, args.sequence_length):
                ids = token_ids[start : min(start + args.sequence_length, target)].unsqueeze(0).to(args.device)
                model(input_ids=ids, use_cache=False)
                print(f"collected {min(start + ids.shape[1], target):,}/{target:,} tokens")
    finally:
        for handle in handles:
            handle.remove()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (layer, kind), stats in moments.items():
        up = stats.basis()
        torch.save(up.unsqueeze(0), args.output_dir / f"L{layer}_{kind}_U.pt")
        torch.save(up.T.contiguous().unsqueeze(0), args.output_dir / f"L{layer}_{kind}_Ut.pt")
    print(f"PCA bases saved to {args.output_dir}")


if __name__ == "__main__":
    main()
