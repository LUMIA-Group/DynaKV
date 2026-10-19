#!/usr/bin/env python3
"""Evaluate perplexity on WikiText-2 with the soft/hard DynaKV model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from dynakv import get_model_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True, choices=["llama", "qwen"])
    parser.add_argument("--dataset-path", type=Path, default=Path("data/wikitext2"))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = get_model_class(args.model_type).from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    dataset = (
        load_from_disk(str(args.dataset_path))
        if args.dataset_path.exists()
        else load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    )
    if isinstance(dataset, DatasetDict):
        dataset = dataset["test"]
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]
    samples = (ids.numel() - 1) // args.sequence_length
    if args.max_samples is not None:
        samples = min(samples, args.max_samples)

    negative_log_likelihood = 0.0
    token_count = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index in range(samples):
            start = index * args.sequence_length
            batch = ids[start : start + args.sequence_length + 1].unsqueeze(0).to(device)
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            negative_log_likelihood += loss.item()
            token_count += batch.shape[1] - 1
            print(f"sample {index + 1}/{samples}")
    print(f"WikiText-2 PPL: {math.exp(negative_log_likelihood / token_count):.4f}")


if __name__ == "__main__":
    main()
