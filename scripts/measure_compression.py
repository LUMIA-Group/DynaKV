#!/usr/bin/env python3
"""Measure the mean hard-mask retention and resulting KV compression."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from dynakv import get_model_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True, choices=["llama", "qwen"])
    parser.add_argument("--dataset-path", type=Path, default=Path("data/wikitext2"))
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = get_model_class(args.model_type).from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", alpha=0.0
    ).eval()
    dataset = (
        load_from_disk(str(args.dataset_path))
        if args.dataset_path.exists()
        else load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    )
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0][: args.tokens]
    device = next(model.parameters()).device
    rates = []
    with torch.inference_mode():
        for start in range(0, ids.numel(), args.sequence_length):
            batch = ids[start : start + args.sequence_length].unsqueeze(0).to(device)
            if batch.shape[1] < 2:
                continue
            output = model(
                input_ids=batch,
                labels=batch,
                use_cache=False,
                num_items_in_batch=batch.numel(),
            )
            rates.append(float(output.infos["mask_rate"]))
    retention = sum(rates) / len(rates)
    print(f"retention_rate={retention:.6f}")
    print(f"compression_rate={1.0 - retention:.6f}")


if __name__ == "__main__":
    main()
