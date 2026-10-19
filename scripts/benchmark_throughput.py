#!/usr/bin/env python3
"""Benchmark generation throughput (defaults: BS 1/2/4, 100 tokens, mean of 3)."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from dynakv import get_model_class


@torch.inference_mode()
def timed_generate(model, inputs, new_tokens: int, pad_token_id: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True, choices=["llama", "qwen"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--generation-length", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--implementation",
        choices=["physical", "research"],
        default="physical",
        help="Use the real compressed cache/kernels or the accuracy-reference model.",
    )
    parser.add_argument("--prompt", default="DynaKV reduces the memory used by attention by")
    args = parser.parse_args()

    if args.implementation == "physical":
        os.environ["DYNAKV_PHYSICAL_KERNELS"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_model_class(args.model_type).from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    results = {
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "generation_length": args.generation_length,
        "implementation": args.implementation,
        "measurements": {},
    }
    for batch_size in args.batch_sizes:
        inputs = tokenizer([args.prompt] * batch_size, padding=True, return_tensors="pt").to("cuda")
        for _ in range(args.warmup_runs):
            timed_generate(model, inputs, args.generation_length, tokenizer.pad_token_id)
        seconds = [
            timed_generate(model, inputs, args.generation_length, tokenizer.pad_token_id)
            for _ in range(args.runs)
        ]
        rates = [batch_size * args.generation_length / value for value in seconds]
        results["measurements"][str(batch_size)] = {
            "tokens_per_second": sum(rates) / len(rates),
            "runs": rates,
        }
        print(
            f"BS={batch_size}: "
            f"{results['measurements'][str(batch_size)]['tokens_per_second']:.2f} token/s"
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
