#!/usr/bin/env python3
"""Fine-tune the DynaKV gating and projection parameters on RedPajama."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from dynakv import get_model_class


def tokenize_batch(examples, tokenizer):
    eos = tokenizer.eos_token or ""
    return tokenizer([text + eos for text in examples["text"]], add_special_tokens=False)


def pack_batch(examples, sequence_length: int):
    concatenated = {key: sum(examples[key], []) for key in examples}
    total = len(concatenated["input_ids"])
    total = total // sequence_length * sequence_length
    packed = {
        key: [values[i : i + sequence_length] for i in range(0, total, sequence_length)]
        for key, values in concatenated.items()
    }
    packed["labels"] = [row.copy() for row in packed["input_ids"]]
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True, choices=["llama", "qwen"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/redpajama"))
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    deepspeed_config = json.loads(Path(args.deepspeed).read_text(encoding="utf-8"))
    expected = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    mismatches = {
        key: (deepspeed_config.get(key), value)
        for key, value in expected.items()
        if deepspeed_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CLI and DeepSpeed batch settings disagree: {mismatches}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = get_model_class(args.model_type)
    model = model_class.from_pretrained(
        args.model,
        alpha=args.alpha,
        t=args.temperature,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    if args.dataset_path.exists():
        dataset = load_from_disk(str(args.dataset_path))
    else:
        dataset = load_dataset(
            "liang2kl/RedPajama-Data-1T-Sample-Backup",
            revision="c2c1504dcb5887121b04787e36055e32aeb9017c",
            cache_dir=args.cache_dir,
        )
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    original_columns = dataset.column_names
    dataset = dataset.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=original_columns,
        desc="Tokenizing RedPajama",
    )
    dataset = dataset.map(
        partial(pack_batch, sequence_length=args.sequence_length),
        batched=True,
        num_proc=args.num_proc,
        desc=f"Packing {args.sequence_length}-token sequences",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        tf32=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
