#!/usr/bin/env python3
"""Download the two datasets used by the DynaKV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    # The original togethercomputer repository was retired. This is a pinned
    # parquet backup of the same V1 sample (about 3.0 GB, 11 shards).
    "redpajama": (
        "liang2kl/RedPajama-Data-1T-Sample-Backup",
        None,
        "c2c1504dcb5887121b04787e36055e32aeb9017c",
    ),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", None),
}


def download(name: str, root: Path, cache_dir: Path) -> None:
    repo, subset, revision = DATASETS[name]
    destination = root / name
    if destination.exists():
        print(f"[skip] {name}: {destination} already exists")
        return
    print(f"[download] {repo}" + (f" ({subset})" if subset else ""))
    dataset = load_dataset(repo, subset, revision=revision, cache_dir=str(cache_dir))
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    print(f"[ready] {name}: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument(
        "--dataset", choices=["all", *DATASETS], required=True,
        help="Download one explicitly selected dataset (or both with 'all').",
    )
    args = parser.parse_args()
    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
