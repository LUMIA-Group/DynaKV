#!/usr/bin/env python3
"""Download one released checkpoint directory from Elemmire/DynaKV."""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subfolder", help="For example: qwen3-8B-base/qwen_a90")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    path = snapshot_download(
        repo_id="Elemmire/DynaKV",
        allow_patterns=[f"{args.subfolder}/**"],
        local_dir=args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
