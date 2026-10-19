#!/usr/bin/env bash
set -euo pipefail
exec bash train_8B.sh qwen "${1:?initialized model path}" "${2:?alpha (90, 150, or 250)}" "${3:-outputs/qwen/a${2}}"
