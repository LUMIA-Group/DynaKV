#!/usr/bin/env bash
set -euo pipefail
exec bash train_8B.sh llama "${1:?initialized model path}" "${2:?alpha (19, 32, or 50)}" "${3:-outputs/llama/a${2}}"
