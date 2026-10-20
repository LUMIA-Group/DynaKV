# DynaKV

Official research code for **One Size Does Not Fit All: Token-Wise Adaptive Compression for KV Cache**.
DynaKV applies a PCA-initialized spectral transform and learns a token-wise cutoff,
so important tokens retain more KV dimensions while redundant tokens retain fewer.

This repository supports the base checkpoints:

- `meta-llama/Meta-Llama-3-8B`
- `Qwen/Qwen3-8B-Base`

## Released operating points

| Backbone | Alpha | Measured KV compression |
|---|---:|---:|
| Llama 3 8B | 19 | 47% |
| Llama 3 8B | 32 | 29% |
| Llama 3 8B | 50 | 20% |
| Qwen3 8B Base | 90 | 48% |
| Qwen3 8B Base | 150 | 30% |
| Qwen3 8B Base | 250 | 18% |

Trained checkpoints are published at
[`Elemmire/DynaKV`](https://huggingface.co/Elemmire/DynaKV/tree/main). For example:

```bash
python scripts/download_checkpoint.py qwen3-8B-base/qwen_a90
```

The current Hub tree contains `llama3-8B/llama_a19`, `llama_a32`, `llama_a50`,
and `qwen3-8B-base/qwen_a90`. The same downloader accepts future Qwen folders
such as `qwen_a150` and `qwen_a250` after they are uploaded.

## Installation

DynaKV training and Triton kernels require Linux and an NVIDIA GPU. Create the
reproducible environment with:

```bash
conda env create -f environment.yml
conda activate dynakv
```

For data preparation or CPU-side tests only:

```bash
python -m pip install -e .[dev]
```

Llama 3 is gated on Hugging Face. Accept Meta's license and run `hf auth login`
before downloading it.

## Reproduce training

### 1. Download data

```bash
python scripts/prepare_data.py --dataset redpajama
python scripts/prepare_data.py --dataset wikitext2
```

The original `togethercomputer/RedPajama-Data-1T-Sample` Hub repository is no
longer available. The downloader therefore pins a 3.0 GB parquet backup of the
same V1 sample (`liang2kl/RedPajama-Data-1T-Sample-Backup`, commit
`c2c1504d...`) and saves it to `data/redpajama`. WikiText-2 raw is saved to
`data/wikitext2`. Both directories are ignored by Git; downloads use the
Hugging Face cache and can resume.
Nothing is downloaded during installation or import; users must run these
commands explicitly. To avoid the 3 GB RedPajama download when only evaluating,
run only the WikiText-2 command.

### 2. Compute PCA initialization

Run once for each base model. The default follows the paper's 10,000-token
WikiText calibration setting and collects pre-RoPE K/V projections.

```bash
python scripts/pca_initialization.py \
  --model meta-llama/Meta-Llama-3-8B \
  --output-dir pca_artifacts/llama3-8b

python scripts/pca_initialization.py \
  --model Qwen/Qwen3-8B-Base \
  --output-dir pca_artifacts/qwen3-8b-base
```

### 3. Build initialized checkpoints

```bash
python scripts/modify_model.py \
  --model meta-llama/Meta-Llama-3-8B \
  --model-type llama \
  --pca-dir pca_artifacts/llama3-8b \
  --output-dir outputs/llama3-8b-initialized

python scripts/modify_model.py \
  --model Qwen/Qwen3-8B-Base \
  --model-type qwen \
  --pca-dir pca_artifacts/qwen3-8b-base \
  --output-dir outputs/qwen3-8b-initialized
```

### 4. Train

```bash
bash train_llama.sh outputs/llama3-8b-initialized 19
bash train_qwen.sh outputs/qwen3-8b-initialized 90
```

The wrappers reproduce the separate reference schedules while keeping the total
training volume identical:

| Backbone | Context | Steps | Micro-batch/GPU | Accumulation | 1-GPU tokens |
|---|---:|---:|---:|---:|---:|
| Llama 3 8B | 8,192 | 500 | 4 | 8 | 131,072,000 |
| Qwen3 8B Base | 32,768 | 250 | 1 | 16 | 131,072,000 |

The reference runs use one NVIDIA H200 GPU, matching the paper's approximately
128M-token post-training budget. Override `NUM_GPUS` only together with the step
or batch settings if exact training-volume comparability matters.

## Evaluation

Perplexity:

```bash
python scripts/eval_ppl.py --model PATH --model-type llama
```

Actual hard-mask retention and KV compression:

```bash
python scripts/measure_compression.py --model PATH --model-type llama
```

Throughput (defaults to batch sizes 1, 2, and 4; 100 generated tokens; mean of
three measured runs):

```bash
python scripts/benchmark_throughput.py --model PATH --model-type llama
```

## Physical cache compression and kernel routing

`src/dynakv/kernels` contains one physical variable-length cache and both CUDA
attention paths supplied with the project:

- `fused.py`: one fused decode kernel for cache lengths below 500 tokens;
- `two_stage.py`: a QK/softmax/V path for cache lengths of 500 tokens or more.

`KernelPolicy` chooses the path from the current cache length at every decode
step. Change the boundary with `KernelPolicy(switch_tokens=...)`. The compact
research model in `custom_llama.py` and `custom_qwen3.py` is the differentiable
training/evaluation implementation; the Triton directory is the physical-storage
inference implementation. The current fused kernels implement the Llama attention
math. Qwen3 has Q/K RMSNorm, so it must not silently reuse those kernels until
that normalization is fused and numerically validated.

## Repository layout

```text
configs/                 DeepSpeed configurations for each backbone
scripts/                 data, PCA, conversion, training, and evaluation CLIs
src/dynakv/              model code and dynamic compressed-cache kernels
tests/                   fast CPU-side regression tests
```

## Notes

- The custom model classes target `transformers==4.53.x`; internal Transformers
  APIs are version-sensitive.
- The released training runs were performed on a single **NVIDIA H200** GPU. The
  physical-cache inference kernels and throughput measurements were developed
  and evaluated on an **NVIDIA GeForce RTX 4090**. Report the GPU, CUDA, PyTorch,
  prompt length, batch size, and generation length with reproduced results;
  throughput and peak memory are not directly comparable across GPU families.
- PCA artifacts, datasets, and checkpoints are intentionally excluded from Git.
- Run `python -m unittest discover -s tests` before publishing changes.

## Citation

```bibtex
@inproceedings{lu2026dynakv,
  title     = {One Size Does Not Fit All: Token-Wise Adaptive Compression for KV Cache},
  author    = {Lu, Liming and Qiu, Kaixi and Zhou, Jiayu and Kai, Jushi and others},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```
