# `generate_image.py` Usage Guide

## Overview
`generate_image.py` is a command-line helper for running Hugging Face Diffusers text-to-image pipelines with practical defaults for RTX 5080-class hardware. It handles model loading, scheduler swaps, VRAM-saving tactics, and reproducible inference so you can focus on the prompt engineering.

## Quick Start
```bash
python scripts/generate_image.py \
  --model stabilityai/sdxl-turbo \
  --prompt "Retro-futuristic skyline at golden hour" \
  --out outputs/skylines/turbo.png
```

## Execution Flow
```
+-----------------------+    +--------------------------------+    +-------------------------------+    +------------------------------+
|   Parse CLI options   | -> |   Configure runtime defaults   | -> |   Load diffusion pipeline     | -> |   Generate image & report    |
+-----------------------+    +--------------------------------+    +-------------------------------+    +------------------------------+
                                          |                                            |
                                          v                                            v
                          +-------------------------------+          +-----------------------------------+
                          |   Enable VRAM optimizations   | -------> |   Persist image & runtime stats   |
                          +-------------------------------+          +-----------------------------------+
```

## Command Cheat Sheet
- `--model` *(required)* — Hugging Face model ID, e.g. `stabilityai/sdxl-turbo` or `runwayml/stable-diffusion-v1-5`.
- `--prompt` *(required)* — Positive text prompt describing the desired image.
- `--negative-prompt` — Optional negative prompt to steer the diffusion away from unwanted traits.
- `--out` — Output image path (directories are created automatically).
- `--steps` — Inference steps; defaults to **4** for Turbo models and **30** otherwise.
- `--cfg` — Classifier-free guidance; defaults to **0.0** for Turbo and **7.5** for standard models.
- `--size` — Target resolution such as `1024x1024`; the script rounds to multiples of 8 for safety.
- `--seed` — Integer seed for deterministic runs (matched generator device to CUDA when available).
- `--dtype` — One of `fp16`, `bf16`, or `auto`; controls model precision and associated VRAM cost.
- `--scheduler` — Choose between automatic, DPM-Solver++ (`dpm`), or Euler ancestral (`euler_a`).
- `--attn-slicing` — Enable attention slicing to reduce peak memory during inference.
- `--offload` — Activate CPU offload; pairs well with attention slicing for low-VRAM devices.

## Key Mechanics
### Environment & Hardware Report
At startup the script prints the PyTorch version, CUDA version, GPU model, and compute capability. This confirms whether GPU acceleration is active before any heavy work begins, mirroring the "Environment check" block in the sample run.

### Model Materialization
- Uses `AutoPipelineForText2Image.from_pretrained` with `use_safetensors=True` for secure weight loading.
- `variant="fp16"` is auto-selected when `--dtype fp16` so the pipeline pulls half-precision checkpoints when available.
- Scheduler swaps rely on config cloning: DPM-Solver++ with Karras noise or Euler ancestral when requested.

### VRAM Controls
- `--offload` calls `enable_model_cpu_offload`, keeping large layers on CPU until needed.
- `--attn-slicing` (or implicit activation via `--offload`) breaks attention into smaller tiles; VAE slicing is always enabled for additional savings.
- On completion, the script logs `torch.cuda.max_memory_allocated()` so you can benchmark footprint per prompt.

### Deterministic Reproducibility
Providing `--seed` locks the sampling path via a `torch.Generator` bound to the active device, enabling apples-to-apples comparisons when testing schedulers or prompts.

### Prompt Execution
During generation you will see a summary such as `Generating: 1024x1024, steps=4, cfg=0.0, seed=42`. Runtime is split into load, generation, and total elapsed seconds, helping you size cold-start vs. warm-cache behavior.

## Interpreting Sample Output
| Log Section | Meaning |
| --- | --- |
| `=== Environment check ===` | Verifies CUDA 12.8 with the RTX 5080, confirming GPU execution (no CPU fallback). |
| `model_index.json ...` | Hugging Face pulls model metadata (`model_index.json`, `config.json`, scheduler configs, tokenizer files). These cache under `~/.cache/huggingface`. |
| `text_encoder_2 ... safetensors` | Large safetensor weight files downloading; first run shows progress bars with size, ETA, and throughput. Subsequent runs load from cache almost instantly. |
| Progress meter (`3.60M/1.39G ...`) | Real-time download status: total size, estimated remaining time, and current bandwidth. |
| `Saved -> outputs/...png` | Confirms successful image write along with GPU memory usage and timing metrics. |

## Compatibility Notes
- When you rely on CPU offloading inside transformers, ensure `transformers>=4.26`; earlier versions may not recognize advanced flags like `offload_state_dict=True` and will abort before inference.
- Keep NVIDIA drivers aligned with CUDA 12.8 to retain TensorFloat-32 optimizations the script enables via `torch.backends.cuda` hints.

## Suggested Experiments
1. **Scheduler Shootout** — Hold the seed constant and compare `--scheduler auto`, `--scheduler dpm`, and `--scheduler euler_a` to evaluate detail vs. coherence.
2. **VRAM Budgeting** — Toggle `--offload` and `--attn-slicing` while watching the reported `Max GPU memory` for quantifiable savings.
3. **Resolution Sweeps** — Try `768x1344` and `1344x768` to understand aspect-ratio impacts; the rounding-to-8 safeguard ensures stable diffusion compatibility.
