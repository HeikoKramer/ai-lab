# AI Lab Model Smoke Test (`scripts/test_model.py`)

The `scripts/test_model.py` helper verifies that the AI Lab workstation can run
GPU-backed Hugging Face text generation workloads with the current PyTorch
nightly build. It prints a compact environment report, loads the requested
model, generates a short sample, and surfaces timing/memory information so you
can quickly confirm that the stack (CUDA → PyTorch → Transformers) is wired up
correctly.

---

## Features & How to Use

| Purpose | Command |
| --- | --- |
| Default quick test (auto dtype) | `python scripts/test_model.py` |
| Half precision (fast, less VRAM) | `python scripts/test_model.py --dtype float16` |
| Offline mode (cached models only) | `python scripts/test_model.py --offline` |
| Custom model + prompt | `python scripts/test_model.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 50 --prompt "Summarize this paragraph:"` |

> 💡 The `tmod` shell helper wraps these commands and automatically uses the AI
> Lab virtual environment interpreter (see [Shortcuts](#shortcuts--automation)).

---

## What the script does (quick mental model)

1. **GPU & PyTorch**
   - Imports PyTorch and prints the version + compiled CUDA runtime.
   - Calls `torch.cuda.is_available()` and reports the active GPU name, compute
     capability, and supported SM arch list.
2. **Hugging Face model loading**
   - Downloads (or reuses cached) tokenizer + model weights via
     `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`.
   - Moves the model to the selected device, respecting your precision request
     (`--dtype`).
3. **Text generation**
   - Builds a `transformers.pipeline("text-generation")` with the resolved
     model/tokenizer pair.
   - Generates a short sample based on your prompt and reports generation time
     plus GPU memory usage.

If any layer fails, the script exits with a non-zero status so CI or shell
aliases can detect the error.

---

## Step-by-step basic usage

Assumptions: your project root is `$AI_LAB_ROOT` (defaults to `/mnt/e/Projects/ai-lab`) and
your virtual environment lives at `$AI_LAB_ROOT/.venv`.

1. **Activate the virtual environment (optional if already active).**
   ```bash
   source "$AI_LAB_ROOT/.venv/bin/activate"
   ```
2. **Run the default smoke test (GPT-2, auto precision, 30 new tokens).**
   ```bash
   python scripts/test_model.py
   ```
3. **Try common variations as needed.**
   - Less VRAM on modern NVIDIA: `python scripts/test_model.py --dtype float16`
   - Use BF16 on Blackwell GPUs: `python scripts/test_model.py --dtype bfloat16`
   - Force a specific cached model: `python scripts/test_model.py --model ai-lab/your-model`
4. **Run offline after caching models.**
   ```bash
   python scripts/test_model.py --offline
   ```
5. **Temporarily pin to CPU-only execution** (only when debugging; much slower).
   ```bash
   CUDA_VISIBLE_DEVICES= python scripts/test_model.py
   ```

Each invocation prints an environment banner, loading timing, generation timing,
and the resulting text. Watch for download errors, CUDA availability warnings,
or unusually slow throughput.

---

## Which parameters can you change?

All runtime tuning happens via CLI flags—no source edits required:

- `--model`: Hugging Face repo or local path (`openai-community/gpt2` by default).
- `--dtype`: Precision mode (`auto`, `float16`, `bfloat16`, `float32`).
- `--device`: CUDA device index (0 by default).
- `--prompt`: Input text to seed the generation.
- `--max-new-tokens`: Number of tokens to generate (30 by default).
- `--temperature`: Sampling temperature (higher = more randomness).
- `--top-p`: Top-p nucleus sampling cutoff.
- `--offline`: Toggle Hugging Face offline mode (`HF_HUB_OFFLINE=1`).

Advanced users can edit defaults directly inside `parse_args()` if persistent
changes are needed.

---

## Prerequisites

- Python 3.9+
- PyTorch nightly with CUDA 12.8 support (downloaded via the NVIDIA index).
- `transformers`, `accelerate`, and `sentencepiece`.
- Access to the Hugging Face Hub (or cached models when `--offline`).

When the dependencies are missing, the script prints installation hints and
exits with status code 1 before attempting GPU work.

---

## Output anatomy

- **Environment check:** Torch version, CUDA runtime, GPU capability, and SM
  arch list.
- **Model loading:** Duration and post-load GPU memory consumption.
- **Generation:** Pipeline configuration, elapsed time, and generated text.
- **Completion marker:** `Test complete.` for successful runs.

---

## Shortcuts & automation

- `tmod` (WSL/Linux): Runs `scripts/test_model.py` with the virtualenv's Python
  interpreter, forwarding any additional CLI args.
- `tmod` (Windows PowerShell): Currently prints a reminder that the helper is
  only available under WSL.

The helper function relies on the `AI_LAB_ROOT` environment variable, which is
exported automatically by the dotfiles and can be overridden before sourcing
`~/.bashrc` or the Windows PowerShell profile.

---

## Troubleshooting tips

- **`torch.cuda.is_available()` is `False`:** Verify that the NVIDIA driver and
  CUDA runtime match the PyTorch build; ensure you are running inside WSL with a
  GPU-capable kernel.
- **`HF_HUB_OFFLINE` issues:** If downloads fail, remove the `--offline` flag or
  run `huggingface-cli login` to refresh credentials.
- **OOM errors:** Reduce `--max-new-tokens`, switch to `--dtype float16`, or
  choose a smaller model.
- **Slow generation:** Check that the pipeline is using the GPU (`device=0`) and
  that no conflicting `CUDA_VISIBLE_DEVICES` settings are applied.

---

## Related files

- [`scripts/test_model.py`](../scripts/test_model.py): Implementation.
- [`dot_bashrc`](../dot_bashrc): Exports `AI_LAB_ROOT` and loads the `tmod`
  helper on WSL/Linux.
- [`readonly_Documents/PowerShell/Microsoft.PowerShell_profile.ps1.tmpl`](../readonly_Documents/PowerShell/Microsoft.PowerShell_profile.ps1.tmpl):
  Provides the Windows placeholder implementation of `tmod`.
