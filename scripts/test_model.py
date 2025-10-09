#!/usr/bin/env python3
"""Quick GPU + Hugging Face smoke test for the AI Lab environment."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable


DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "test_model.md"
DOC_REFERENCE = DOC_PATH.relative_to(Path(__file__).resolve().parent.parent)

ESSENTIAL_OVERVIEW = textwrap.dedent(
    f"""
    ------------------------------------------------------------
    Quick reference (see {DOC_REFERENCE} for the full walkthrough)

    Features & How to Use:
      • Default smoke test: python scripts/test_model.py
      • Half precision:    python scripts/test_model.py --dtype float16
      • Offline mode:      python scripts/test_model.py --offline
      • Custom prompt:     python scripts/test_model.py --prompt "Write a haiku"

    What the script does (quick mental model):
      1. Checks PyTorch + CUDA availability and reports GPU details.
      2. Loads a Hugging Face tokenizer/model pair onto your selected device.
      3. Generates text with the Transformers text-generation pipeline and
         reports timing + GPU memory usage.

    Step-by-step basic usage:
      1. Activate the AI Lab virtualenv (source "$AI_LAB_ROOT/.venv/bin/activate").
      2. Run the default smoke test or add flags as needed (dtype, prompt, etc.).
      3. Review the printed environment + generation report for issues.

    Which parameters can you change?
      --model, --dtype, --device, --prompt, --max-new-tokens,
      --temperature, --top-p, --offline
    ------------------------------------------------------------
    """
)


class ExtendedHelpParser(argparse.ArgumentParser):
    """ArgumentParser that appends an essential overview to the help text."""

    def format_help(self) -> str:  # type: ignore[override]
        base = super().format_help()
        return f"{base}\n{ESSENTIAL_OVERVIEW}"


def build_parser() -> ExtendedHelpParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = ExtendedHelpParser(
        description="Quick GPU + HF pipeline test",
        formatter_class=formatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", "-help", action="help", help="Show this help message and exit")
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="Model name or local path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Precision: auto | float16 | bfloat16 | float32",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time there was a curious AI lab,",
        help="Prompt to generate text from",
    )
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in Hugging Face offline mode (model must be cached)",
    )
    return parser


def import_dependencies():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency availability check
        message = textwrap.dedent(
            """
            Missing dependencies: {error}

            Install the required packages inside your AI Lab virtual environment, e.g.:
              pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
              pip install --upgrade transformers accelerate sentencepiece
            """
        ).strip()
        print(message.format(error=exc), file=sys.stderr)
        sys.exit(1)

    return torch, AutoModelForCausalLM, AutoTokenizer, pipeline


def resolve_dtype(requested: str, torch_module) -> tuple[str, object]:
    dtype_map = {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    key = requested.lower()
    if key not in dtype_map:
        print(f"Warning: Unknown dtype '{requested}', falling back to auto.")
        key = "auto"
    return key, dtype_map[key]


def print_environment_report(torch_module, device_index: int) -> None:
    print("\n=== Environment check ===")
    print(f"torch version: {torch_module.__version__}")
    print(f"CUDA runtime:  {torch_module.version.cuda}")
    cuda_available = torch_module.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        device_count = torch_module.cuda.device_count()
        if device_index >= device_count:
            print(
                f"Requested device index {device_index} but only {device_count} device(s) detected.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"Device: {torch_module.cuda.get_device_name(device_index)}")
        print(f"Capability: {torch_module.cuda.get_device_capability(device_index)}")
        print(f"Supported arch list: {torch_module.cuda.get_arch_list()}")


def configure_torch(torch_module) -> None:
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")
    if hasattr(torch_module, "backends") and hasattr(torch_module.backends, "cuda"):
        matmul = getattr(torch_module.backends.cuda, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = True


def load_model_and_tokenizer(args, torch_module, AutoModelForCausalLM, AutoTokenizer):
    print("\n=== Loading model & tokenizer ===")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype_key, dtype = resolve_dtype(args.dtype, torch_module)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=None if dtype == "auto" else dtype,
        device_map={"": args.device} if torch_module.cuda.is_available() else None,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    if torch_module.cuda.is_available():
        allocated = torch_module.cuda.memory_allocated(args.device) / 1024 ** 2
        print(f"GPU memory allocated after load: {allocated:.1f} MB")

    return tokenizer, model, dtype_key, dtype


def run_generation(args, torch_module, pipeline_fn, tokenizer, model, dtype_key, dtype) -> None:
    print("\n=== Generating text ===")
    generator_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
    }
    if torch_module.cuda.is_available():
        if not hasattr(model, "hf_device_map"):
            generator_kwargs["device"] = args.device
    else:
        generator_kwargs["device"] = -1
    if dtype_key != "auto":
        generator_kwargs["torch_dtype"] = dtype

    text_generator = pipeline_fn("text-generation", **generator_kwargs)

    start = time.time()
    output = text_generator(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )[0]["generated_text"]
    gen_time = time.time() - start

    print(f"\n--- Output ({gen_time:.2f}s) ---")
    print(output)
    print("\nTest complete.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    torch_module, AutoModelForCausalLM, AutoTokenizer, pipeline_fn = import_dependencies()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    configure_torch(torch_module)
    print_environment_report(torch_module, args.device)
    tokenizer, model, dtype_key, dtype = load_model_and_tokenizer(
        args, torch_module, AutoModelForCausalLM, AutoTokenizer
    )
    run_generation(args, torch_module, pipeline_fn, tokenizer, model, dtype_key, dtype)


if __name__ == "__main__":
    main()
