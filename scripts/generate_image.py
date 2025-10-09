import argparse, os, re, time
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

def parse_size(s):
    m = re.match(r"^(\d+)[xX](\d+)$", s)
    if not m: raise ValueError("Size must look like 1024x1024")
    w, h = int(m.group(1)), int(m.group(2))
    # SD/SDXL prefer multiples of 8/32; we round to multiple of 8 to be safe
    w = (w // 8) * 8; h = (h // 8) * 8
    return w, h

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id, e.g. stabilityai/sdxl-turbo")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--out", default="out.png")
    p.add_argument("--steps", type=int, default=None, help="Inference steps")
    p.add_argument("--cfg", type=float, default=None, help="Guidance scale")
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=["auto","fp16","bf16"], default="fp16")
    p.add_argument("--scheduler", choices=["auto","dpm","euler_a"], default="auto")
    p.add_argument("--attn-slicing", action="store_true", help="Enable attention slicing for memory efficiency")
    p.add_argument("--offload", action="store_true", help="Enable model CPU offload to save VRAM")
    args = p.parse_args()

    # Sensible defaults per model
    is_turbo = "turbo" in args.model.lower()
    steps = args.steps if args.steps is not None else (4 if is_turbo else 30)
    cfg   = args.cfg   if args.cfg   is not None else (0.0 if is_turbo else 7.5)

    w, h = parse_size(args.size)

    # torch dtype selection
    if args.dtype == "fp16":
        torch_dtype = torch.float16
        variant = "fp16"
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
        variant = None  # most repos publish fp16; bf16 works via dtype override
    else:
        torch_dtype = None
        variant = None

    print("=== Environment check ===")
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
    print()

    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=True
    )
    # Scheduler
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
    elif args.scheduler == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # VRAM tactics
    if args.offload:
        pipe.enable_model_cpu_offload()   # accelerate offload
    else:
        pipe.to("cuda")
    if args.attn_slicing or args.offload:
        pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    load_s = time.time() - t0
    print(f"Loaded pipeline in {load_s:.2f}s")

    # Deterministic seed (optional)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

    # Small perf tweak on modern GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    print(f"Generating: {w}x{h}, steps={steps}, cfg={cfg}, seed={args.seed}")
    t1 = time.time()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=w, height=h,
        generator=generator
    )
    img = result.images[0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)

    gen_s = time.time() - t1
    total  = time.time() - t0
    print(f"Saved -> {args.out}")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Max GPU memory: {mem:.2f} GB")
    print(f"Time: load {load_s:.2f}s | gen {gen_s:.2f}s | total {total:.2f}s")

if __name__ == "__main__":
    main()
