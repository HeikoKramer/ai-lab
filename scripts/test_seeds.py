#!/usr/bin/env python3
import os
import time
import subprocess
import csv
from diffusers import AutoPipelineForText2Image
import torch
from datetime import datetime

# === Configuration ===
model_id = "stabilityai/sdxl-turbo"
character_base = "a detailed anime girl with long silver hair, blue eyes, wearing a white futuristic jacket, medium shot, clean background"
emotions = [
    "happy", "sad", "angry", "surprised", "confused",
    "scared", "excited", "bored", "sleepy", "crying",
    "smiling", "blushing", "embarrassed", "determined", "laughing",
    "nervous", "shocked", "serious", "thinking", "shy"
]
steps = 6
cfg = 1.0

# === Output paths ===
project_root = os.getenv("AI_LAB_ROOT", "/mnt/e/Projects/ai-lab")
output_dir = os.path.join(project_root, "outputs", "anime_emotions")
os.makedirs(output_dir, exist_ok=True)

# Also copy results to Windows Downloads
win_downloads = "/mnt/c/Users/%s/Downloads/ai-lab_anime_emotions" % os.getenv("USER", "Heiko")
os.makedirs(win_downloads, exist_ok=True)

# === Device setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load pipeline ===
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

# === CSV logging ===
csv_path = os.path.join(output_dir, f"anime_emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_fields = ["index", "emotion", "duration_sec", "vram_used_MB", "output_file"]

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_fields)

    total_start = time.time()

    for i, emotion in enumerate(emotions):
        full_prompt = f"{character_base}, expressing {emotion} emotion, consistent face and outfit"
        torch.manual_seed(1234)  # keep same base appearance
        start = time.time()
        image = pipe(full_prompt, num_inference_steps=steps, guidance_scale=cfg).images[0]
        duration = time.time() - start

        # VRAM usage
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            vram_used = int(result.stdout.strip().split("\n")[0])
        except Exception:
            vram_used = -1

        out_name = f"{i+1:02d}_{emotion}.png"
        out_path = os.path.join(output_dir, out_name)
        image.save(out_path)

        # Copy to Windows downloads
        subprocess.run(["cp", out_path, win_downloads], check=False)

        print(f"Image {i+1}/{len(emotions)} | {emotion} | Time: {duration:.2f}s | VRAM: {vram_used} MB")
        writer.writerow([i+1, emotion, round(duration, 2), vram_used, out_path])
        csvfile.flush()

    total_time = time.time() - total_start
    print(f"\nDone. {len(emotions)} images generated in {total_time:.2f}s total.")
    print(f"CSV log saved to: {csv_path}")
    print(f"Copies in: {win_downloads}")
