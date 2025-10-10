# Emotion Seed Batch Generator (`scripts/test_seeds.py`)

The `scripts/test_seeds.py` workflow automates Hugging Face Diffusers batches for
an anime-style character while cycling through a curated list of emotions. It is
designed for reproducible face consistency across generations, quick VRAM
profiling, and mirrored export to both the WSL project tree and a Windows
Downloads folder.

---

## Quick Start

```bash
python scripts/test_seeds.py
```

The script requires GPU-enabled PyTorch with Diffusers installed. By default it
writes PNG files to `$AI_LAB_ROOT/outputs/anime_emotions` and copies the same
images to `C:/Users/<you>/Downloads/ai-lab_anime_emotions` (through the WSL
mount).

---

## Execution Flow

```
+--------------------------+    +-------------------------+    +----------------------------+
|    Resolve file paths    | -> |    Load SDXL-Turbo      | -> |  Iterate over emotion list |
+--------------------------+    +-------------------------+    +----------------------------+
                                                                  |
                                                                  v
                                +-------------------------+    +---------------------------+
                                |  Seed & run inference   | -> |  Log CSV, copy PNG assets |
                                +-------------------------+    +---------------------------+
                                                                  |
                                                                  v
                                                          +-----------------------+
                                                          |  Print run summary    |
                                                          +-----------------------+
```

---

## Configuration Cheat Sheet

| Setting | Location | Purpose |
| --- | --- | --- |
| `model_id` | Top of file | Hugging Face pipeline to load (`stabilityai/sdxl-turbo`). |
| `character_base` | Top of file | Prompt foundation shared by every generation. |
| `emotions` | Top of file | Ordered list of emotion modifiers appended per run. |
| `steps` | Top of file | Diffusion steps (defaults to 6 for SDXL Turbo speed). |
| `cfg` | Top of file | Guidance scale (1.0 keeps Turbo coherence). |
| `project_root` | Derived from `AI_LAB_ROOT` | Output base directory inside the repo. |
| `win_downloads` | Derived from `$USER` | Mirror path under Windows Downloads. |

Modify these constants directly in the script to customize prompts, destinations,
or sampling parameters.

---

## What happens during a run

1. **Device discovery** – Chooses CUDA when available, otherwise falls back to CPU.
2. **Pipeline setup** – Downloads (or reuses cached) SDXL Turbo weights in FP16 and
   moves the pipeline to the selected device.
3. **Emotion loop** – For each entry in `emotions`:
   - Resets `torch.manual_seed(1234)` to preserve character identity.
   - Combines the base prompt with the emotion descriptor.
   - Generates an image with the configured steps and guidance scale.
   - Polls `nvidia-smi` for current VRAM usage (records `-1` when unavailable).
   - Saves the PNG under `outputs/anime_emotions` and copies it to the Windows folder.
   - Appends a CSV row containing index, emotion, duration, VRAM, and file path.
4. **Summary** – Prints total runtime and the locations of the image set and CSV log.

If any subprocess copy fails, the script continues so that the primary PNG export
and CSV log are preserved.

---

## CSV log structure

The script emits a timestamped CSV named like
`anime_emotions_YYYYMMDD_HHMMSS.csv` in the output directory.

| Column | Meaning |
| --- | --- |
| `index` | 1-based counter matching the emotion order. |
| `emotion` | Prompt suffix used for the frame. |
| `duration_sec` | Generation time rounded to two decimals. |
| `vram_used_MB` | Latest `nvidia-smi` reading or `-1` when unavailable. |
| `output_file` | Absolute path to the saved PNG in WSL. |

Use this log to benchmark prompt variants or copy results into spreadsheets.

---

## Customization tips

- **Change the character look** by editing `character_base` with new descriptors
  (e.g., outfit, art style, lighting) while keeping the emotion list intact.
- **Swap the backbone model** by updating `model_id` to any compatible
  Diffusers text-to-image pipeline, adjusting `steps`/`cfg` for non-Turbo models.
- **Trim or extend emotions** by editing the list; order controls the PNG prefixes.
- **Disable Windows mirroring** by commenting out or removing the `subprocess.run`
  copy block if not using WSL/Windows dual paths.
- **Log extra metadata** by adding columns to `csv_fields` and extending the row
  writer with your custom measurements (e.g., seed per emotion).

---

## Related files

- [`scripts/test_seeds.py`](../scripts/test_seeds.py) — Implementation source.
- [`docs/generate_image.md`](generate_image.md) — Companion guide for single
  prompt text-to-image runs.
- [`docs/test_model.md`](test_model.md) — Text-generation smoke test reference.
