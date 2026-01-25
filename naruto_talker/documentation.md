# Technical Documentation: Naruto Talker

## Overview
Naruto Talker is an AI-powered pipeline that generates animated videos of a character speaking a given text. It integrates multiple state-of-the-art models to achieve text-to-image generation, text-to-speech synthesis, and lip-syncing.

## Architecture
The pipeline consists of four main stages:

```mermaid
graph TD
    A[User Input] -->|Prompts & Text| B[Text-to-Image (T2I)]
    A -->|Text| C[Text-to-Speech (TTS)]
    B -->|Portrait Image| D[Lip-Syncing]
    C -->|Audio| D
    D -->|Raw Video| E[Face Restoration]
    E -->|Final Video| F[Output]
```

### 1. Text-to-Image (T2I)
- **Model**: [SDXL (Stable Diffusion XL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) via `diffusers` library.
- **Function**: Generates a high-quality portrait of the character based on a text description.
- **Configuration**:
  - `guidance_scale`: 7.5
  - `num_inference_steps`: 30
  - Negative prompting used to avoid low-quality results.

### 2. Text-to-Speech (TTS)
- **Model**: [Qwen2-Audio (Instruct)](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) / Qwen-TTS.
- **Function**: Converts the input text into a speech audio file (`.wav`).
- **Features**: 
  - Voice design capability (cloning or description-based generation).
  - Multi-language support.

### 3. Lip-Syncing
- **Model**: [MuseTalk](https://github.com/TMElyralab/MuseTalk).
- **Function**: Animates the generated portrait to sync lip movements with the generated audio.
- **Process**:
  - Detects the face in the source image.
  - Modifies the mouth region frame-by-frame to match the audio phonemes.
- **Key Parameters**:
  - `bbox_shift`: Adjusts the bounding box of the face detection (-10 recommended).

### 4. Face Restoration
- **Model**: [GFPGAN](https://github.com/TencentARC/GFPGAN).
- **Function**: Enhances the visual quality of the face in the generated video, removing artifacts and improving sharpness.
- **Configuration**:
  - `upscale`: 2 (2x upscaling).
  - `bg_upsampler`: None (face only).

## File Structure
- `src/t2i.py`: Text-to-Image generation logic.
- `src/tts.py`: Text-to-Speech interface.
- `src/lipsync.py`: MuseTalk integration.
- `src/restoration.py`: GFPGAN face restoration.
- `app.py`: Main entry point and Gradio UI.
- `run_full_pipeline.py`: Headless execution script for testing.
