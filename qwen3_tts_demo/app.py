# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
import os
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login

# Optional flash-attn support
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

print(f"Flash Attention available: {HAS_FLASH_ATTN}")
print(f"CUDA available: {torch.cuda.is_available()}")

HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    try:
        return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        print(f"Loading model {model_type} {model_size}...")
        model_path = get_model_path(model_type, model_size)
        
        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        # Determine attention implementation
        attn_impl = "flash_attention_2" if (HAS_FLASH_ATTN and device == "cuda") else "eager"
        print(f"Using device: {device}, dtype: {dtype}, attn: {attn_impl}")
        
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            token=HF_TOKEN,
            attn_implementation=attn_impl,
        )
        print(f"Model loaded successfully.")
    return loaded_models[key]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B)."""
    print(f"Generating Voice Design: {text[:20]}...")
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        # MVP uses 1.7B VoiceDesign as per Original Space defaults
        # But maybe we should allow 0.6B if available? The Space hardcoded 1.7B for this function.
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice design generation completed successfully!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {type(e).__name__}: {e}"

# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Demo (Local)
"""
        )

        with gr.Tabs():
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column():
                        vd_text = gr.Textbox(label="Text to Speak", lines=3, placeholder="Enter text here...")
                        vd_language = gr.Dropdown(choices=LANGUAGES, value="Auto", label="Language")
                        vd_description = gr.Textbox(label="Voice Description", lines=2, placeholder="E.g. A cheerful young female voice...")
                        vd_button = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        vd_output = gr.Audio(label="Generated Audio")
                        vd_status = gr.Textbox(label="Status", interactive=False)
                
                vd_button.click(
                    generate_voice_design,
                    inputs=[vd_text, vd_language, vd_description],
                    outputs=[vd_output, vd_status]
                )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
