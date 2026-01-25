
import sys
import os
sys.path.append(os.path.dirname(__file__))

from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download
import torch

def check_langs():
    model_type = "VoiceDesign"
    model_size = "1.7B"
    print(f"Loading {model_type} {model_size} config...")
    
    # We don't need to load the whole model to check config if we are clever, 
    # but Qwen3TTSModel.from_pretrained loads everything.
    # Let's try to just load the config or processor if possible.
    # Actually, the supported languages might be in the processor or model config.
    
    model_path = snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    
    # Try loading processor only
    from transformers import AutoProcessor, AutoConfig
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor, Qwen3TTSForConditionalGeneration
    
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Check if processor has language map
    print("Processor keys:", processor.__dict__.keys())
    
    # The model (Qwen3TTSForConditionalGeneration) implements get_supported_languages
    # We might need to instantiate it. But maybe we can read it from config?
    
    print("Loading Model...")
    # Load on CPU to save VRAM and just for inspection
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="cpu", 
        torch_dtype=torch.float32
    )
    
    langs = model.get_supported_languages()
    print("Supported Languages:", langs)

if __name__ == "__main__":
    check_langs()
