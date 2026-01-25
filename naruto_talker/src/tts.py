import torch
import numpy as np
import sys
import os

# Ensure qwen_tts is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



class TTSPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.current_model_type = None
        self.model = None
        self.model_size = "1.7B"
        
    def load_model(self, target_type="VoiceDesign"):
        if self.model is not None and self.current_model_type == target_type:
            return

        # Unload if different model is needed
        if self.model is not None:
            self.unload()

        from qwen_tts import Qwen3TTSModel
        from huggingface_hub import snapshot_download
        
        print(f"Loading TTS model: {target_type} {self.model_size}...")
        try:
            # HuggingFace repo naming convention for Qwen3-TTS
            # VoiceDesign -> Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
            # Base (Cloning) -> Qwen/Qwen3-TTS-12Hz-1.7B-Base
            repo_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-{target_type}"
            model_path = snapshot_download(repo_id)
            
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="sdpa" if self.device=="cuda" else "eager"
            )
            self.current_model_type = target_type
            print(f"TTS model ({target_type}) loaded.")
        except Exception as e:
            print(f"Failed to load TTS model {target_type}: {e}")
            raise e

    def unload(self):
        if self.model is not None:
             print(f"Unloading TTS model ({self.current_model_type})...")
             del self.model
             self.model = None
             self.current_model_type = None
             import gc
             gc.collect()
             torch.cuda.empty_cache()
             print("TTS model unloaded.")

    def generate(self, text, voice_description=None, ref_audio_path=None, language="auto", speed=1.0):
        """
        Generates audio using either Voice Design (prompt) or Voice Cloning (audio).
        
        Args:
            text (str): The speech text.
            voice_description (str): Text description for Voice Design.
            ref_audio_path (str): Path to reference audio for Voice Cloning.
            language (str): Output language.
            speed (float): Speed modifier (currently implemented via prompt hints).
        """
        
        # Dispatch logic
        if ref_audio_path and os.path.exists(ref_audio_path):
            # Mode: Voice Cloning
            
            # 1. sanitize audio input (convert to WAV 16-bit 44.1kHz mono/stereo)
            # This fixes librosa/soundfile errors with m4a/mp3 etc.
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            
            base_name = os.path.splitext(os.path.basename(ref_audio_path))[0]
            clean_wav_path = os.path.join(os.path.dirname(ref_audio_path), f"{base_name}_clean.wav")
            
            # Only convert if not already wav or if we just want to be safe
            print(f"Sanitizing reference audio: {ref_audio_path} -> {clean_wav_path}")
            # -y override, -vn no video, -acodec pcm_s16le, -ar 44100
            os.system(f"{ffmpeg_path} -y -v warning -i \"{ref_audio_path}\" -vn -acodec pcm_s16le -ar 44100 \"{clean_wav_path}\"")
            
            target_path = clean_wav_path if os.path.exists(clean_wav_path) else ref_audio_path

            target_type = "Base"
            self.load_model(target_type)
            print(f"Generating TTS (Clone Mode) for: {text[:20]}... (Lang: {language})")
            
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=target_path,
                # ref_text could be used for ICL if we had the transcript of the ref audio, 
                # but for simplicity we rely on x_vector or internal ICL if implied?
                # The wrapper says: if x_vector_only_mode=False (default), ref_text is REQUIRED.
                # If we don't have ref_text, we MUST use x_vector_only_mode=True.
                x_vector_only_mode=True
            )
            
        else:
            # Mode: Voice Design
            target_type = "VoiceDesign"
            self.load_model(target_type)
            print(f"Generating TTS (Design Mode) for: {text[:20]}... (Lang: {language})")
            
            # Speed handling via prompt injection
            modified_desc = voice_description if voice_description else "A neutral voice"
            if speed < 0.8:
                modified_desc += ", speaking slowly"
            elif speed > 1.2:
                modified_desc += ", speaking quickly"
                
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=modified_desc,
                non_streaming_mode=True,
            )

        return sr, wavs[0]
