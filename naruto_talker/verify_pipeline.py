import sys
import os

print("Verifying imports...")
try:
    import torch
    print(f"Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    
    import diffusers
    print(f"Diffusers: {diffusers.__version__}")
    
    from src.t2i import T2IPipeline
    print("T2IPipeline imported.")
    
    from src.i2v import I2VPipeline
    print("I2VPipeline imported.")
    
    from src.tts import TTSPipeline
    print("TTSPipeline imported.")
    
    from src.lipsync import LipSyncPipeline
    print("LipSyncPipeline imported.")
    
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
