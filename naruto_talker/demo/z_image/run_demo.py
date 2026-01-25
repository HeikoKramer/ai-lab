import os
import sys
import time
import subprocess
import webbrowser

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# Import torch explicitly to fix the "name 'torch' is not defined" error
import torch
import cv2
from PIL import Image
import imageio

from src.t2i import T2IPipeline
from src.i2v import I2VPipeline

def main():
    print("=== Z-Image Demo: Video Generation (No Audio) ===")
    
    # Defaults
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    base_prompt = "portrait close-up of shōnen ninja anime character, centered face, detailed anime style, semi-realistic, soft lighting, high quality, blonde spiky hair, blue eyes, whisker marks on cheeks"
    
    MAX_RETRIES = 10
    
    from src.verification import VerificationPipeline

    # Initialize Verification Pipeline ONCE if possible to save time?
    # No, we need VRAM for T2I/I2V. So we must load/unload verification too?
    # FaceAlignment is relatively small compared to SDXL/LTX. Let's try to keep it logic clean.
    # Actually, let's load verification on demand or keep it if it fits. 
    # SFD is ~100MB? It's small.
    
    # NOTE: We'll re-instantiate Verification inside the loop or keeps it lightweight.
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n##########################################")
        print(f"### ATTEMPT {attempt}/{MAX_RETRIES}")
        print(f"##########################################")
        
        current_seed = 42 + (attempt * 333) # Deterministic but changing seed
        print(f"Seed: {current_seed}")

        # --- Step 1: Text-to-Image ---
        print("\n--- Step 1: Text-to-Image ---")
        t2i_pipe = T2IPipeline()
        try:
            # Force 512x512 for maximum stability with LTX-Video
            portrait_img = t2i_pipe.generate(base_prompt, seed=current_seed, width=512, height=512)
            portrait_path = os.path.join(output_dir, f"portrait_attempt_{attempt}.png")
            portrait_img.save(portrait_path)
            print(f"Portrait saved to: {portrait_path}")
        except Exception as e:
            print(f"T2I Generation failed: {e}")
            t2i_pipe.unload()
            continue
            
        t2i_pipe.unload()
        del t2i_pipe
        torch.cuda.empty_cache()
        
        # --- Step 2: Face Verification ---
        print("\n--- Step 2: Face Verification ---")
        verifier = VerificationPipeline()
        if not verifier.verify_image(portrait_path):
             print(f"FAILURE: Generated portrait (Attempt {attempt}) does not have a detectable face. Retrying...")
             verifier.close()
             del verifier
             continue
        
        print("Portrait verified.")
        
        # --- Step 3: Image-to-Video ---
        print("\n--- Step 3: Image-to-Video ---")
        i2v_pipe = I2VPipeline()
        
        try:
             # Reload clean image
             portrait_img_clean = Image.open(portrait_path)
             # Generate
             frames = i2v_pipe.generate(portrait_img_clean, seed=current_seed, num_frames=25)
        except Exception as e:
             print(f"I2V Generation error: {e}")
             i2v_pipe.unload()
             verifier.close()
             continue
             
        i2v_pipe.unload()
        del i2v_pipe
        torch.cuda.empty_cache()
        
        # --- Step 4: Video Verification ---
        print("\n--- Step 4: Video Verification ---")
        # Reuse existing verifier
        if not verifier.verify_video(frames, sample_rate=3, success_threshold=0.7):
             print(f"FAILURE: Generated video (Attempt {attempt}) failed verification (faces lost/distorted). Retrying...")
             verifier.close()
             del verifier
             continue
             
        verifier.close()
        del verifier
        
        # --- Success ---
        print("\n>>> SUCCESS! Generation Passed All Checks. <<<")
        final_video_path = os.path.join(output_dir, "final.mp4")
        imageio.mimsave(final_video_path, frames, fps=25)
        print(f"Final confirmed video saved to: {final_video_path}")
        
        print(f"Opening {final_video_path}...")
        try:
            webbrowser.open(f"file://{os.path.abspath(final_video_path)}")
        except:
            pass
        return # Exit main on success

    print("\nXXX ERROR: Max retries reached. Could not generate a valid video. XXX")

if __name__ == "__main__":
    main()
