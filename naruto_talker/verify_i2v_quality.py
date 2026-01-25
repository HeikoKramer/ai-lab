import sys
import os
import torch
import cv2
from src.i2v import I2VPipeline
from PIL import Image

def test_i2v():
    print("Initializing I2VWrapper...")
    try:
        wrapper = I2VPipeline()
    except Exception as e:
        print(f"Failed to init wrapper: {e}")
        return

    img_path = "/home/heiko/projects/ai-lab/naruto_talker/outputs/run_1769241170/portrait.png"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}, using dummy image")
        # create dummy
        img = Image.new('RGB', (1024, 1024), color = 'red')
        img_path = "/tmp/dummy_test.png"
        img.save(img_path)

    print(f"Using image: {img_path}")
    print("Generating video...")
    
    # wrapper.generate signature: input_image (PIL), seed, num_frames, negative_prompt
    # But wait, my code modified generate to take input_image object? 
    # Let's check line 46: width, height = input_image.size
    # So it expects a PIL image object, NOT a path.
    
    pil_img = Image.open(img_path)
    
    try:
        # returns list of PIL images (frames)
        frames = wrapper.generate(
            input_image=pil_img, 
            seed=42,
            num_frames=25
        )
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Generated {len(frames)} frames.")
    
    if len(frames) == 0:
        print("FAILURE: No frames generated.")
        return

    # Check resolution of first frame
    w, h = frames[0].size
    print(f"Resolution: {w}x{h}")
    
    # Expected: min width 512.
    # If input was 1024x1024, scale approx 0.5 -> ~512x512.
    # Actually logic: 1024 < 1024 (False). 1024 > 1024 (False).
    # scale = 1.0 (default)
    # Then width * scale = 1024. 1024 < 512 (False).
    # So it should keep 1024x1024 ideally?
    # Wait, my logic was:
    # if width > 1024 ... scale = 1024/max.
    # If input is 1024, it stays 1024?
    # Maybe 1024 is too big for VRAM?
    # The original code had scale 0.5 for everything.
    # My new logic only downscales if > 1024.
    
    if w >= 512 and h >= 512:
        print("SUCCESS: Resolution is acceptable (>= 512).")
    else:
        print("FAILURE: Resolution too small.")

    # Save to verify visually if needed (optional)
    # export_to_video(frames, "/tmp/test_output.mp4", fps=24)

if __name__ == "__main__":
    test_i2v()
