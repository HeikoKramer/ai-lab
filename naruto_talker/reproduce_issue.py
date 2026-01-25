import torch
from src.t2i import T2IPipeline
from src.i2v import I2VPipeline
from diffusers.utils import export_to_video
import os

def check_outputs():
    os.makedirs("debug_outputs", exist_ok=True)
    
    # 1. T2I Generation
    print("Step 1: Testing T2I Generation...")
    t2i = T2IPipeline()
    
    # Simulate app.py logic roughly
    visual_desc = "blonde spiky hair, blue eyes, whisker marks on cheeks"
    # app.py appends emotions
    visual_desc += ", serious expression, determined look"
    
    # generate(self, character_desc, seed=None, width=832, height=1216, steps=28, cfg=7.0)
    image = t2i.generate(
        character_desc=visual_desc,
        seed=42
    )
    image_path = "debug_outputs/t2i_result.png"
    image.save(image_path)
    print(f"Saved T2I result to {image_path}")
    print(f"Image size: {image.size}")
    
    # 2. I2V Generation
    print("Step 2: Testing I2V Generation...")
    i2v = I2VPipeline()
    
    frames = i2v.generate(
        input_image=image,
        seed=42,
        num_frames=75 # User log said 75 frames (generated silent video 75 frames)
    )
    
    video_path = "debug_outputs/i2v_result.mp4"
    export_to_video(frames, video_path, fps=24)
    print(f"Saved I2V result to {video_path}")
    
    # Check resolution
    print(f"Video Frame Size: {frames[0].size}")

if __name__ == "__main__":
    check_outputs()
