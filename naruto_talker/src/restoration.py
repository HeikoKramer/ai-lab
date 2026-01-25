import os
import torch
import cv2
import glob
import numpy as np
import imageio_ffmpeg
from gfpgan import GFPGANer

class RestorationPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.restorer = None
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
    def load_model(self):
        if self.restorer is not None:
            return

        print("Loading GFPGAN model...")
        # GFPGANer will auto-download weights to experiments/pretrained_models
        # We use v1.4 (clean version)
        self.restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            upscale=1, # No upscaling, just restoration
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )
        print("GFPGAN model loaded.")

    def unload(self):
        if self.restorer is not None:
            print("Unloading GFPGAN model...")
            del self.restorer
            self.restorer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("GFPGAN model unloaded.")

    def process_video(self, input_path, output_path):
        self.load_model()
        
        # Setup temp
        temp_dir = "/tmp/restoration"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Extract frames
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print("Restoration: Extracting frames...")
        os.system(f"{self.ffmpeg_path} -v fatal -y -i {input_path} {frames_dir}/%08d.png")
        
        # Process frames
        input_imgs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        restored_dir = os.path.join(temp_dir, "restored")
        os.makedirs(restored_dir, exist_ok=True)
        
        print(f"Restoring {len(input_imgs)} frames...")
        
        for i, img_path in enumerate(input_imgs):
            img = cv2.imread(img_path)
            
            # Restore
            # cropped_faces, restored_faces, restored_img
            _, _, restored_img = self.restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=True, # Focus on the character face
                paste_back=True,
                weight=0.2 # Very low weight to just remove blur, ignoring realism
            )
            
            if restored_img is not None:
                save_path = os.path.join(restored_dir, f"{i:08d}.png")
                cv2.imwrite(save_path, restored_img)
            else:
                print(f"Frame {i} restoration failed, using original.")
                save_path = os.path.join(restored_dir, f"{i:08d}.png")
                cv2.imwrite(save_path, img)

        # Encode video
        print("Restoration: Encoding video...")
        
        # Get Frame Rate from input? Assume 25
        # Better: Probe it, but 25 is our standard
        
        # To preserve audio, we extract audio from input and mux
        temp_vid = os.path.join(temp_dir, "temp_restored.mp4")
        os.system(f"{self.ffmpeg_path} -y -v warning -r 25 -i {restored_dir}/%08d.png -vcodec libx264 -pix_fmt yuv420p {temp_vid}")
        
        # Mux audio from original input if it exists
        if os.system(f"{self.ffmpeg_path} -v fatal -i {input_path} -vn -f null -") == 0:
             # Audio exists
             os.system(f"{self.ffmpeg_path} -y -v warning -i {input_path} -i {temp_vid} -map 1:v -map 0:a -c:v copy -c:a copy {output_path}")
        else:
             # No audio? Just move
             import shutil
             shutil.move(temp_vid, output_path)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        return output_path
