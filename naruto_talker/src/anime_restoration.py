import torch
import cv2
import os
import shutil
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class AnimeRestorationPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.restorer = None
        # We use realesr-animevideov3 (SRVGGNetCompact) which is fast and good for anime video
        self.model_name = 'realesr-animevideov3'

    def load_model(self):
        if self.restorer is not None:
            return

        print(f"Loading RealESRGAN model: {self.model_name}...")
        
        # Define model architecture for anime video v3 (compact)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        
        # Auto-download weights logic is built into RealESRGANer usually, but we need to specify path or let it fetch.
        # We'll rely on default behavior or explicit load.
        # Note: scale=4 for animevideov3
        
        # Auto-download weights logic:
        # We manually ensure weights exist before initializing RealESRGANer
        # because passing model_path=None causes a crash in some versions.

        
        # Manually load weights if RealESRGANer doesn't auto-fetch by name.
        # Actually RealESRGANer typically requires model_path.
        # Let's try to find where it is or download it.
        # For safety/speed, we can try to use the pre-trained url.
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
        model_path = os.path.join('weights', 'realesr-animevideov3.pth')
        if not os.path.exists(model_path):
             os.makedirs('weights', exist_ok=True)
             print(f"Downloading {self.model_name} weights...")
             torch.hub.download_url_to_file(file_url, model_path)
             
        self.restorer = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            half=True,
            device=self.device,
        )
        print("RealESRGAN model loaded.")

    def unload(self):
        if self.restorer is not None:
            print("Unloading RealESRGAN model...")
            del self.restorer
            self.restorer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("RealESRGAN model unloaded.")

    def process_frame(self, img_bgr):
        """
        Enhance a single cv2 image (BGR, uint8) in memory.
        Returns enhanced image (BGR, uint8).
        """
        self.load_model()
        # RealESRGANer enhance takes a numpy image.
        # outscale=2 or 4. 
        # For our use case (mouth crop 256 -> 1024), 4x is good.
        try:
             output, _ = self.restorer.enhance(img_bgr, outscale=4)
             return output
        except Exception as e:
             print(f"RealESRGAN frame enhancement failed: {e}")
             return img_bgr

    def process_video(self, input_path, output_path, prompt=None, strength=None):
        """
        prompt/strength arguments are ignored for RealESRGAN, kept for ABI compatibility.
        """
        self.load_model()
        
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        temp_dir = "/tmp/realesrgan_anime"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print("RealESRGAN: Extracting frames...")
        os.system(f"{ffmpeg_exe} -v fatal -y -i {input_path} {frames_dir}/%08d.png")
        
        input_imgs = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        restored_dir = os.path.join(temp_dir, "restored")
        os.makedirs(restored_dir, exist_ok=True)
        
        print(f"Restoring {len(input_imgs)} frames with {self.model_name}...")
        
        for i, img_path in enumerate(input_imgs):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # restore
            # outscale=0.5 because simple 4x upscaling makes it 2048px which is too big/slow.
            # We want to keep 512x512 but CLEANED.
            # actually our input is 512. Model upscales 4x -> 2048.
            # So we output 0.25 scale?
            # RealESRGANer `enhance` returns output.
            output, _ = self.restorer.enhance(img, outscale=1.0) # Keep scale 4x? No, 512->2048.
            # If we want 512 result, we can downscale after?
            # Or outscale=0.25 (if supported)?
            # Let's try keeping it 512 by resizing AFTER restoration.
            # Actually, `enhance` accepts `outscale`. Upscaling cleans artifacts.
            
            # Use outscale=1.0 -> 2048x2048. Then resize to 512x512.
            # This is slow.
            # Is there a 1x model? No.
            # Alternative: Resize input to 128, restore 4x to 512?
            # That loses details.
            # Correct path: Restore 4x (2048), then Resize back to 512. This supersampling improves quality.
            
            # Save
            save_path = os.path.join(restored_dir, f"{i:08d}.png")
            # Downscale back to 512
            if output.shape[0] > 512:
                output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(save_path, output)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(input_imgs)}")

        # Encode
        print("RealESRGAN: Encoding video...")
        temp_vid = os.path.join(temp_dir, "restored.mp4")
        os.system(f"{ffmpeg_exe} -y -v warning -r 25 -i {restored_dir}/%08d.png -vcodec libx264 -pix_fmt yuv420p -crf 18 {temp_vid}")
        
        # Mux Audio
        os.system(f"{ffmpeg_exe} -y -v warning -i {input_path} -i {temp_vid} -map 1:v -map 0:a -c:v copy -c:a copy {output_path}")
        
        shutil.rmtree(temp_dir)
        return output_path
