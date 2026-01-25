import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
import os

class I2VPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.model_id = "Lightricks/LTX-Video" # Open weights (usually v0.9.1)
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            print(f"Loading I2V model: {self.model_id}...")
            self.pipe = LTXConditionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            # Disable CPU offloading to prevent RAM exhaustion
            # self.pipe.enable_model_cpu_offload()
            # Disable tiling explicitly to avoid artifacts
            self.pipe.vae.disable_tiling()
            
            # Patch scheduler to avoid "mu" error
            # Lightricks/LTX-Video config might enable dynamic shifting but missing mu default in this diffusers version
            if hasattr(self.pipe.scheduler.config, "use_dynamic_shifting") and self.pipe.scheduler.config.use_dynamic_shifting:
                # Ensure mu is present (default often 1.0 or derived)
                if not hasattr(self.pipe.scheduler.config, "mu") or self.pipe.scheduler.config.mu is None:
                     self.pipe.scheduler.config.use_dynamic_shifting = False
                     print("Disabled use_dynamic_shifting in scheduler to avoid missing 'mu' error.")
            
            print("I2V model loaded.")

    def unload(self):
        if self.pipe is not None:
            print("Unloading I2V model...")
            del self.pipe
            self.pipe = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("I2V model unloaded.")

    def generate(self, input_image, seed=None, num_frames=81, prompt=None, negative_prompt=None, motion_speed="Subtle"):
        self.load_model()
        
        # Helper for resolution alignment
        def round_to_nearest_resolution_acceptable_by_vae(height, width):
            height = height - (height % self.pipe.vae_spatial_compression_ratio)
            width = width - (width % self.pipe.vae_spatial_compression_ratio)
            return height, width

        width, height = input_image.size
        # No initial rounding here, we scale first.
        
        # Prepare condition
        temp_video_path = "/tmp/temp_i2v_input.mp4"
        export_to_video([input_image], temp_video_path, fps=24)
        video_cond = load_video(temp_video_path)
        condition1 = LTXVideoCondition(video=video_cond, frame_index=0)
        
        # Motion Presets
        if motion_speed == "Static":
             # "Frozen" background, min movement
             stability_prompt = "static background, frozen background, no camera motion, still camera, subtle blinking only"
             safety_negative = "camera motion, panning, zooming, rotating camera, shaking, moving background, dynamic action, warping"
        elif motion_speed == "Dynamic":
             # Allow more movement
             stability_prompt = "cinematic camera movement, dynamic angle, natural head movement, blinking"
             safety_negative = "distorted, morphing, jittery, low quality"
        else:
             # Default "Subtle"
             stability_prompt = "subtle head movement, natural blinking, slight breathing motion, stable face, no exaggerated expressions, no talking mouth"
             safety_negative = "worst quality, inconsistent motion, blurry, jittery, distorted, talking, open mouth, morphing face, fast camera pan"
        
        if prompt:
             final_prompt = f"{prompt}, {stability_prompt}"
        else:
             final_prompt = stability_prompt
             
        if negative_prompt is None:
            negative_prompt = safety_negative
        else:
            negative_prompt = f"{negative_prompt}, {safety_negative}"

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating silent video ({num_frames} frames)...")
        
        # Smart scaling: Ensure minimum dimension is at least 512 for quality
        # AND Ensure maximum dimension is not > 768 for stability/VRAM.
        min_dim = 512
        max_dim = 768
        
        scale = 1.0
        
        # Logic: 
        # 1. Scale down if too big
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
        
        # 2. Scale up if too small (but don't override 1 if it creates conflict? usually min < max)
        elif width < min_dim or height < min_dim:
             scale = min_dim / min(width, height)
        
        # Apply scale
        gen_width = int(width * scale)
        gen_height = int(height * scale)
        
        # 3. Final clamp check just in case ratio is extreme
        # If gen_width is still big? (1024x100 -> 768x75. Fine.)
        # What if gen_width < 512 after downscaling? 
        # e.g. 100x2000 -> scale=768/2000=0.38 -> 38x768. Bad.
        # But aspect ratio is preserved. 
        # LTX needs reasonably standard resolutions.
        # Ideally we stick to multiples of 32.
        
        # Override to strict buckets for stability if needed, but let's try just rounding first.
        # But force a minimum of 480 or 512 to avoid abstraction.
        
        gen_height, gen_width = round_to_nearest_resolution_acceptable_by_vae(gen_height, gen_width)
        
        # Enforce strict 32 divisibility just in case VAE compression is tricky
        gen_height = (gen_height // 32) * 32
        gen_width = (gen_width // 32) * 32
        
        print(f"I2V Generation Resolution: {gen_width}x{gen_height} (Original: {width}x{height})")
        
        video_frames = self.pipe(
            conditions=[condition1],
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=gen_width,
            height=gen_height,
            num_frames=num_frames,
            num_inference_steps=40, # Increased for stability
            guidance_scale=3.0, # Explicit guidance
            generator=generator,
            output_type="pil",
        ).frames[0]
        
        # Clean up temp
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return video_frames # List of PIL images
