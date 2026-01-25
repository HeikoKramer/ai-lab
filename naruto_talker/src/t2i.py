import torch
from diffusers import StableDiffusionXLPipeline
import os

class T2IPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.model_id = "stabilityai/sdxl-turbo"
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            print(f"Loading T2I model: {self.model_id}...")
            # Most modern turbo models work with AutoPipeline.
            # Most modern turbo models work with AutoPipeline.
            from diffusers import AutoPipelineForText2Image
            # Load directly to GPU to avoid RAM spikes (User reported 100% RAM usage)
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16, # Use float16 for VRAM savings
                use_safetensors=True,
            ).to(self.device)
            
            # self.pipe.enable_model_cpu_offload() # DISABLED: Causing RAM exhaustion
            print("T2I model loaded (SDXL-Turbo, float16).")

    def unload(self):
        if self.pipe is not None:
            print("Unloading T2I model...")
            del self.pipe
            self.pipe = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("T2I model unloaded.")

    def generate(self, prompt, negative_prompt=None, seed=None, width=768, height=768, steps=4, cfg=0.0): # Turbo settings: 4 steps for better anatomy
        self.load_model()
        
        full_prompt = prompt
        
        base_negative = (
            "mutated, extra eyes, third eye, wrong anatomy, "
            "photorealistic, 3d, lowres, blurry, extra limbs, deformed face, bad teeth, "
            "open mouth, warped mouth, multiple faces, text, watermark, logo, jpeg artifacts, "
            "grid, split screen, distorted"
        )
        
        if negative_prompt:
            final_negative = f"{negative_prompt}, {base_negative}"
        else:
            final_negative = base_negative
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        print(f"Generating portrait with prompt: {full_prompt[:100]}...")
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=final_negative,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        
        return image
