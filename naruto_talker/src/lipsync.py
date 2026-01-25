import torch
import os
import sys
import numpy as np
import cv2
import glob
import shutil
import pickle
import imageio_ffmpeg
from omegaconf import OmegaConf
from transformers import WhisperModel

# Ensure MuseTalk is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "MuseTalk"))

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

class LipSyncPipeline:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.project_root = os.path.join(os.path.dirname(__file__), "..", "MuseTalk")
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using ffmpeg from: {self.ffmpeg_path}")
        
        # Hardcoded configs for MVP
        self.unet_config_path = os.path.join(self.project_root, "models/musetalk/musetalk.json")
        self.unet_model_path = os.path.join(self.project_root, "models/musetalk/pytorch_model.bin") 
        
        self.whisper_path = os.path.join(self.project_root, "models/whisper")
        self.vae, self.unet, self.pe = None, None, None
        self.audio_processor = None
        self.whisper_model = None
        self.face_parser = None
        self.version = "v1"
        
    def load_model(self):
        if self.unet is None:
            print("Loading MuseTalk models...")
            
            # Check if v1.5 exists, else v1
            v15_path = os.path.join(self.project_root, "models/musetalkV15/unet.pth")
            if os.path.exists(v15_path):
                self.unet_model_path = v15_path
                self.unet_config_path = os.path.join(self.project_root, "models/musetalkV15/musetalk.json")
                self.version = "v15"
            else:
                self.version = "v1"

            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=self.unet_model_path,
                vae_type=os.path.join(self.project_root, "models/sd-vae"), 
                unet_config=self.unet_config_path,
                device=self.device
            )
            
            self.pe.to(self.device)
            self.vae.vae.to(self.device)
            self.unet.model.to(self.device)
            
            self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_path)
            self.whisper_model = WhisperModel.from_pretrained(self.whisper_path)
            self.whisper_model.to(device=self.device).eval()
            self.whisper_model.requires_grad_(False)
            
            resnet_path = os.path.join(self.project_root, "models/face-parse-bisent/resnet18-5c106cde.pth")
            model_pth = os.path.join(self.project_root, "models/face-parse-bisent/79999_iter.pth")

            if self.version == "v15":
                self.face_parser = FaceParsing(left_cheek_width=90, right_cheek_width=90, resnet_path=resnet_path, model_pth=model_pth)
            else:
                self.face_parser = FaceParsing(resnet_path=resnet_path, model_pth=model_pth)
                
            print(f"MuseTalk ({self.version}) loaded.")

    def unload_diffusion_models(self):
        """
        Unload only the heavy diffusion components (UNet, VAE, Whisper)
        to free up VRAM for the restoration/compositing stage.
        """
        print("Unloading temporary LipSync diffusion models...")
        if self.unet is not None:
             del self.unet
             self.unet = None
        if self.vae is not None:
             del self.vae
             self.vae = None
        if self.pe is not None:
             del self.pe
             self.pe = None
        if self.whisper_model is not None:
             del self.whisper_model
             self.whisper_model = None
             
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def unload(self):
        print("Unloading LipSync models...")
        self.unload_diffusion_models()
        if self.face_parser is not None:
             del self.face_parser
             self.face_parser = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("LipSync models unloaded.")

    def inference(self, video_path, audio_path, output_path, bbox_shift=0, restoration_strength=1.0):
        self.load_model()
        
        # Setup temp paths
        temp_dir = "/tmp/musetalk_processing"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # 1. Extract frames
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print("Extracting frames...")
        # Use verified ffmpeg binary
        os.system(f"{self.ffmpeg_path} -v fatal -y -i {video_path} -start_number 0 {frames_dir}/%08d.png")
        input_img_list = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if not input_img_list:
            raise RuntimeError("Frame extraction failed. No frames found.")
        
        # 2. Audio Features
        print("Processing audio...")
        whisper_feats, librosa_len = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_feats, self.device, self.unet.model.dtype, self.whisper_model, librosa_len, fps=25
        )
        
        # 3. Landmarks
        print(f"Getting landmarks (Shift: {bbox_shift})...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        # 4. Latents
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = min(y2 + 10, frame.shape[0]) # extra margin
            
            # Validate crop dimensions
            if y2 <= y1 or x2 <= x1:
                print(f"Invalid bbox skipped: {bbox}")
                crop = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                crop = frame[max(0, y1):y2, max(0, x1):x2] # Ensure boundaries
                if crop.size == 0:
                     crop = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                     try:
                        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                     except Exception as e:
                        print(f"Resize failed for bbox {bbox}: {e}")
                        crop = np.zeros((256, 256, 3), dtype=np.uint8)

            latents = self.vae.get_latents_for_unet(crop)
            input_latent_list.append(latents)
            
        # 5. Batch Inference
        video_num = len(whisper_chunks)
        # Handle cases where video is shorter than audio
        if len(frame_list) == 0:
             raise RuntimeError("No frames read from video.")
        
        if len(input_latent_list) == 0:
             raise RuntimeError("No faces detected in the video frames. Face alignment failed.")
             
        # Create cycled lists
        frame_list_cycle = frame_list * (video_num // len(frame_list) + 1)
        coord_list_cycle = coord_list * (video_num // len(coord_list) + 1)
        input_latent_list_cycle = input_latent_list * (video_num // len(input_latent_list) + 1)
        
        frame_list_cycle = frame_list_cycle[:video_num]
        coord_list_cycle = coord_list_cycle[:video_num]
        input_latent_list_cycle = input_latent_list_cycle[:video_num]
        
        gen = datagen(
            whisper_chunks, 
            input_latent_list_cycle, 
            batch_size=8, 
            delay_frame=0, 
            device=self.device
        )
        
        res_frame_list = []
        print("Running diffusion...")
        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feat = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
            pred_latents = self.unet.model(latent_batch, torch.tensor([0], device=self.device), encoder_hidden_states=audio_feat).sample
            recon = self.vae.decode_latents(pred_latents)
            res_frame_list.extend(recon)
            
        # 6. Post-processing (Paste back)
        
        # AGGRESSIVE OPTIMIZATION: Unload heavy MuseTalk diffusion models before loading RealESRGAN
        self.unload_diffusion_models()
        
        print(f"Compositing (Strength: {restoration_strength})...")
        from src.anime_restoration import AnimeRestorationPipeline
        mask_restorer = AnimeRestorationPipeline(device=self.device.type)
        mask_restorer.load_model()
        
        composed_frames_dir = os.path.join(temp_dir, "composed")
        os.makedirs(composed_frames_dir, exist_ok=True)
        
        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i]
            ori_frame = frame_list_cycle[i].copy()
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = min(y2 + 10, ori_frame.shape[0])
            
            try:
                # MuseTalk raw output (256x256)
                raw_mouth = res_frame.astype(np.uint8)
                
                # 1. Super-Resolution
                if restoration_strength > 0:
                    high_res_mouth = mask_restorer.process_frame(raw_mouth)
                    
                    target_w, target_h = x2-x1, y2-y1
                    
                    # Resize both to target size
                    enhanced_resized = cv2.resize(high_res_mouth, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    
                    if restoration_strength < 1.0:
                        # Blend: Strength * Enhanced + (1-Strength) * Raw
                        raw_resized = cv2.resize(raw_mouth, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        to_paste = cv2.addWeighted(enhanced_resized, restoration_strength, raw_resized, 1.0 - restoration_strength, 0)
                    else:
                        to_paste = enhanced_resized
                else:
                    # No restoration
                    to_paste = cv2.resize(raw_mouth, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
                
                combine = get_image(ori_frame, to_paste, [x1, y1, x2, y2], fp=self.face_parser)
                cv2.imwrite(f"{composed_frames_dir}/{str(i).zfill(8)}.png", combine)
            except Exception as e:
                print(f"Frame {i} failed: {e}")
                cv2.imwrite(f"{composed_frames_dir}/{str(i).zfill(8)}.png", ori_frame)
        
        mask_restorer.unload()

        # 7. Encode video
        print("Encoding final video...")
        temp_vid = f"{temp_dir}/temp_silent.mp4"
        os.system(f"{self.ffmpeg_path} -y -top 1 -v warning -r 25 -f image2 -i {composed_frames_dir}/%08d.png -vcodec libx264 -vf format=yuv420p {temp_vid}")
        
        # 8. Mux audio
        os.system(f"{self.ffmpeg_path} -y -v warning -i {audio_path} -i {temp_vid} -c:v copy -c:a aac {output_path}")
        
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        return output_path
