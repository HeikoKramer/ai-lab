import cv2
import numpy as np
import torch
import sys
from PIL import Image

try:
    from face_alignment import FaceAlignment, LandmarksType
except ImportError:
    print("WARNING: face_alignment not installed. Verification will be skipped/mocked.")
    FaceAlignment = None

class VerificationPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.fa = None
        if FaceAlignment is not None and torch.cuda.is_available():
            try:
                # Initialize with SFD detector (usually better for varying scales, though heavier)
                # Filter threshold 0.1 helps reduce false positives
                self.fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, 
                                      device=device, 
                                      face_detector='sfd', 
                                      face_detector_kwargs={'filter_threshold': 0.1})
            except Exception as e:
                print(f"Failed to initialize FaceAlignment: {e}")
                self.fa = None

    def verify_image(self, image_input):
        """
        Verifies if an image contains a detected face.
        Args:
            image_input: PIL Image or path to image
        Returns:
            bool: True if face detected, False otherwise
        """
        if self.fa is None:
            print("Verification skipped (model not loaded).")
            return True

        if isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
            if cv_img is None:
                print(f"Error: Could not read image at {image_input}")
                return False
            image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input)
        else:
            print("Error: Unsupported image format for verification.")
            return False

        try:
            preds = self.fa.get_landmarks_from_image(image_rgb)
            if preds is None or len(preds) == 0:
                return False
            return True
        except Exception as e:
            print(f"Verification error: {e}")
            return False

    def check_color_consistency(self, video_frames, threshold=0.5):
        """
        Checks if video maintains color distribution consistency.
        If the video degenerates into a single color blob, the histogram correlation with the first frame will drop 
        OR the variance of the frame will drop significantly.
        """
        if not video_frames:
            return False
            
        ref_img = np.array(video_frames[0])
        ref_hist = cv2.calcHist([ref_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(ref_hist, ref_hist)
        
        passed = 0
        total = 0
        
        for i in range(1, len(video_frames)):
            curr_img = np.array(video_frames[i])
            
            # Check 1: Is the image just a gray blob? (Low variance)
            variance = np.var(curr_img)
            if variance < 100: # Heuristic threshold for "flat color"
                print(f"Frame {i} variance too low ({variance:.2f}). likely gray blob.")
                continue

            # Check 2: Color Histogram Correlation
            curr_hist = cv2.calcHist([curr_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(curr_hist, curr_hist)
            correlation = cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL)
            
            if correlation > threshold:
                passed += 1
            else:
                 print(f"Frame {i} color correlation low ({correlation:.2f}).")
            total += 1
            
        return total > 0 and (passed / total) > 0.8

    def check_structure_consistency(self, video_frames, mse_threshold=2000):
        """
        Checks if the structural content (edges, shapes) remains somewhat consistent.
        Uses MSE of downscaled thumbnails to allow for motion but reject total chaos.
        """
        if not video_frames: return False
        
        start_thumb = video_frames[0].resize((32, 32)).convert('L')
        start_arr = np.array(start_thumb).astype('float32')
        
        passed = 0
        total = 0
        
        for i in range(1, len(video_frames)):
            curr_thumb = video_frames[i].resize((32, 32)).convert('L')
            curr_arr = np.array(curr_thumb).astype('float32')
            
            mse = np.mean((start_arr - curr_arr) ** 2)
            
            # If MSE is HUGE, it means the image completely changed (e.g. to value 0 or random noise)
            # Typically 0-500 is same image, 500-2000 is movement. >3000 is different scene or artifact?
            # Actually, "color cloud" might result in VERY LOW variance (flat) but high MSE if color shift?
            
            if mse < mse_threshold:
                passed += 1
            else:
                print(f"Frame {i} structure MSE high ({mse:.2f}).")
            total += 1
            
        return total > 0 and (passed / total) > 0.8

    def verify_video(self, video_frames, sample_rate=5, success_threshold=0.8):
        """
        Verifies a sequence of video frames.
        """
        if self.fa is None:
            return True

        if not video_frames:
            print("Error: No frames to verify.")
            return False

        # 1. Structural/Color Sanity Check (Cheap)
        print("Running Color/Structure Consistency Checks...")
        if not self.check_color_consistency(video_frames):
             print("FAILURE: Video lost color consistency (turned into blob/cloud).")
             return False
        
        if not self.check_structure_consistency(video_frames):
             print("FAILURE: Video lost structural consistency (high MSE).")
             return False

        total_checked = 0
        passed_frames = 0
        
        # Check first frame (critical)
        if not self.verify_image(video_frames[0]):
             print("Video Verification Failed: First frame has no detectable face.")
             return False

        # Check subsampled frames
        for i in range(0, len(video_frames), sample_rate):
            total_checked += 1
            if self.verify_image(video_frames[i]):
                passed_frames += 1
        
        if total_checked == 0:
            return False

        success_rate = passed_frames / total_checked
        print(f"Video Verification Stats: {passed_frames}/{total_checked} frames passed ({success_rate:.2%})")

        if success_rate >= success_threshold:
            return True
        else:
            print(f"Video Verification Failed: Success rate {success_rate:.2f} < {success_threshold}")
            return False

    def close(self):
         if self.fa:
             del self.fa
             self.fa = None
             torch.cuda.empty_cache()
