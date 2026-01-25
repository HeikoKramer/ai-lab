import sys
import os
import cv2
import torch
import face_alignment
import numpy as np

def verify_face_detect(image_path):
    print(f"Testing face detection on: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found.")
        return

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image.")
        return
        
    print(f"Image shape: {frame.shape}")

    # Init face alignment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Try with lower threshold for anime
        print("Initializing FaceAlignment with low threshold (0.1)...")
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            flip_input=False, 
            device=device,
            face_detector='sfd',
            face_detector_kwargs={'filter_threshold': 0.1}
        )
        
        # Detect
        preds = fa.get_landmarks(frame)
        
        if preds is None:
            print("FAILURE: No faces detected (preds is None).")
        else:
            print(f"SUCCESS: Detected {len(preds)} faces.")
            # Print bounding box estimate
            for i, lm in enumerate(preds):
                x_min, y_min = np.min(lm, axis=0)
                x_max, y_max = np.max(lm, axis=0)
                print(f"Face {i}: Bbox [{x_min}, {y_min}, {x_max}, {y_max}]")
                
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to the one from the failed run if possible, or search for one
        # Based on user logs: run_1769242533
        img_path = "/home/heiko/projects/ai-lab/naruto_talker/outputs/run_1769242533/portrait.png"
        
    verify_face_detect(img_path)
