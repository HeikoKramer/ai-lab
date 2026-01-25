import sys
import face_alignment
from face_alignment import FaceAlignment, LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

# REMOVED: mmpose imports
# from mmpose.apis import inference_topdown, init_model
# from mmpose.structures import merge_data_samples

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Lower threshold for anime support
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device, face_detector='sfd', face_detector_kwargs={'filter_threshold': 0.1})

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    
    # We process image by image using FaceAlignment
    # Adapting logic to match original function structure
    
    average_range_minus = []
    average_range_plus = []
    
    print("Extracting landmarks using FaceAlignment...")
    for frame in tqdm(frames):
        # fa expects RGB usually, cv2 is BGR
        # But let's check fa default. Usually it handles it or expects RGB.
        # face_alignment typically expects RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get landmarks and boxes directly from FA
        # fa.get_detections_for_batch was used for bbox in original code
        # fa.get_landmarks_from_image returns list of landmarks
        
        # We need both bbox and landmarks.
        # The original code loops batches. I'll simplify to frame-by-frame for robustness in patch.
        
        try:
            # fa (pip version) get_landmarks_from_image returns list of numpy arrays or None
            preds = fa.get_landmarks_from_image(frame_rgb)
        except Exception as e:
            print(f"Face detection failed: {e}")
            coords_list.append(coord_placeholder)
            continue

        if preds is None or len(preds) == 0:
            coords_list.append(coord_placeholder)
            continue
            
        # Assume first face
        face_land_mark = preds[0].astype(np.int32)
        
        # dlib/face_alignment 68 points logic preserved

        # Calculate bbox from landmarks
        # Pip version get_landmarks doesn't give us the detector box 'dets' separately easily
        # So we MUST rely on landmarks to form the bbox.
        # Fallback to bbox logic in original code is skipped since we don't have 'dets'.
        
        # Create a safe bbox from landmarks
        x_min = np.min(face_land_mark[:, 0])
        x_max = np.max(face_land_mark[:, 0])
        y_min = np.min(face_land_mark[:, 1])
        y_max = np.max(face_land_mark[:, 1])
        
        # Pad slightly to act as detection box? 
        # Original logic used 'f' from detector as fallback.
        # Here we just use landmarks bbox.
        f = [x_min, y_min, x_max, y_max] # simplified detection box
             
        half_face_coord = face_land_mark[29]
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
        
        if upperbondrange != 0:
             half_face_coord[1] = upperbondrange + half_face_coord[1]
             
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)
        
        f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
        x1, y1, x2, y2 = f_landmark
        
        if y2-y1 <= 0 or x2-x1 <= 0 or x1 < 0:
            # Check if fallback f is valid
            fx1, fy1, fx2, fy2 = f
            if fy2-fy1 <= 0 or fx2-fx1 <= 0:
                 coords_list.append(coord_placeholder)
                 print("Invalid fallback bbox, using placeholder:", f)
            else:
                 coords_list.append(f) # fallback to detection bbox
                 print("Using fallback bbox:", f)
        else:
            coords_list.append(f_landmark)

    return coords_list, frames

# Stub for compatibility if anything else imports it
def get_bbox_range(img_list, upperbondrange=0):
    return "function disabled in lite mode"
