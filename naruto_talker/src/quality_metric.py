import cv2
import numpy as np

def calculate_laplacian_variance(image_path_or_array):
    """
    Calculates the Laplacian Variance of an image.
    Higher value = Sharper image.
    Lower value (< 100) = Blurry.
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
    else:
        img = image_path_or_array
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_video_sharpness(video_path, sample_rate=5, threshold=100.0):
    """
    Checks the average sharpness of a video.
    Returns: (average_variance, passed_boolean, per_frame_variances)
    """
    cap = cv2.VideoCapture(video_path)
    variances = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            var = calculate_laplacian_variance(frame)
            variances.append(var)
            
        frame_count += 1
        
    cap.release()
    
    if not variances:
        return 0.0, False, []
        
    avg_variance = np.mean(variances)
    passed = avg_variance > threshold
    return avg_variance, passed, variances
