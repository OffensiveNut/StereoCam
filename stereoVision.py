import cv2
import sys
import glob
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import os

import triangulation as tri 
import calibration

# Initialize stereo matcher for depth computation
print("Initializing stereo matcher...")
# Using SGBM for much better quality depth estimation
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,  # Increased for better depth range
    blockSize=7,        # Smaller block size for finer details
    P1=8 * 3 * 7**2,    # Smoothness parameters
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,  # Lower for more matches
    speckleWindowSize=50,  # Enabled to remove noise
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

def create_depth_colormap(disparity):
    """Create a colored depth map with proper depth scaling"""
    # Clip negative disparities and normalize
    disparity_clipped = np.clip(disparity, 0, None)
    
    # Normalize disparity to 0-255 range
    disparity_normalized = cv2.normalize(disparity_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply color map (JET colormap for heat-like visualization)
    depth_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    return depth_colored, disparity_normalized

def detect_objects_in_depth(disparity, focal_length=600, baseline=0.05):
    """Detect objects using blob detection on the depth map"""
    # Convert disparity to actual depth in meters
    # depth = (focal_length * baseline) / disparity
    # But we'll work directly with disparity for blob detection
    
    # Clip and normalize disparity
    disparity_clipped = np.clip(disparity, 1, None)  # Avoid division by zero
    disparity_normalized = cv2.normalize(disparity_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply filters to clean up the depth map
    filtered = cv2.medianBlur(disparity_normalized, 15)
    filtered = cv2.GaussianBlur(filtered, (9, 9), 0)
    
    # Create binary mask for objects at reasonable distances
    # Higher disparity = closer objects
    close_mask = cv2.inRange(filtered, 30, 255)  # Focus on closer objects
    
    # Morphological operations to clean up blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours (blobs)
    contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small noise
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get average disparity in the blob region
            roi_disparity = disparity_clipped[y:y+h, x:x+w]
            mask_roi = close_mask[y:y+h, x:x+w]
            
            if np.sum(mask_roi) > 0:
                avg_disparity = np.mean(roi_disparity[mask_roi > 0])
                
                # Convert disparity to depth (improved calculation)
                if avg_disparity > 0:
                    # Use a more realistic focal length and baseline for depth calculation
                    depth_cm = (focal_length * baseline * 100) / (avg_disparity / 16.0)  # SGBM disparity is scaled by 16
                    
                    # Clamp to reasonable range
                    depth_cm = max(10, min(depth_cm, 500))  # Between 10cm and 5m
                    
                    objects.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'depth': depth_cm,
                        'disparity': avg_disparity
                    })
    
    return objects, close_mask

print("Stereo system initialized successfully")


right_idx, left_idx = 0, 2
cap_right = None
cap_left = None

print(f"Trying camera configuration: Right={right_idx}, Left={left_idx}")

cap_right = cv2.VideoCapture(right_idx)
cap_left = cv2.VideoCapture(left_idx)
time.sleep(0.1)

if cap_right.isOpened() and cap_left.isOpened():
    # Test if we can actually read frames
    ret_right, _ = cap_right.read()
    ret_left, _ = cap_left.read()
    
    if ret_right and ret_left:
        print(f"✓ Successfully opened cameras: Right={right_idx}, Left={left_idx}")

    else:
        print(f"✗ Cameras opened but can't read frames")
        cap_right.release()
        cap_left.release()
else:
    print(f"✗ Failed to open cameras")
    if cap_right:
        cap_right.release()
    if cap_left:
        cap_left.release()
    cap_right = None
    cap_left = None

# Check if cameras opened successfully
if not cap_right or not cap_right.isOpened():
    print("Error: Could not open right camera")
    sys.exit()
if not cap_left or not cap_left.isOpened():
    print("Error: Could not open left camera")
    sys.exit()

print("Cameras initialized successfully")
print("Press 'q' to quit the application")

frame_rate = 120
B = 7 # baseline in cm
f = 4 # focal length in mm
alpha = 60 # camera FOV in degrees

# main program loop
while (cap_right.isOpened() and cap_left.isOpened()):
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()
    
    frame_right, frame_left = calibration.undistort_rectify(frame_right, frame_left)
    
    if not success_right or not success_left:
        print("Error: Could not read images from cameras.")
        break
    else:   
        start = time.time()
        
        # Convert to grayscale for processing
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.equalizeHist(gray_left)
        
        # Apply Gaussian blur to reduce noise
        gray_right = cv2.GaussianBlur(gray_right, (5, 5), 0)
        gray_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
        
        # Compute disparity map using stereo matching
        disparity = stereo.compute(gray_left, gray_right)
        
        # Create enhanced depth visualization
        depth_colored, disparity_normalized = create_depth_colormap(disparity)
        
        # Detect objects using blob detection on depth map
        detected_objects, object_mask = detect_objects_in_depth(disparity)
        
        # Create display frames
        display_left = frame_left.copy()
        display_right = frame_right.copy()
        
        # Draw detected objects
        for i, obj in enumerate(detected_objects):
            center = obj['center']
            bbox = obj['bbox']
            depth = obj['depth']
            
            # Draw bounding box
            cv2.rectangle(display_left, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(display_right, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(display_left, center, 8, (0, 0, 255), -1)
            cv2.circle(display_right, center, 8, (0, 0, 255), -1)
            
            # Draw depth information
            depth_text = f"{depth:.1f}cm"
            cv2.putText(display_left, depth_text, (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_right, depth_text, (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"Object {i+1} detected at depth: {depth:.1f}cm")
        
        # Add detection statistics
        num_objects = len(detected_objects)
        if num_objects > 0:
            depths = [obj['depth'] for obj in detected_objects]
            avg_depth = np.mean(depths)
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            
            cv2.putText(display_left, f"Objects: {num_objects}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_left, f"Closest: {min_depth:.1f}cm", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_left, f"Farthest: {max_depth:.1f}cm", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_left, "No objects detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_right, "No objects detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        
        cv2.putText(display_right, f"FPS: {int(fps)}", (50, display_right.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_left, f"FPS: {int(fps)}", (50, display_left.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"FPS: {int(fps)}", (50, depth_colored.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display windows
        cv2.imshow("Right Camera", display_right)
        cv2.imshow("Left Camera", display_left)
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
