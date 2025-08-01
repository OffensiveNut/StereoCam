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

# Initialize feature detector (ORB is fast and accurate)
orb = cv2.ORB_create(nfeatures=1000)  # Increased features for better matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize WLS filter for disparity post-processing
left_matcher = stereo
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.2)

def create_depth_colormap(disparity, gray_left, gray_right):
    """Create a colored depth map similar to the reference image"""
    try:
        # Apply WLS filter for better disparity quality
        disparity_right = right_matcher.compute(gray_right, gray_left)
        disparity_filtered = wls_filter.filter(disparity, gray_left, None, disparity_right)
    except:
        # Fallback to original disparity if WLS filtering fails
        disparity_filtered = disparity
    
    # Normalize disparity to 0-255 range
    disparity_normalized = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply color map (JET colormap for heat-like visualization)
    depth_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # Enhance the visualization with bilateral filter
    depth_colored = cv2.bilateralFilter(depth_colored, 9, 75, 75)
    
    return depth_colored

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
        depth_colored = create_depth_colormap(disparity, gray_left, gray_right)
        
        # Find keypoints and descriptors using ORB
        kp_right, des_right = orb.detectAndCompute(gray_right, None)
        kp_left, des_left = orb.detectAndCompute(gray_left, None)
        
        # Match features between left and right images
        matches = []
        if des_right is not None and des_left is not None and len(kp_right) > 0 and len(kp_left) > 0:
            matches = bf.match(des_right, des_left)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Keep only good matches (top 50 or those with distance < threshold)
            good_matches = matches[:min(50, len(matches))]
            good_matches = [m for m in good_matches if m.distance < 50]
        else:
            good_matches = []
        
        # Draw matched features and calculate depths
        matched_frame_right = frame_right.copy()
        matched_frame_left = frame_left.copy()
        
        depth_measurements = []
        
        for i, match in enumerate(good_matches[:20]):  # Process top 20 matches
            # Check bounds to prevent overflow
            if match.queryIdx >= len(kp_right) or match.trainIdx >= len(kp_left):
                continue
                
            # Get coordinates of matched points
            pt_right = kp_right[match.queryIdx].pt
            pt_left = kp_left[match.trainIdx].pt
            
            # Only process if the match makes sense (left point should be to the left)
            if pt_left[0] < pt_right[0]:
                # Calculate depth using triangulation
                depth = tri.find_depth(pt_right, pt_left, frame_right, frame_left, B, f, alpha)
                
                if depth > 0 and depth < 1000:  # Filter out unrealistic depths
                    depth_measurements.append({
                        'point_right': pt_right,
                        'point_left': pt_left,
                        'depth': depth,
                        'quality': 1.0 / (match.distance + 1)  # Higher quality for lower distance
                    })
                    
                    # Draw circles on detected points
                    cv2.circle(matched_frame_right, (int(pt_right[0]), int(pt_right[1])), 8, (0, 255, 0), 2)
                    cv2.circle(matched_frame_left, (int(pt_left[0]), int(pt_left[1])), 8, (0, 255, 0), 2)
                    
                    # Draw depth information
                    cv2.putText(matched_frame_right, f"{depth:.1f}cm", 
                               (int(pt_right[0] + 10), int(pt_right[1] - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    print(f"Object detected at depth: {depth:.2f}cm")
        
        # Add detection count and statistics
        num_detections = len(depth_measurements)
        if num_detections > 0:
            depths = [d['depth'] for d in depth_measurements]
            avg_depth = np.mean(depths)
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            
            cv2.putText(matched_frame_right, f"Objects: {num_detections}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(matched_frame_right, f"Avg: {avg_depth:.1f}cm", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(matched_frame_right, f"Range: {min_depth:.1f}-{max_depth:.1f}cm", (50, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(matched_frame_right, "No objects detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(matched_frame_left, "No objects detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        
        cv2.putText(matched_frame_right, f"FPS: {int(fps)}", (50, matched_frame_right.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(matched_frame_left, f"FPS: {int(fps)}", (50, matched_frame_left.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"FPS: {int(fps)}", (50, depth_colored.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display windows
        cv2.imshow("Right Camera", matched_frame_right)
        cv2.imshow("Left Camera", matched_frame_left)
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
