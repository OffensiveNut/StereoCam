import cv2
import sys
import glob
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import os

# Reduce logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import triangulation as tri 
import calibration

from ultralytics import YOLO

# Initialize YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # yolov8n is the nano version (fastest)
print("YOLO model loaded successfully")


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
        
        # Run YOLO detection on both frames
        results_right = model(frame_right, verbose=False)
        results_left = model(frame_left, verbose=False)
        
        # Get top 5 detections for each frame
        detections_right = []
        detections_left = []
        
        # Process right frame detections
        if len(results_right[0].boxes) > 0:
            boxes_right = results_right[0].boxes
            confidences = boxes_right.conf.cpu().numpy()
            # Filter by confidence threshold (51%) and sort by confidence
            valid_indices = np.where(confidences > 0.51)[0]
            if len(valid_indices) > 0:
                valid_confidences = confidences[valid_indices]
                sorted_indices = np.argsort(valid_confidences)[::-1][:5]  # Top 5 among valid ones
                top_indices = valid_indices[sorted_indices]
            else:
                top_indices = []
            
            for idx in top_indices:
                box = boxes_right[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                class_name = model.names[cls]
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections_right.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })
        
        # Process left frame detections
        if len(results_left[0].boxes) > 0:
            boxes_left = results_left[0].boxes
            confidences = boxes_left.conf.cpu().numpy()
            # Filter by confidence threshold (51%) and sort by confidence
            valid_indices = np.where(confidences > 0.51)[0]
            if len(valid_indices) > 0:
                valid_confidences = confidences[valid_indices]
                sorted_indices = np.argsort(valid_confidences)[::-1][:5]  # Top 5 among valid ones
                top_indices = valid_indices[sorted_indices]
            else:
                top_indices = []
            
            for idx in top_indices:
                box = boxes_left[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                class_name = model.names[cls]
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections_left.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })
        
        # Draw detections and calculate depths
        if len(detections_right) == 0 or len(detections_left) == 0:
            cv2.putText(frame_right, "No objects detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_left, "No objects detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Match objects between left and right frames by class and position
            for i, det_right in enumerate(detections_right):
                # Find closest matching object in left frame
                best_match = None
                min_distance = float('inf')
                
                for det_left in detections_left:
                    # Match by class and proximity
                    if det_right['class_id'] == det_left['class_id']:
                        # Calculate distance between centers
                        dx = det_right['center'][0] - det_left['center'][0]
                        dy = det_right['center'][1] - det_left['center'][1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance < min_distance and distance < 100:  # Threshold for matching
                            min_distance = distance
                            best_match = det_left
                
                if best_match:
                    # Calculate depth
                    depth = tri.find_depth(det_right['center'], best_match['center'], 
                                         frame_right, frame_left, B, f, alpha)
                    
                    # Draw bounding box and info on right frame
                    x1, y1, x2, y2 = det_right['bbox']
                    cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det_right['class']}: {det_right['confidence']:.2f}"
                    depth_label = f"Depth: {depth:.1f}cm"
                    
                    cv2.putText(frame_right, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame_right, depth_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw bounding box on left frame
                    x1_l, y1_l, x2_l, y2_l = best_match['bbox']
                    cv2.rectangle(frame_left, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 0), 2)
                    cv2.putText(frame_left, label, (x1_l, y1_l-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame_left, depth_label, (x1_l, y2_l+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    print(f"{det_right['class']}: {depth:.2f}cm (confidence: {det_right['confidence']:.2f})")
                else:
                    # Draw unmatched detection on right frame only
                    x1, y1, x2, y2 = det_right['bbox']
                    cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{det_right['class']}: {det_right['confidence']:.2f} (No match)"
                    cv2.putText(frame_right, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        
        cv2.putText(frame_right, f"FPS: {int(fps)}", (50, frame_right.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_left, f"FPS: {int(fps)}", (50, frame_left.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Right Camera", frame_right)
        cv2.imshow("Left Camera", frame_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
