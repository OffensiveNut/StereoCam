import cv2
import sys
import glob
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import os

# Reduce MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import triangulation as tri 
import calibration

import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Try different camera configurations
camera_configs = [
    (0, 2),  # Original configuration
    (0, 4),  # Alternative if camera 2 fails
    (2, 4),  # Alternative if camera 0 fails
]

cap_right = None
cap_left = None

for right_idx, left_idx in camera_configs:
    print(f"Trying camera configuration: Right={right_idx}, Left={left_idx}")
    
    cap_right = cv2.VideoCapture(right_idx)
    time.sleep(0.1)
    cap_left = cv2.VideoCapture(left_idx)
    time.sleep(0.1)
    
    if cap_right.isOpened() and cap_left.isOpened():
        # Test if we can actually read frames
        ret_right, _ = cap_right.read()
        ret_left, _ = cap_left.read()
        
        if ret_right and ret_left:
            print(f"✓ Successfully opened cameras: Right={right_idx}, Left={left_idx}")
            break
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
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while (cap_right.isOpened() and cap_left.isOpened()):
        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()
        
        frame_right, frame_left = calibration.undistort_rectify(frame_right, frame_left)
        
        if not success_right or not success_left:
            print("Error: Could not read images from cameras.")
            break
        else:   
            start = time.time()
            
            # convert bgr to rgb
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)
            
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            
            center_right = 0
            center_left = 0
            
            if results_right.detections:
                for id,detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bBox = detection.location_data.relative_bounding_box
                    
                    h,w,c = frame_right.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    cv2.putText(frame_right, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
            if results_left.detections:
                for id,detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bBox = detection.location_data.relative_bounding_box
                    
                    h,w,c = frame_left.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    cv2.putText(frame_left, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_left, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
                cv2.putText(frame_right, f"Distance: " + str(round(depth,1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_left, f"Distance: " + str(round(depth,1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Depth: {depth:.2f} cm")
            
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            
            cv2.putText(frame_right, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_left, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Right Camera", frame_right)
            cv2.imshow("Left Camera", frame_left)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
