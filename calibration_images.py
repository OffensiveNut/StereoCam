import cv2
import threading
import time

# Initialize cameras with optimized settings for synchronization
def initialize_camera(camera_id, buffer_size=1):
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        # Set buffer size to minimum to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        # Set consistent frame rate
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Set consistent resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Disable auto exposure for consistent timing
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    return cap

# Initialize both cameras simultaneously
print("Initializing cameras...")
cap = initialize_camera(2)
cap2 = initialize_camera(0)

# Verify both cameras are working
if not cap.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras")
    exit()

print("Cameras initialized successfully")

num = 0

# Synchronized frame capture function
def capture_synchronized_frames():
    # Capture frames as close to simultaneously as possible
    ret1, frame1 = cap.read()
    ret2, frame2 = cap2.read()
    return ret1, frame1, ret2, frame2

while cap.isOpened() and cap2.isOpened():
    
    # Use synchronized capture function
    success1, img, success2, img2 = capture_synchronized_frames()
    
    if not success1 or not success2:
        print("Failed to capture from one or both cameras")
        continue
    
    k = cv2.waitKey(5)
    if k == ord('q'):
        break
    if k == ord('s'):
        # Save images with timestamp for verification
        timestamp = int(time.time() * 1000)  # milliseconds
        cv2.imwrite(f'images/stereoLeft/imageL{num}_{timestamp}.png', img)
        cv2.imwrite(f'images/stereoRight/imageR{num}_{timestamp}.png', img2)
        print(f'Synchronized image pair {num} saved with timestamp {timestamp}!')
        num += 1
        
    cv2.imshow('Left Camera', img)
    cv2.imshow('Right Camera', img2)

# Clean up
cap.release()
cap2.release()
cv2.destroyAllWindows()
print("Cameras released and windows closed")