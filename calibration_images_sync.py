import cv2
import threading
import time
import queue
import numpy as np

class SynchronizedStereoCapture:
    def __init__(self, left_camera_id=2, right_camera_id=0):
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.left_cap = None
        self.right_cap = None
        self.left_queue = queue.Queue(maxsize=2)
        self.right_queue = queue.Queue(maxsize=2)
        self.capture_event = threading.Event()
        self.stop_capture = threading.Event()
        self.frame_counter = 0
        
    def initialize_cameras(self):
        """Initialize both cameras with optimal settings for synchronization"""
        print("Initializing cameras...")
        
        # Initialize cameras
        self.left_cap = cv2.VideoCapture(self.left_camera_id)
        self.right_cap = cv2.VideoCapture(self.right_camera_id)
        
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            print("Error: Could not open one or both cameras")
            return False
            
        # Configure cameras for synchronization
        cameras = [self.left_cap, self.right_cap]
        for cap in cameras:
            # Minimal buffer to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set consistent frame rate
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Set consistent resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Disable auto exposure for consistent timing
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # Set exposure manually if needed
            # cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            
        print("Cameras initialized successfully")
        return True
        
    def capture_left_frames(self):
        """Thread function for capturing left camera frames"""
        while not self.stop_capture.is_set():
            if self.capture_event.wait(timeout=0.1):
                ret, frame = self.left_cap.read()
                if ret:
                    timestamp = time.time()
                    try:
                        self.left_queue.put((frame, timestamp), timeout=0.01)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                self.capture_event.clear()
                
    def capture_right_frames(self):
        """Thread function for capturing right camera frames"""
        while not self.stop_capture.is_set():
            if self.capture_event.wait(timeout=0.1):
                ret, frame = self.right_cap.read()
                if ret:
                    timestamp = time.time()
                    try:
                        self.right_queue.put((frame, timestamp), timeout=0.01)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                        
    def get_synchronized_frames(self):
        """Get synchronized frames from both cameras"""
        # Trigger simultaneous capture
        self.capture_event.set()
        
        try:
            # Get frames with timeout
            left_frame, left_time = self.left_queue.get(timeout=0.1)
            right_frame, right_time = self.right_queue.get(timeout=0.1)
            
            # Calculate synchronization error
            sync_error = abs(left_time - right_time) * 1000  # in milliseconds
            
            return True, left_frame, True, right_frame, sync_error
            
        except queue.Empty:
            return False, None, False, None, 0
            
    def start_capture_threads(self):
        """Start the capture threads"""
        self.left_thread = threading.Thread(target=self.capture_left_frames)
        self.right_thread = threading.Thread(target=self.capture_right_frames)
        
        self.left_thread.daemon = True
        self.right_thread.daemon = True
        
        self.left_thread.start()
        self.right_thread.start()
        
    def stop_capture_threads(self):
        """Stop the capture threads"""
        self.stop_capture.set()
        self.capture_event.set()  # Wake up threads
        
        if hasattr(self, 'left_thread'):
            self.left_thread.join(timeout=1)
        if hasattr(self, 'right_thread'):
            self.right_thread.join(timeout=1)
            
    def release(self):
        """Release resources"""
        self.stop_capture_threads()
        
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
            
        cv2.destroyAllWindows()

def main():
    # Create synchronized capture object
    stereo_capture = SynchronizedStereoCapture(left_camera_id=2, right_camera_id=0)
    
    if not stereo_capture.initialize_cameras():
        return
        
    # Start capture threads
    stereo_capture.start_capture_threads()
    
    # Wait a moment for threads to start
    time.sleep(0.5)
    
    num = 0
    sync_errors = []
    
    print("Press 's' to save synchronized image pair, 'q' to quit")
    print("Synchronization error will be displayed in milliseconds")
    
    try:
        while True:
            success1, left_frame, success2, right_frame, sync_error = stereo_capture.get_synchronized_frames()
            
            if success1 and success2:
                sync_errors.append(sync_error)
                
                # Display synchronization info
                info_text = f"Sync Error: {sync_error:.2f}ms | Avg: {np.mean(sync_errors[-10:]):.2f}ms"
                cv2.putText(left_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(right_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Left Camera (Synchronized)', left_frame)
                cv2.imshow('Right Camera (Synchronized)', right_frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    # Save synchronized images
                    timestamp = int(time.time() * 1000)
                    left_filename = f'images/stereoLeft/imageL{num}_sync_{timestamp}.png'
                    right_filename = f'images/stereoRight/imageR{num}_sync_{timestamp}.png'
                    
                    cv2.imwrite(left_filename, left_frame)
                    cv2.imwrite(right_filename, right_frame)
                    
                    print(f'Synchronized pair {num} saved! Sync error: {sync_error:.2f}ms')
                    num += 1
            else:
                print("Failed to capture synchronized frames")
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        stereo_capture.release()
        print("Cameras released and threads stopped")
        
        if sync_errors:
            print(f"\nSynchronization Statistics:")
            print(f"Average sync error: {np.mean(sync_errors):.2f}ms")
            print(f"Max sync error: {np.max(sync_errors):.2f}ms")
            print(f"Min sync error: {np.min(sync_errors):.2f}ms")

if __name__ == "__main__":
    main()
