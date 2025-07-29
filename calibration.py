import sys 
import numpy as np
import cv2
import time
import imutils
import os

# Check if calibration file exists
if not os.path.exists("stereo_map.xml"):
    print("Error: stereo_map.xml not found!")
    print("Please run stereo_calibration.py first to generate calibration data.")
    sys.exit()

cv_file = cv2.FileStorage()
cv_file.open("stereo_map.xml", cv2.FILE_STORAGE_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

# Check if calibration data loaded successfully
if stereoMapL_x is None or stereoMapL_y is None or stereoMapR_x is None or stereoMapR_y is None:
    print("Error: Could not load calibration data from stereo_map.xml")
    print("Please run stereo_calibration.py again to regenerate calibration data.")
    sys.exit()

print("Calibration data loaded successfully")

def undistort_rectify(frameR, frameL):
    undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)
    undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
    return undistortedR, undistortedL
    
