import numpy as np
import cv2
import glob

chessboardSize = (5, 10)  # Number of inner corners per a chessboard row and column
frame_size = (640, 480)  # Size of the images used for calibration

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane for left camera
imgpointsR = []  # 2d points in image plane for right camera

imageLeft = glob.glob('images/stereoLeft/*.png')
imageRight = glob.glob('images/stereoRight/*.png')

print(f"Found {len(imageLeft)} left images and {len(imageRight)} right images")

successful_pairs = 0
for i, (imgLeft, imgRight) in enumerate(zip(sorted(imageLeft), sorted(imageRight))):
    print(f"Processing pair {i+1}: {imgLeft} and {imgRight}")
    
    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
    
    if imgL is None:
        print(f"Error: Could not read left image {imgLeft}")
        continue
    if imgR is None:
        print(f"Error: Could not read right image {imgRight}")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    print(f"Image sizes: Left {grayL.shape}, Right {grayR.shape}")

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)
    
    print(f"Chessboard detection: Left={retL}, Right={retR}")

    if retL and retR:
        objpoints.append(objp)
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpointsL.append(cornersL2)
        imgpointsR.append(cornersR2)
        successful_pairs += 1
        
        cv2.drawChessboardCorners(imgL, chessboardSize, cornersL2, retL)
        cv2.drawChessboardCorners(imgR, chessboardSize, cornersR2, retR)
        cv2.imshow('Left Camera', imgL)
        cv2.imshow('Right Camera', imgR)    
        cv2.waitKey(100)
    else:
        print(f"Skipping pair {i+1} - chessboard not detected in both images")

cv2.destroyAllWindows()
print(f"Successfully processed {successful_pairs} image pairs")

if successful_pairs == 0:
    print("Error: No chessboard patterns were detected in any image pairs!")
    print("Please check:")
    print("1. The chessboard size is correct (currently set to 9x6)")
    print("2. The images contain clear chessboard patterns")
    print("3. The chessboard patterns are not cut off at the edges")
    exit(1)

if successful_pairs < 10:
    print(f"Warning: Only {successful_pairs} successful image pairs found.")
    print("For good calibration results, it's recommended to have at least 10-20 image pairs.")

# Calibrate the cameras
print("Calibrating left camera...")
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frame_size, None, None)
print("Calibrating right camera...")
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frame_size, None, None)

heightL, widthL, channelsL = imgL.shape
heightR, widthR, channelsR = imgR.shape

newCameraMatrixL, roiL = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
newCameraMatrixR, roiR = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1],
    criteria=criteria_stereo, flags=flags)

rectifyScale = 0  # 0 for full crop, 1 for no crop
rectL, rectR, projMatrixL, projMatrixR, Q, roiL, roiR = cv2.stereoRectify(
    newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

print("saving parameters...")
cv_file = cv2.FileStorage("stereo_map.xml", cv2.FILE_STORAGE_WRITE)

cv_file.write("stereoMapL_x", stereoMapL[0])
cv_file.write("stereoMapL_y", stereoMapL[1])
cv_file.write("stereoMapR_x", stereoMapR[0])
cv_file.write("stereoMapR_y", stereoMapR[1])
cv_file.release()