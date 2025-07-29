import cv2
import numpy as np

def triangulate_points(imgL, imgR, stereo_maps):
    """
    Triangulate points from stereo image pair to compute depth
    """
    # Undistort and rectify images
    from calibration import undistort_rectify
    rectifiedL, rectifiedR = undistort_rectify(imgL, imgR, stereo_maps)
    
    # Convert to grayscale
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY) if len(rectifiedL.shape) == 3 else rectifiedL
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY) if len(rectifiedR.shape) == 3 else rectifiedR
    
    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
    
    # Compute disparity map
    disparity = stereo.compute(grayL, grayR)
    
    return disparity, rectifiedL, rectifiedR

def compute_depth_map(imgL, imgR, stereo_maps, baseline=60.0, focal_length=700.0):
    """
    Compute depth map from stereo image pair
    
    Args:
        imgL, imgR: Left and right stereo images
        stereo_maps: Stereo calibration maps
        baseline: Distance between cameras in mm
        focal_length: Focal length in pixels
    
    Returns:
        depth_map: Depth map in mm
        disparity: Raw disparity map
        rectified images
    """
    disparity, rectL, rectR = triangulate_points(imgL, imgR, stereo_maps)
    
    # Convert disparity to depth
    # Depth = (baseline * focal_length) / disparity
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    
    # Avoid division by zero
    valid_pixels = disparity > 0
    depth_map[valid_pixels] = (baseline * focal_length) / disparity[valid_pixels]
    
    return depth_map, disparity, rectL, rectR

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right       #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity                    #Depth in [mm]

    return abs(zDepth)

def create_depth_colormap(depth_map, max_depth=2000):
    """
    Create a colorized depth map for visualization
    
    Args:
        depth_map: Input depth map
        max_depth: Maximum depth for normalization
    
    Returns:
        colorized_depth: RGB image showing depth information
    """
    # Normalize depth map
    depth_normalized = np.clip(depth_map / max_depth, 0, 1)
    
    # Convert to 8-bit
    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    colorized_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    
    return colorized_depth
