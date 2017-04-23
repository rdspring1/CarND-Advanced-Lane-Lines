import numpy as np
import glob
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from line_detector import threshold, find_window_centroids, plot_window_centroids, curvature, draw

# Calibration Settings
nx = 9
ny = 6
height = 720
width = 1280

# Window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

"""
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
"""
imgpoints = []
objpoints = []
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

images = glob.glob("camera_cal/calibration*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

# Perspective Transformation Settings
# define 4 source points src = np.float32([[,],[,],[,],[,]])
src = np.float32([[1150, height], [690, 450], [595, 450], [220, height]])
# define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[1020, height], [1000, 0], [300, 0], [300, height]])

# use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def process_image(img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    undist = cv2.undistort(bgr_img, mtx, dist, None, mtx)
    rgb_undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    binary_img, color_binary = threshold(undist)
    warped = cv2.warpPerspective(binary_img, M, (width, height), flags=cv2.INTER_NEAREST)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    output, left, right = plot_window_centroids(warped, window_centroids, window_width, window_height)
    result_warped, left_pts, right_pts = curvature(undist, left, right)
    result = draw(rgb_undist, left_pts, right_pts, Minv, width, height)
    return result

VideoFileClip("challenge.mp4").fl_image(process_image).write_videofile('output.mp4', audio=False)

