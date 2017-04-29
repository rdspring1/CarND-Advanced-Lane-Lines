import numpy as np
import glob
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from line_detector import threshold, find_window_centroids, plot_window_centroids, draw, sanity_check
from line import Line

# Calibration Settings
nx = 9
ny = 6
height = 720
width = 1280

# Window settings
window_width = 25
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 25 # How much to slide left and right for searching

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
src = np.float32([[1110, height], [690, 450], [595, 450], [220, height]])
# define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[1000, height], [1000, 0], [350, 0], [350, height]])

# use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)


left_line = Line()
right_line = Line()

def process_image(img):
    rgb_undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = cv2.cvtColor(rgb_undist, cv2.COLOR_RGB2BGR)
    binary_img, color_binary = threshold(undist)
    warped = cv2.warpPerspective(binary_img, M, (width, height), flags=cv2.INTER_NEAREST)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    output, left, right = plot_window_centroids(warped, window_centroids, window_width, window_height)
    #status, left_pts, right_pts = curvature(undist, left, right, height)

    left_status, left_fit = left_line.curvature(left)
    right_status, right_fit = right_line.curvature(right)

    if not sanity_check(left_status, right_status, left_fit, right_fit, window_centroids):
        left_pts = left_line.avg(height)
        right_pts = right_line.avg(height)
        return draw(rgb_undist, left_pts, right_pts, Minv, width, height)
    else:
        left_pts = left_line.update(left, left_fit, height)
        right_pts = right_line.update(right, right_fit, height)
        return draw(rgb_undist, left_pts, right_pts, Minv, width, height)

VideoFileClip("challenge.mp4").fl_image(process_image).write_videofile('output.mp4', audio=False)
