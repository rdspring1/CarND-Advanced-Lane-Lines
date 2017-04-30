## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted Calibration Image"
[image2]: ./output_images/test1.png "Undistorted Road Image"
[image3]: ./output_images/binary.png "Binary Thresholded Image"
[image4]: ./output_images/warped_straight_lines.png "Warp Perspective"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/output.png "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view)

---
### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "P4.ipynb".  

I start by preparing the "object points", which will be the 3D (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, and that the object points are the same for each calibration image.
'objp' is just a replicated array of 3D coordinates.
'objpoints' will be appended with a copy of 'objp' each time I successfully detect all chessboard corners in a test image.
'imgpoints' will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the 'cv2.calibrateCamera()' function.
I applied this distortion correction to the test image using the 'cv2.undistort()' function and obtained this result: 

![alt text][image1]

### Pipeline (Single Images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of three filters to create my thresholded binary image (thresholding steps at cells 5, 6, 7).

1) My first filter is calibrated for detecting yellow lines. It is a combination of thresholds on the HLS color space.
Hue - (0, 45)
Saturation - (80, 255)
Lighting - (80, 255)
This Hue range covers between red and dull yellow colors. 
The Saturation and Lightness thresholds detects dull yellow lines while avoiding the gray parts of the street or the dark shadows.

2) My second filter is a simple threshold on the Lightness value for detecting white lines.

3) My third filter is a Sobel X-Axis gradient. However, I combine this gradient filter with basic Saturation and Lightness minimum threshold.
This filter detects distant yellow and white lines, which are important for determining the curvature of the lane.
The Saturation and Lightness thresholds prevent the filter from detecting spurious black lines.

Here is an example of my output for this step.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is found in cells 8,9,10 in the iPython Notebook.
I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 220, 720      | 300, 720      |
| 1110 720      | 1000, 720     |
| 690, 450      | 1000, 0       |

I verified that my perspective transform was working by drawing the `src` and `dst` points onto a test image and its warped counterpart.
The lines appeared parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for detecting the lane-line pixels is found in cells 8,9,10 in the iPython Notebook.
The function names are 'find_window_centroids', 'plot_window_centroids', 'curvature'.

1) Find_Window_Centroids - This function searchs for the lane lines using the convolution operator.
First, we find the starting positions for the left and right lanes. The bottom quarter of the image is summed vertically. 
Then, the left and right halves of the image are convolved with a mask of size 'window_width'.
The pixel with the maximum value is the center of the lane line.

Next, the image is divided into several vertical layers using the 'window height' parameter.
The layer is summed vertically and the covolutution operator is used to find the pixel with the largest value.
The search for the next layer's center is restricted to the 'margin' surround the previous layer's center.
This process is repeated for each layer in the image.

2) Plot_Window_Centroids - This function applies a mask to the binary image in order to select the lane line pixels around the window centroids.

3) Curvature - This function fits a second-order polynomial using the lane-line pixels detected around the window centroids.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this step in cell 13 in the iPython Notebook with a function called 'curvature'.

1. Radius of Curvature of the Lane
I calculate the curvature of the lane using the coefficients for the polynomial and the formula for the radius of curvature.
The pixel coordinates are multiplied by the 'xm_per_pix' and 'ym_per_pix' constants to convert the values to meters.
I take the average of the left and right curvature as the overall curvature of the lane.

2. Position of the vehicle with respect to the center
I measure the width of the lane using the window centroids of the bottom layer.
I use the left lane centoid as an offset and half of the lane width to find the center of the lane.
Then, I take the difference between the center of the image and the center of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 15 in the iPython Notebook with a function called 'draw'. 
Here is an example of my result on a test image:
![alt text][image6]

---

### Pipeline (video)

Here is [my result](./project_result.mp4) for the project video.
Here is [my result](./challenge_result.mp4) for the challenge video.

---

### Discussion
Briefly discuss any problems / issues you faced in your implementation of this project.

####1. Where will your pipeline likely fail?
My pipeline fails to detect sharp curves under difficult lighting conditions.
The pixels are too faint and distant, so my pipeline does not pick them up.
Since my pipeline fails to detect that part of the curve, there is a lag between the drawn lane area and the actual road.

####2. What could you do to make it more robust?
Perhaps, I should tune a pair of binary image functions for finding lane lines in very bright or dark conditions.

I implemented a few basic features to improve the robustness of my pipeline.
 + An average of 5-10 'good' images is used in order to generate the lane lines using the deque data structure.
 + A basic sanity check function to keep only the 'good' images. I focus on ensuring that lines are parallel with a simple width threshold.
