# coding: utf-8

# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    imshape = img.shape
    vertices = np.array([[(200,imshape[0]),(imshape[1]/2-50, imshape[0]/2+85), (imshape[1]/2+100, imshape[0]/2+85), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def threshold(img):
    """
        Use color transforms, gradients, etc., to create a thresholded binary image.
    """    
    img_cpy = np.copy(img)
    g_channel = img_cpy[:,:,1]
    r_channel = img_cpy[:,:,2]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    # Threshold Hue channel - For Yellow Lines
    yellow_h_thresh=(0, 45)
    yellow_s_thresh = (80, 255)
    yellow_l_thresh = (100, 255)
    yellow_binary = np.zeros_like(h_channel)
    yellow_binary[(s_channel >= yellow_s_thresh[0]) & (s_channel <= yellow_s_thresh[1])
             & (l_channel >= yellow_l_thresh[0]) & (l_channel <= yellow_l_thresh[1]) 
             & (h_channel >= yellow_h_thresh[0]) & (h_channel <= yellow_h_thresh[1])] = 1
    
    # Threshold Lightness channel - For White Lines
    l_thresh=(190, 255)
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Threshold x gradient - Distant Lines - Poor Color
    sobel_kernel=3
    sx_thresh=(20, 95)
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)    
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))    
    sx_binary = np.zeros_like(scaled_sobelx)
    sx_binary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack((sx_binary, l_binary, yellow_binary))
    binary_img = np.zeros(color_binary.shape[:-1])
    binary_img[(sx_binary == 1) | (yellow_binary == 1) | (l_binary == 1)] = 1
    return region_of_interest(binary_img), color_binary

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        l_max = np.amax(conv_signal[l_min_index:l_max_index])

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        r_max = np.amax(conv_signal[r_min_index:r_max_index])
        
        l_center = -1 if l_max == 0 else l_center
        r_center = -1 if r_max == 0 else r_center
        
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
    return window_centroids

def plot_window_centroids(img, window_centroids, window_width, window_height):
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            left, right = window_centroids[level]
            # Window_mask is a function to draw window areas
            # Add graphic points from window mask here to total pixels found
            if left > 0:
                l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if right > 0:
                r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel 
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    else:
        # If no window centers found, just display orginal road image
        output = np.array(cv2.merge((img,img,img)),np.uint8)
    return output, np.nonzero(l_points), np.nonzero(r_points)


def curvature(img, left, right):
    ploty = np.linspace(0, 719, num=720)
    lefty, leftx = left
    righty, rightx = right
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_pts = np.array([(y, x) for y,x in zip(left_fitx, ploty)], np.int32)
    right_pts = np.array([(y, x) for y,x in zip(right_fitx, ploty)], np.int32)
    #cv2.polylines(img, [left_pts], False, (255, 0, 0), thickness=10)
    #cv2.polylines(img, [right_pts], False, (255, 0, 0), thickness=10)

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(np.array(lefty)*ym_per_pix, np.array(leftx)*xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.array(righty)*ym_per_pix, np.array(rightx)*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad_real = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_real = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    #print(left_curverad_real, 'm', right_curverad_real, 'm')
    return img, left_fitx, right_fitx

def draw(img, left_fitx, right_fitx, Minv, width, height):
    # Create an image to draw the lines on
    ploty = np.linspace(0, height-1, num=height)
    img_zero = np.zeros_like(img).astype(np.uint8)

    # Draw the lane onto the warped blank image
    #cv2.polylines(img_zero, [pts_left], False, (255, 0, 0), thickness=25)
    #cv2.polylines(img_zero, [pts_right], False, (255, 0, 0), thickness=25)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img_zero, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(img_zero, Minv, (width, height))
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result
