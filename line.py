import numpy as np
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, N=10):
        # was the line detected in the last iteration?
        self.detected = False  
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0,0,0], dtype='float')

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        # polynomial coefficients averaged over the last n fits of the line
        self.recent_fit = deque(maxlen=N)

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  

        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=N)

        #average x values of the fitted line over the last n iterations
        self.bestx = np.mean(self.recent_xfitted)

        #radius of curvature of the line in some units
        self.radius_of_curvature = None 

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

    def curvature(self, pts):
        y, x = pts
        # line completely not detected
        if len(x) == 0:
            return False, []
    
        # Fit a second order polynomial to pixel positions in each fake lane line
        return True, np.polyfit(y, x, 2)

    def avg(self, height):
        if len(self.recent_fit) == 0:
            return False, [], 0

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.mean(self.recent_fit, axis=0)

        ploty = np.linspace(0, height-1, num=height)
        # Calculate x values for current polynomial coefficients
        fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]

        #average x values of the fitted line over the last n iterations
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        return True, self.bestx, self.radius_of_curvature

    def update(self, pts, fit, height):
        ploty = np.linspace(0, height-1, num=height)
        y_eval = np.max(ploty)
        y, x = pts

        self.diffs = fit - self.current_fit
        self.current_fit = fit

        # polynomial coefficients averaged over the last n fits of the line
        self.recent_fit.append(self.current_fit)

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.mean(self.recent_fit, axis=0)

        # Calculate x values for current polynomial coefficients
        fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]

        # x values of the last n fits of the line
        self.recent_xfitted.append(fitx)

        #average x values of the fitted line over the last n iterations
        self.bestx = np.mean(self.recent_xfitted, axis=0)
    
        pts = np.array([(y, x) for y,x in zip(self.bestx, ploty)], np.int32)
        curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
        #print(curverad, right_curverad)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(np.array(y)*ym_per_pix, np.array(x)*xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        # Now our radius of curvature is in meters
        #print(curverad_real, 'm')
        return self.bestx, self.radius_of_curvature
