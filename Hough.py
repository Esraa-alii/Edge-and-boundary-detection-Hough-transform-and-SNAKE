import numpy as np
import cv2 as cv
import math
from PIL import Image
import matplotlib.pyplot as plt
# from libs import helper
from collections import defaultdict


def hough_peaks(H, peaks, neighborhood_size=3):
    """Calculate the indices of the peaks.
    Args:
        H: the accumulator
        peaks: number of line peaks.
        neighborhood_size (int, optional): the size of the region to detect 1 line within. Defaults to 3.
    Returns:
        indices (np.ndarray): the indices of the peaks.
        H: accumulator that holds only the line points.
    """
    
    indices = []
    H1 = np.copy(H)
    
    # loop through number of peaks to identify
    for i in range(peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H to be 2d array
        indices.append(H1_idx)

        idx_y, idx_x = H1_idx  # separate x, y indices from argmax(H)
        
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (neighborhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (neighborhood_size / 2)
        if (idx_x + (neighborhood_size / 2) + 1) > H.shape[1]:
            max_x = H.shape[1]
        else:
            max_x = idx_x + (neighborhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (neighborhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (neighborhood_size / 2)
        if (idx_y + (neighborhood_size / 2) + 1) > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (neighborhood_size / 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if x == min_x or x == (max_x - 1):
                    H[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    H[y, x] = 255

    # return the indices and the original Hough space with selected points
    return indices, H

def hough_lines_draw(color,img, indices, rhos, thetas):
    """Draw lines according to specific rho and theta
    Args:
        img (np.ndarray): image to draw on
        indices: indices of the peaks points
        rhos: norm distances of each line from origin
        thetas: the angles between the norms and the horizontal x axis
    """ 

    for i in range(len(indices)):
        # get lines from rhos and thetas
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # these are then scaled so that the lines go off the edges of the image
        y1 = int(y0 + 1000 * (a))
        x1 = int(x0 + 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        if color =='Red':
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif color =='Blue':
            cv.line(img, (x1, y1), (x2, y2), (0,0,205), 2)
        elif color == 'Green':
            cv.line(img, (x1, y1), (x2, y2), (50,205,50), 2)



def line_detection(image: np.ndarray,T_low,T_upper):
    """Fucntion that detects lines in hough domain
    Args:
        image (np.ndarray())
    Returns:
        accumulator: hough domain curves
        rhos: norm distances of each line from origin
        thetas: the angles between the norms and the horizontal x axis
    """
    
    grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurImg = cv.GaussianBlur(grayImg, (5,5), 1.5)
    edgeImg = cv.Canny(blurImg, T_low, T_upper)
    

    height, width = edgeImg.shape
    
    maxDist = int(np.around(np.sqrt(height**2 + width**2)))
    
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.linspace(-maxDist, maxDist, 2*maxDist)
    
    accumulator = np.zeros((2 * maxDist, len(thetas)))
    
    for y in range(height):
        for x in range(width):
            if edgeImg[y,x] > 0:
                for k in range(len(thetas)):
                    r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + maxDist, k] += 1
                    
    return accumulator, thetas, rhos


def hough_lines(T_low,T_high,neighborhood_size,color,source: np.ndarray,peaks):
    """detect lines and draw them on the image
    Args:
        source (np.ndarray): image
        peaks (int, optional): number of line peaks. Defaults to 10.
    Returns:
        image: image with detected lines
    """
    
    src = np.copy(source)
    H, thetas, rhos = line_detection(src,T_low,T_high)
    indicies, H = hough_peaks(H, peaks, neighborhood_size) 
    hough_lines_draw(color,src, indicies, rhos, thetas)
    plt.imshow(src)
    plt.axis("off")
    plt.savefig("images/output/hough_line.jpeg")   
    return src



def houghCircles(circle_color,img_path:str, r_min:int = 20, r_max:int = 100, delta_r:int = 1, num_thetas:int = 100, bin_threshold:float = 0.4, min_edge_threshold:int = 100, max_edge_threshold:int = 200, pixel_threshold:int = 20,  post_process:bool = True):
    
    """Function that detects circles in haugh domain.
    Args:
        img_path: image path.
        r_min: minimum detected circles radius.
        r_max maximum detected circles radius.
        delta_r: the step to arrange the radius values.
        nun_thetas: the step to arrange the thetas values.
        bin_threshold: the threshold that decide whether takes a bin or not.
        min_edge_threshold: canny edge detection min threshold constant.
        max_edge_threshold: canny edge detection max threshold constant.
        pixel_threshold: filltering the duplicate drwan circles pixel threshold
    Returns:
        output_img: (np.ndarray): the output image where circles are drwan at the edges of circles shapes.
        
    """
    
    input_img = cv.imread(img_path)

    #Edge detection on the input image
    edge_image = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    #ret, edge_image = cv.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    edge_image = cv.Canny(edge_image, min_edge_threshold, max_edge_threshold)

    if edge_image is None:
        print ("Error in input image!")
        return

    #image size
    img_height, img_width = edge_image.shape[:2]
    
    # R and Theta ranges
    dtheta = int(360 / num_thetas)
    
    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)
    
    ## Radius ranges from r_min to r_max 
    rs = np.arange(r_min, r_max, step=delta_r)
    
    circle_candidates = condidateCircles(thetas, rs, num_thetas)

    accumulator = calculateAccumlator(img_height, img_width, edge_image, circle_candidates)
    
    # Output image with detected lines drawn
    output_img = input_img.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
    out_circles = []
    
    # Sort the accumulator based on the votes for the candidate circles 
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))
    
    # Post process the results, can add more post processing later.
    if post_process :
        out_circles = postProcces(out_circles, pixel_threshold)
        
    # Draw shortlisted circles on the output image
    for x, y, r, v in out_circles:
        if circle_color =='Red':
            output_img = cv.circle(output_img, (x,y), r, (255,0,0), 2)
        elif circle_color =='Blue':
           output_img = cv.circle(output_img, (x,y), r, (0,0,205), 2)
        elif circle_color == 'Green':
            output_img = cv.circle(output_img, (x,y), r, (50,205,50), 2)
    
    return output_img


def calculateAccumlator(img_height, img_width, edge_image, circle_candidates):
    # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
    # aready present in the dictionary instead of throwing exception.
    accumulator = defaultdict(int)
    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
                # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
    return accumulator

def condidateCircles(thetas, rs, num_thetas):
    # Evaluate and keep ready the candidate circles dx and dy for different delta radius
    # based on the the parametric equation of circle.
    # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
    # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)

    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    circle_candidates = []

    for r in rs:
        for t in range(num_thetas):
            #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
            #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
            #but its better to pre-calculate and use it here.
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    return circle_candidates

def postProcces(out_circles, pixel_threshold):
    postprocess_circles = []
    for x, y, r, v in out_circles:
      # Exclude circles that are too close of each other
      # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    return postprocess_circles

