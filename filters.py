import random
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def apply_Gaussian_Noise(img,mean,sigma):
  img_width,img_height=img.shape
  gauss_noise=np.zeros((img_width,img_height),dtype=np.uint8)
  cv2.randn(gauss_noise,mean,sigma)
  gauss_noise=(gauss_noise*0.5).astype(np.uint8)
  gauss_noise_img=cv2.add(img,gauss_noise)
  plt.imshow(gauss_noise_img,cmap='gray')
  plt.axis("off")
  plt.savefig("images/output/Gaussian_noise.jpeg")


def Apply_uniform_noise(img,noise_value):
  # we create a uniform distribution whose lower and upper bounds are the minimum and maximum pixel values (0 and 255 respectively) along the dimensions of the image.
  img_width,img_height=img.shape
  uni_noise=np.zeros((img_width,img_height),dtype=np.uint8)
  cv2.randu(uni_noise,0,255)
  uni_noise=(uni_noise*noise_value).astype(np.uint8)
  un_img=cv2.add(img,uni_noise)
  plt.imshow(un_img,cmap='gray')
  plt.axis("off")
  plt.savefig("images/output/Uniform_noise.jpeg")

def Apply_salt_and_papper_noise(img,num_of_white_PX,num_of_black_PX):
     # Getting the dimensions of the image
    row , col = img.shape  
    # Randomly pick some pixels in the image for coloring them white
    number_of_pixels = random.randint(0, num_of_white_PX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_axis=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_axis=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_axis][x_axis] = 255
          
    # Randomly pick some pixels in the image for coloring them black
    number_of_pixels = random.randint(0 , num_of_black_PX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_axis=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_axis=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_axis][x_axis] = 0
    plt.imshow(img,cmap='gray')
    plt.axis("off")
    plt.savefig("images/output/Salt & pepper_noise.jpeg")
    
    # median filter


def apply_average_filter(img,kernal_size):
  image_width, image_height = img.shape

  # Develop Averaging filter mask
  mask = np.ones([kernal_size, kernal_size], dtype = int)
  mask = mask / (kernal_size*kernal_size)

  # Convolve  mask over the image
  img_new = np.zeros([image_width, image_height])

  for i in range(1, image_width-1):
    for j in range(1, image_height-1):
      temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
      
      img_new[i, j]= temp
      
  img_new = img_new.astype(np.uint8)
  plt.axis('off')
  plt.imshow(img_new, cmap="gray")
  plt.savefig("images/output/average_filter.jpeg")

def apply_median_filter(img):
# Obtain the number of rows and columns 
# of the image
    img_width, img_height = img.shape
    
    # Traverse the image. For every 3X3 area, 
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new1 = np.zeros([img_width, img_height])
    
    for i in range(1, img_width-1):
        for j in range(1, img_height-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            
            temp = sorted(temp)
            img_new1[i, j]= temp[4]
    
    img_new1 = img_new1.astype(np.uint8)
    plt.imshow(img_new1, cmap="gray")
    plt.axis('off')   
    plt.savefig("images/output/Median_filter.jpeg") 


def apply_convolution(img_grayscale, mask):
    """a function that performs the convolution of the gaussian filter + the input image 
    Args:
        img_grayscale (array): the input image 
        mask (array): the mask(kernel)
    Returns:
        filtered_img
    """
    row, col = img_grayscale.shape
    masked_row, masked_col = mask.shape
    new = np.zeros((row + masked_row - 1, col + masked_col - 1))
    # setting the boundries of the image array
    masked_col = masked_col//2
    masked_row = masked_row//2
    filtered_img = np.zeros(img_grayscale.shape)
    new[masked_row:new.shape[0]-masked_row, masked_col:new.shape[1]-masked_col] = img_grayscale
    # looping over the image row indices
    for i in range(masked_row, new.shape[0]-masked_row):

        # looping over the image coloumn indices
        for j in range(masked_col, new.shape[1]-masked_col):
            temp = new[i-masked_row:i+masked_row+1, j-masked_row:j+masked_row+1]
            result = temp*mask
            filtered_img[i-masked_row, j-masked_col] = result.sum()

    return filtered_img

def gaussian_kernal(width, height, sigma):
    """_summary_
    Args:
        width : rows
        height : columns
        sigma: the standard deviation 
    Returns:
        gaussian: the filter array
    """
    # empty array
    gaussian = np.zeros((width, height))
    # setting the boundries of the filter
    width = width//2
    height = height//2
    # looping over rows
    for x in range(-width, width + 1):
 
       
        for y in range(-height, height + 1):
            # applying the equation of gaussian
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+width, y+height] = (1/x1)*x2

    return gaussian

def Apply_gaussian_filter(image_path: str ,sigma):
    """ function that calls the gaussian_filter and correlation functions and does some conversions
    Args:
        image_path (str): input image
    Returns:
        gaussian_filtered_img:
    """
    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)

    # kernal size 9x9
    kernal = gaussian_kernal(9, 9, sigma)
    after_convolution = apply_convolution(img_grayscale, kernal)

    gaussian_filtered_img = after_convolution.astype(np.uint8)

   
    plt.imshow(gaussian_filtered_img,cmap='gray')    
    plt.axis('off') 
    plt.savefig("images/output/Gaussian_filter.jpeg")              
    # return gaussian_filtered_img

