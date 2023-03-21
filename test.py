import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2

# Parameters For hand.png image
alpha = 20
beta = 0.01
gamma = 2
w_line = 1
w_edge = 8

num_xpoints = 180
num_ypoints = 180
num_iterations = 100

img = imread("hand.png",True)
image_src = np.copy(img)

contour_x, contour_y, WindowCoordinates = fn.create_square_contour(source=image_src,
                                                                num_xpoints=num_xpoints, num_ypoints=num_ypoints,x_position= 95, y_position=40)

ExternalEnergy = gamma * fn.calculate_external_energy(image_src, w_line, w_edge)
cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()
ax[0].set_title('Input image')
ax[0].imshow(img, cmap='gray')
ax[0].plot(np.r_[cont_x, cont_x[0]],
        np.r_[cont_y, cont_y[0]], c=(1, 1, 0), lw=2)
ax[0].set_axis_off()
perimeter = 0 
for iteration in range(num_iterations):
    cont_x, cont_y = fn.iterate_contour(source=image_src, contour_x=cont_x, contour_y=cont_y,
                                        external_energy=ExternalEnergy, window_coordinates=WindowCoordinates,
                                        alpha=alpha, beta=beta)
    result = []
    for j in range(len(cont_x)):
        result = cv2.circle(image_src,(int(cont_x[j]),int(cont_y[j])),3,[255,255,0],-1)
        if (j < len(cont_x) -1):
            dist = np.sqrt((cont_x[j] - cont_x[j+1]) **2  + (cont_y[j] - cont_y[j+1]) **2 )
            perimeter += dist
cv2.imwrite("result" + str(iteration) + ".jpg",result)

area =  fn.calculateAreaPerimeter(contour_x=cont_x, contour_y=cont_y)
print ("area = " ,area)
print ("perimeter = " ,perimeter)
ax[1].set_title('Output image')
ax[1].imshow(img, cmap='gray')
ax[1].plot(np.r_[cont_x, cont_x[0]],
        np.r_[cont_y, cont_y[0]], c=(1, 1, 0), lw=2)
ax[1].set_axis_off()

plt.tight_layout()
plt.show()