import streamlit as st
from skimage.io import imread, imsave
import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
from skimage.util import img_as_ubyte
col1, col2 = st.columns(2)
with st.sidebar:
    image_input = st.file_uploader("Input Image",type=["jpg","png","jpeg"])
if image_input is not None:
    image = imread(image_input,True) # reading image in garyscale
    with col1:
        st.header("Input")
        st.image(image) # showing the grayscale image
    with st.sidebar:
        alpha = st.slider(label = "alpha", min_value = 0.0, max_value = 10.0, step = 0.01)
        beta = st.slider(label = "beta", min_value = 0.0, max_value = 10.0, step = 0.01)
        gamma = st.slider(label = "gamma", min_value = 0.0, max_value = 10.0, step = 0.01)
        w_line = st.slider(label = "line weight", min_value = 0.0, max_value = 10.0, step = 0.01)
        w_edge = st.slider(label = "edge weight", min_value = 0.0, max_value = 10.0, step = 0.01)
        num_iteration = st.slider(label = "maximum iteration number", min_value = 0, max_value = 100, step = 1)
    num_xpoints = 180
    num_ypoints = 180
    # num_iterations = 100

    contour_x, contour_y, WindowCoordinates = fn.create_square_contour(source=image,
                                                                    num_xpoints=num_xpoints, num_ypoints=num_ypoints,x_position= 95, y_position=40)

    ExternalEnergy = gamma * fn.calculate_external_energy(image, w_line, w_edge)
    cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].set_title('Input image')
    ax[0].imshow(image, cmap='gray')
    ax[0].plot(np.r_[cont_x, cont_x[0]],
            np.r_[cont_y, cont_y[0]], c=(1, 1, 0), lw=2)
    ax[0].set_axis_off()
    perimeter = 0
    iteration = 0
    result = []

    for iteration in range(num_iteration):
        cont_x, cont_y = fn.iterate_contour(source=image, contour_x=cont_x, contour_y=cont_y,
                                            external_energy=ExternalEnergy, window_coordinates=WindowCoordinates,
                                            alpha=alpha, beta=beta)
        result = []
        for j in range(len(cont_x)):
            result = cv2.circle(image,(int(cont_x[j]),int(cont_y[j])),3,[255,255,0],-1)
            if (j < len(cont_x) -1):
                dist = np.sqrt((cont_x[j] - cont_x[j+1]) **2  + (cont_y[j] - cont_y[j+1]) **2 )
                perimeter += dist
    result = img_as_ubyte(result)

    imsave("result.jpg",result)

    area =  fn.calculateAreaPerimeter(contour_x=cont_x, contour_y=cont_y)
    print ("area = " ,area)
    print ("perimeter = " ,perimeter)
    ax[1].set_title('Output image')
    ax[1].imshow(image, cmap='gray')
    ax[1].plot(np.r_[cont_x, cont_x[0]],
            np.r_[cont_y, cont_y[0]], c=(1, 1, 0), lw=2)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.savefig("output.png")
    st.image("output.png")