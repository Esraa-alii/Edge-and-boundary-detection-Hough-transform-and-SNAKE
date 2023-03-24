import streamlit as st
import Hough as hough
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import os
import edge_detection
import Ellipsed as Elp
import snake_countor as snake
import functions as fn
import numpy as np
import plotly.graph_objects as go

# vars
num_of_lines=0
image_path1=' '
option=0

path='images'
st.set_page_config(page_title="Image Processing",layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    
with st.sidebar:
    st.title('Upload an image')
    uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg','webp'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        plt.imread(uploaded_file)
        image_path1=os.path.join(path,uploaded_file.name)
        st.title("Options")
        option = st.selectbox("",["Canny edge detector","Line Detection","Ellipse Detection","Circle Detection","Snake"])
        if option=='Line Detection':
            num_of_lines = st.slider(label="Number of lines", min_value=1, max_value=300, step=1)
            T_low = st.slider(label="Low Threshold",min_value=1, max_value=400, step=10)
            T_high = st.slider(label="High Threshold",min_value=10, max_value=500, step=10)
            neighborhood_size=st.slider(label="Neighborhood size",min_value=1, max_value=200, step=5)
            line_color = st.selectbox("Lines color",["Red","Blue","Green"])

        if option=='Canny edge detector':
            canny_kernal = st.selectbox('Select kernal size',('3x3','5x5'))
            canny_sigma = st.number_input('Sigma', min_value=1, value=10, step=2)
        if option=='Ellipse Detection':
            Thickness = st.slider(label="Thickness", min_value=1, max_value=5, step=1)
            elipse_color = st.selectbox("Lines color",["Red","Blue","Green"])
        if option =='Circle Detection':
            r_min = st.slider(label="minimum radius", min_value=30, max_value=100, step=2)
            r_max = st.slider(label="maximum radius", min_value=50, max_value=250, step=5)
            bin_threshold = st.slider(label="bin threshold", min_value=0.1, max_value=1.0, step=0.1)
            pixel_threshold = st.slider(label="pixel threshold", min_value=10, max_value=100, step=5)
            circle_color = st.selectbox("Lines color",["Red","Blue","Green"])

        if option =='Snake':
            alpha = st.slider(label="Alpha", min_value=0.001, max_value=2.0, step=0.001)
            beta = st.slider(label="Beta", min_value=0.01, max_value=2.0, step=0.01)
            gamma = st.slider(label="Gamma", min_value=10, max_value=200, step=5)
            iterations=st.slider(label="Number of iterations", min_value=1, max_value=800, step=1)
            sigma = st.slider(label="Sigma", min_value=1, max_value=200, step=1)


input_img, resulted_img = st.columns(2)
with input_img :
    if uploaded_file is not None:
        st.title("Input images")
        image = Image.open(uploaded_file)
        st.image(uploaded_file)
    
with resulted_img:
    if uploaded_file is not None:
        st.title("Output")
    if option=='Line Detection':
        houghLine=hough.hough_lines(T_low,T_high,neighborhood_size,line_color,image,num_of_lines)
        st.image("images/output/hough_line.jpeg")

    if option =='Canny edge detector':
        image1=cv2.imread(image_path1,0)
        if canny_kernal == '3x3':
            st.image(edge_detection.canny_detector(image1, 3, canny_sigma))
        elif canny_kernal == '5x5':
            st.image(edge_detection.canny_detector(image1, 5, canny_sigma))
    if option == 'Ellipse Detection':
        Elp.edge_ellipsed_detector(uploaded_file,Thickness,elipse_color)
        st.image("./images/output/elp.jpeg")

    if option == 'Circle Detection':
        circle_img = hough.houghCircles(circle_color, image_path1, r_min , r_max)
        plt.imshow(circle_img)
        plt.axis('off')
        plt.savefig("./images/output/hough_circle.jpeg")   
        st.image("./images/output/hough_circle.jpeg")
    if option == "Snake":
        og_image = Image.open(image_path1)
        gray_image = ImageOps.grayscale(og_image)
        resized_gray_img = snake.resize_img(gray_image)
        np_image = np.array(gray_image)
        resized_np_image = np.array(resized_gray_img)
        area,perimeter=snake.activeContour(resized_np_image,alpha,beta,gamma,iterations,sigma)
        st.image("./images/output/snake.jpeg")
        st.write("Area = ",area)
        st.write("perimeter = ",perimeter)
        # st.subheader("Area = ",area)
        # with st.sidebar:
        #     center_x = st.slider("center x-coordinates",min_value=0,max_value=image1.shape[1])
        #     center_y = st.slider("center y-coordinates",min_value=0,max_value=image1.shape[0])
        #     radius = st.slider("radius",min_value=0,max_value=int(np.min(image1.shape)/2))
        #     num = st.slider("points number",min_value=0,max_value=100)
        #     circle= fn.circle_contour(center_x,center_y,radius,num)
        # perimeter,area= fn.calculations(circle)
        # print(perimeter,area)