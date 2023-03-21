import cv2
import numpy as np
import matplotlib.pyplot as plt



def edge_ellipsed_detector(input,thickness,color):
    
    img = plt.imread(input)   
    img = cv2.resize(img,(512,512))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # ret , threshold = cv2.threshold(imgray,20,255,0)
    edges = cv2.Canny(imgray,100, 200)

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # print("Number of contours = "+ str(len(contours)))
    # print(contours[0])
    if color == 'Red':
        cv2.drawContours(img,contours,-1,(255, 0, 0),thickness)
    elif color == 'Blue':
        cv2.drawContours(img,contours,-1,(0,0,205),thickness)
    elif color == 'Green':
        cv2.drawContours(img,contours,-1,(50,205,50),thickness)

    plt.imshow(img)
    plt.axis("off")
    plt.savefig('./images/output/elp.jpeg')
    