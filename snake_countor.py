import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from scipy import ndimage as filt



def Matrix_A(a, b, N):
    """
    a: float
    alpha parameter
    b: float
    beta parameter
    N: int
    N is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    """
    row = np.r_[
        -2*a - 6*b,
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N, N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A


def external_edge_force(img, sigma=30.):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.
    img: ndarray
        The image.
    """
    # Gaussian smoothing.
    smoothed = filt.gaussian_filter(
        (img-img.min()) / (img.max()-img.min()), sigma)

    # Gradient of the image in x and y directions.
    giy, gix = np.gradient(smoothed)
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient(gmi)

    def fx(x, y):
        """
        Return external edge force in the x direction.
        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1]-1] = img.shape[1]-1
        y[y > img.shape[0]-1] = img.shape[0]-1

        return ggmix[(y.round().astype(int), x.round().astype(int))]

    def fy(x, y):
        """
        Return external edge force in the y direction.
        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[x < 0] = 0.
        y[y < 0] = 0.

        x[x > img.shape[1]-1] = img.shape[1]-1
        y[y > img.shape[0]-1] = img.shape[0]-1

        return ggmiy[(y.round().astype(int), x.round().astype(int))]

    return fx, fy


def iterate_snake(x, y, a, b, fx, fy, gamma, n_iters, return_all=True):
    """
    x: ndarray
        intial x coordinates of the snake
    y: ndarray
        initial y coordinates of the snake
    a: float
        alpha parameter
    b: float
        beta parameter
    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.
    fy: callable
        see fx.
    gamma: float
        step size of the iteration
    n_iters: int
        number of times to iterate the snake
    return_all: bool
        if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    A = Matrix_A(a, b, x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x, y))
        y_ = np.dot(B, y + gamma*fy(x, y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append((x_.copy(), y_.copy()))

    if return_all:
        return snakes
    else:
        return (x, y)

    


def activeContour(img,alpha,beta,gamma,iterations,sigma):

    t = np.arange(0, 2*np.pi, 0.1)
    x = 153+105*np.cos(t)
    y = 156+120*np.sin(t)

   

    # fx and fy are callable functions
    fx, fy = external_edge_force(img, sigma)

    snakes = iterate_snake(
        x=x,
        y=y,
        a=alpha,
        b=beta,
        fx=fx,
        fy=fy,
        gamma=gamma,
        n_iters=iterations,
        return_all=True
    )
  


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0, 1, 0), lw=2)

    for i, snake in enumerate(snakes):

        if i % 10 == 0:
            ax.plot(np.r_[snake[0], snake[0][0]], np.r_[
                    snake[1], snake[1][0]], c=(0, 0, 1), lw=2)
        

    x=snake[0]
    y=snake[1]
    perimeter = 0
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))  
    for i in np.arange(len(x)-1):
        distance = np.sqrt(np.square(x[i+1]-x[i])+np.square(y[i+1]-y[i]))
        perimeter += distance

    # cnt = contours[0]
    # area1 = cv2.contourArea(x)  # Area of first contour
    # perimeter = cv2.arcLength(x, True)  # Perimeter of first contour 
    # area1 = cv2.contourArea(snake)
    # x = snake[:, 0]
    # y = snake[:, 1]
    # for i in np.arange(len(x)-1):
    #     # calculate the distance between the current point and the next point
    #     distance = np.sqrt(np.square(x[i+1]-x[i])+np.square(y[i+1]-y[i]))
    #     perimeter += distance
    # print("Detected Contour with Area: ",abs(area) )

    # Plot the last one a different color.
    ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]],
            np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1, 0, 0), lw=2)
    # plt.title("Active Contour")
    # plt.savefig('./output/active_contour.png', bbox_inches='tight')
    # print('active_contour.png saved successfully in output directory.')
    plt.savefig('images/output/snake.jpeg')
    # plt.show()
    return abs(area),perimeter

def resize_img(img: np.ndarray, basewidth: int = 300):
    w_percent = (basewidth/float(img.size[0]))
    h_size = int((float(img.size[1])*float(w_percent)))
    resized_img = img.resize((basewidth, h_size), Image.ANTIALIAS)
    return resized_img


