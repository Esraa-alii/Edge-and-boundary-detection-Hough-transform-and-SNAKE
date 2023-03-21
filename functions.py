import itertools
import numpy as np

def create_gaussian_kernel(kernel_size: int, std_dev: float):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * std_dev ** 2))
        * np.exp(
            (-1 * ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
            / (2 * std_dev ** 2)
        ),
        (kernel_size, kernel_size),
    )
    return kernel / np.sum(kernel)

def convolution(image, kernel):
    convolvedImage = np.zeros(image.shape)
    paddingHeight = int((len(kernel) - 1) / 2)
    paddingWidth = int((len(kernel[0]) - 1) / 2)

    padded_image = np.zeros(
        (len(image) + (2 * paddingHeight), len(image[0]) + (2 * paddingWidth))
    )

    padded_image[
        paddingHeight : padded_image.shape[0] - paddingHeight,
        paddingWidth : padded_image.shape[1] - paddingWidth,
    ] = image
    for row in range(len(image)):
        for col in range(len(image[0])):
            convolvedImage[row, col] = np.sum(
                kernel
                * padded_image[row : row + len(kernel), col : col + len(kernel[0])]
            )
    return convolvedImage

def gaussian_filter(image, size: int, std_dev: float):
    kernel = create_gaussian_kernel(size, std_dev)
    filtered_image = convolution(image, kernel)
    # filtered_image = filtered_image.astype(np.uint8)
    return filtered_image

def sobel(img):
    sobelHKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVKernel = np.flip(sobelHKernel.T)
    filteredImgH = convolution(img, sobelHKernel)
    filteredImgV = convolution(img, sobelVKernel)
    return np.sqrt(pow(filteredImgH, 2.0) + pow(filteredImgV, 2.0))

def calculateAreaPerimeter (contour_x: np.ndarray, contour_y: np.ndarray):
    area=0.5*np.sum(contour_y[:-1]*np.diff(contour_x) - contour_x[:-1]*np.diff(contour_y))
    return abs(area)

def iterate_contour(source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                    external_energy: np.ndarray, window_coordinates: list,
                    alpha: float, beta: float):

    src = np.copy(source)
    cont_x = np.copy(contour_x)
    cont_y = np.copy(contour_y)

    contour_points = len(cont_x)

    for Point in range(contour_points):
        MinEnergy = np.inf
        TotalEnergy = 0
        NewX = None
        NewY = None
        for Window in window_coordinates:
            CurrentX, CurrentY = np.copy(cont_x), np.copy(cont_y)
            CurrentX[Point] = CurrentX[Point] + Window[0] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
            CurrentY[Point] = CurrentY[Point] + Window[1] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

            try:
                TotalEnergy = - external_energy[CurrentY[Point], CurrentX[Point]] + calculate_internal_energy(CurrentX,
                                                                                                              CurrentY,
                                                                                                              alpha,
                                                                                                              beta)
            except:
                pass

            if TotalEnergy < MinEnergy:
                MinEnergy = TotalEnergy
                NewX = CurrentX[Point] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
                NewY = CurrentY[Point] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

        cont_x[Point] = NewX
        cont_y[Point] = NewY

    return cont_x, cont_y


def create_square_contour(source, num_xpoints, num_ypoints,x_position,y_position):
    step = 5

    t1_x = np.arange(0, num_xpoints, step)
    t2_x = np.repeat((num_xpoints) - step, num_xpoints // step)
    t3_x = np.flip(t1_x)
    t4_x = np.repeat(0, num_xpoints // step)

    t1_y = np.repeat(0, num_ypoints // step)
    t2_y = np.arange(0, num_ypoints, step)
    t3_y = np.repeat(num_ypoints - step, num_ypoints // step)
    t4_y = np.flip(t2_y)

    contour_x = np.array([t1_x, t2_x, t3_x, t4_x]).ravel()
    contour_y = np.array([t1_y, t2_y, t3_y, t4_y]).ravel()

    contour_x = contour_x + (source.shape[1] // 2) - x_position
    contour_y = contour_y + (source.shape[0] // 2) - y_position

    WindowCoordinates = GenerateWindowCoordinates(5)

    return contour_x, contour_y, WindowCoordinates

def GenerateWindowCoordinates(Size: int):
    Points = list(range(-Size // 2 + 1, Size // 2 + 1))
    PointsList = [Points, Points]

    coordinates = list(itertools.product(*PointsList))
    return coordinates

def calculate_internal_energy(CurrentX, CurrentY, alpha: float, beta: float):
    points_transpose = np.array((CurrentX, CurrentY))
    Points = points_transpose.T

    next_points = np.roll(Points, 1, axis=0)
    previous_points = np.roll(Points, -1, axis=0)
    displacement = Points - next_points
    point_distances = np.sqrt(displacement[:, 0] ** 2 + displacement[:, 1] ** 2)
    mean_distance = np.mean(point_distances)
    continuous_energy = np.sum((point_distances - mean_distance) ** 2)

    second_dervative = next_points - 2 * Points + previous_points
    curvature = (second_dervative[:, 0] ** 2 + second_dervative[:, 1] ** 2)
    curvature_energy = np.sum(curvature)

    return alpha * continuous_energy + beta * curvature_energy

def calculate_external_energy(source, WLine, WEdge):
    ELine = gaussian_filter(source,7,7)	
    EEdge = sobel(ELine)
    return WLine * ELine + WEdge * EEdge