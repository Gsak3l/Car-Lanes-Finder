import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))  # this should be 422 in this case, starting point
    # y = mx + b => x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines_):
    left_fit = []
    right_fit = []
    for line in lines_:  # looping for each line
        x1, y1, x2, y2 = line.reshape(4)  # reshape into one array with one dimensional array with 4  elements
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # copied it from the internet, no idea how this works
        slope = parameters[0]  # slope value for each row
        intercept = parameters[1]  # intercepts for y axis
        if slope < 0:  # if the line has a negative slope value
            left_fit.append((slope, intercept))  # appending with the left list because it's lower than 0 thus it's left
        else:
            right_fit.append((slope, intercept))  # appending with the right list
    # getting the average from the slopes
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # calling the function for both left and right slope and intercept averages
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# makes the image black and white when there is a big color change
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # converting the image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # reducing noise on the grayscale image
    image_canny = cv2.Canny(blur, 50, 150)  # low and then high threshold, this works with the sharp edges in intensity,
    # more than 150 = light, less than 50 = dark
    return image_canny


# creates a polygon (or triangle) on top of the image, to specify where the road approximately is
def region_of_interest(image):
    height = image.shape[0]  # getting the height of the image
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)  # same amount of pixels, thus same dimensions the original image
    cv2.fillPoly(mask, polygons, 255)  # filling the mask with the white triangle
    masked_image = cv2.bitwise_and(image, mask)  # this isolates the region of interest in this specific case
    return masked_image


def display_lines(image, lines_):
    image_line = np.zeros_like(image)  # empty array with zeros
    if lines_ is not None:  # checking if we have any lines
        for line in lines_:  # printing all the lines
            x1, y1, x2, y2 = line.reshape(4)  # reshape into one array with one dimensional array with 4  elements
            # specifying the coordinates for where the lines have to be drawn
            cv2.line(image_line, (x1, y1), (x2, y2), (239, 255, 0), 10)  # color and line thickness, BRG NOT RGB
    return image_line


# loading the image, this returns a multidimensional array that contains all the relative intensities for each pixel
original_image = cv2.imread('media/test_image.png')
lane_image = np.copy(original_image)  # copying the image
canny_image = canny(lane_image)  # calling the function canny
cropped_image = region_of_interest(canny_image)  # calling the function region_of_interest
# precision in pixels, degree of precision, and threshold, empty array, length of a line in px, max distance in px
# long story short this took me about 1 hour and a half to understand, and detects lines in the image
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)  # calling the function
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # 0.8 and 1 is the weight of the image

cv2.imshow('result', combo_image)  # rendering the image
cv2.waitKey(0)  # without that, the image would disappear really fast, 0 = infinite, goes away when pressing a key

# plt.imshow(canny)
# plt.show()
