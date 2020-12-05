import cv2
import numpy as np
import matplotlib.pyplot as plt


# makes the image black and white when there is a big color change
def canny(image_canny):
    gray = cv2.cvtColor(image_canny, cv2.COLOR_RGB2GRAY)  # converting the image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # reducing noise on the grayscale image
    image_canny = cv2.Canny(blur, 50, 150)  # low and then high threshold, this works with the sharp edges in intensity,
    # more than 150 = light, less than 50 = dark
    return image_canny


# creates a polygon (or triangle) on top of the image, to specify where the road approximately is
def region_of_interest(image_interest):
    height = image_interest.shape[0]  # getting the height of the imge
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image_interest)  # same amount of pixels, thus same dimensions the original image
    cv2.fillPoly(mask, polygons, 255)  # filling the mask with the white triangle
    masked_image = cv2.bitwise_and(image_interest, mask)  # this isolates the region of interest in this specific case
    return masked_image


# loading the image, this returns a multidimensional array that contains all the relative intensities for each pixel
image = cv2.imread('media/test_image.png')
lane_image = np.copy(image)  # copying the image
canny = canny(lane_image)  # calling the function canny

cropped_image = region_of_interest(canny)

cv2.imshow('result', cropped_image)  # render ing the image
cv2.waitKey(0)  # without that, the image would disappear really fast, 0 = infinite

# plt.imshow(canny)
# plt.show()
