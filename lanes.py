import cv2
import numpy as np

# loading the image, this returns a multidimensional array that contains all the relative intensities for each pixel
image = cv2.imread('media/test_image.png')
lane_image = np.copy(image)  # copying the image
grayscale_lane = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)  # converting the image to grayscale
blurred_lane = cv2.GaussianBlur(grayscale_lane, (5, 5), 0)  # reducing noise on the grayscale image
cv2.imshow('result', blurred_lane)  # render ing the image
cv2.waitKey(0)  # without that, the image would disappear really fast, 0 = infinite
