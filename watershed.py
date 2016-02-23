import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def displayImage(picture):
    cv2.imshow('ImageWindow', picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Reads in the files
filename = '/Users/lehassell/PycharmProjects/nematode/nems.jpg'
img = cv2.imread(filename, 0)

# First Attempt
# Blur and threshold.  Objs need to be white for contour to work properly
blurred = cv2.GaussianBlur(img, (5, 5), 0)
thresh, th1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7* dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

displayImage(sure_bg)
displayImage(unknown)