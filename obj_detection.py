import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def displayImage(picture):
    cv2.imshow('ImageWindow', picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_image(p1, p2, p3, title1, title2, title3):
    plt.figure(1)
    plt.subplot(311), plt.imshow(p1, cmap=cm.Greys_r), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(p2, cmap=cm.Greys_r), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(p3, cmap=cm.Greys_r), plt.title(title3)
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # Reads in the files
    filename = '/Users/lehassell/PycharmProjects/nematode/nems.jpg'
    original = cv2.imread(filename, 0)

    img = copy.deepcopy(original)

    # First Attempt
    # Blur and threshold.  Objs need to be white for contour to work properly
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh, th1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #displayImage(blurred)
    #displayImage(th1)


    thresh_3 = copy.deepcopy(th1)
    thresh_6 = copy.deepcopy(th1)
    thresh_10 = copy.deepcopy(th1)

    kernel = np.ones((9,9), np.uint8)
    dilation = cv2.dilate(th1, kernel, iterations=1)
    displayImage(dilation)

    kernel_close = np.ones((2,2), np.uint8)
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel_close)
    displayImage(closing)

    """
    # Note 15- 20 looks best
    kernel_3 = np.ones((15, 15), np.uint8)
    kernel_6 = np.ones((17, 17), np.uint8)
    kernel_10 = np.ones((20, 20), np.uint8)

    opening_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_CLOSE, kernel_3)
    opening_6 = cv2.morphologyEx(thresh_6, cv2.MORPH_CLOSE, kernel_6)
    opening_10 = cv2.morphologyEx(thresh_10, cv2.MORPH_CLOSE, kernel_10)

    # Too much overlap in images gonna try to erode
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_6 = np.ones((5, 5), np.uint8)
    kernel_10 = np.ones((7, 7), np.uint8)

    opening_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_CLOSE, kernel_3)
    opening_6 = cv2.morphologyEx(thresh_6, cv2.MORPH_CLOSE, kernel_6)
    opening_10 = cv2.morphologyEx(thresh_10, cv2.MORPH_CLOSE, kernel_10)
    """

    #compare_image(opening_3, opening_6, opening_10, '12', '15', '18')


    # Get the contours
    image, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    i = 0
    print(len(contours))

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        circle_image = cv2.circle(img, center, radius, (0, 255, 0), 5)

    displayImage(circle_image)

    """
    img_2 = copy.deepcopy(original)
    cnt = contours[22]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    circle_image_illustrate = cv2.circle(img_2, center, radius, (0, 255, 0), 5)
    print(radius)
    displayImage(circle_image_illustrate)
    # Problems - lots of little circles, increase blurring
    #            Improper bounding, increase radius?
    #            Detect the boundaries

    # Attempt to look at blurring
    img_3 = copy.deepcopy(original)
    img_4 = copy.deepcopy(original)
    img_5 = copy.deepcopy(original)

    gauss_blur = cv2.GaussianBlur(img_3, (5, 5), 0)
    median_blur = cv2.medianBlur(img_4, 5)
    bilateral_blur = cv2.bilateralFilter(img_5, 9, 75, 75)
    compare_image(gauss_blur, median_blur, bilateral_blur, 'Gaussian', 'Median', 'Bilateral')

    # Attempt to look at improper bounding
    img_6 = copy.deepcopy(original)

    # Still seems to be lacking
    for cnt2 in contours:
        (x2, y2), radius2 = cv2.minEnclosingCircle(cnt2)
        center2 = (int(x2), int(y2))
        radius2 = int(radius2)
        if radius2 > 15:
            circle_image_2 = cv2.circle(img_6, center2, radius2, (0, 255, 0), 5)

    displayImage(circle_image_2)

    # Attempt to look at improper bounding
    img_7 = copy.deepcopy(original)

    # Still seems to be lacking
    for cnt3 in contours:
        rect = cv2.minAreaRect(cnt3)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        square_image = cv2.drawContours(img_7, [box], 0, (0, 255, 0), 5)
    displayImage(square_image)
"""