import numpy as np
import cv2
import copy
from obj_detection import displayImage

def createImage(image_contour, image, counter):
    rectangle = cv2.minAreaRect(image_contour)
    boxed = cv2.boxPoints(rectangle)
    boxed = np.int0(boxed)
    print(boxed)
    print("----------------")
    mins = (np.amin(boxed, axis=0))
    max = (np.amax(boxed, axis=0))
    """
    print(mins[0])
    print(mins[1])
    print(mins[0],max[0],
          mins[1],max[0],
          mins[0],max[1],
          mins[1],max[1],)
    """
    width = max[0] - mins[0]
    height = max[1] - mins[1]
    print('-------------')
    print(width, height)
    if (width > 10) & (height > 10):
        cropped_image = image[mins[1]:mins[1]+height, mins[0]:mins[0]+width]
        counter += 1
        print(counter)
        """
        try:
            displayImage(cropped_image)
        except:
            print('odd nematode')
        """
    return counter


nems = cv2.imread('/Users/lehassell/PycharmProjects/nematode/nems.jpg', 0)


img = copy.deepcopy(nems)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
thresh, th1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#test = contours[1]
#createImage(test, nems)

i=0

for cnt in contours:
    print('*******')
    print(i)
    i = createImage(cnt,nems, i)


#roi =nems[0:250, 0:250]
#displayImage(roi)
