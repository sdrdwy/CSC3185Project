
#用于图像分割

import numpy as np
import cv2
import VesselExtraction as VE

def find_optic_disc(imgpath, radius=15):
    image=cv2.imread(imgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.rectangle(image, (maxLoc[0]-radius,maxLoc[1]-radius),(maxLoc[0]+radius,maxLoc[1]+radius), (255, 0, 0), 2)
    return image

def find_Vessels(imgpath):
    return VE.VesselExtract(imgpath)

def find_macula(imgpath, radius=15):
    image=cv2.imread(imgpath)
    im1=image.copy()
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([10, 10, 10], dtype=np.uint8)
    black_mask = cv2.inRange(image, lower_black, upper_black)
    replace_color = (0, 255, 0)  
    image[black_mask > 0] = replace_color
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.rectangle(im1, (minLoc[0]-radius,minLoc[1]-radius),(minLoc[0]+radius,minLoc[1]+radius), (255, 0, 0), 2)
    return im1

