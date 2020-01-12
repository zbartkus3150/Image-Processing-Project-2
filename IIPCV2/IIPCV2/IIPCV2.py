import numpy as np
import cv2
from math import copysign, log10


# Load the sample image
path1 = '../../isolated/circinatum/l01.jpg'
path3 = '../../isolated/circinatum/l02.jpg'
path2 = '../../isolated/garryana/l01.jpg'
#path = '../../isolated/glabrum/l01.jpg'
#path = '../../isolated/kelloggii/l01.jpg'
#path = '../../isolated/macrophyllum/l02.jpg'
#path = '../../isolated/negundo/l02.jpg'

def makeMask(path, debug):

    image = cv2.imread(path,cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV) 
    low = np.array([0, 0, 0],np.uint8) 
    high = np.array([180, 255, 165],np.uint8)
    mask = cv2.inRange(hsv, low , high)

    ret,thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1], 1), np.uint8), [cont], 0, (255,255,255), cv2.FILLED)

    segmented = cv2.bitwise_and(image , image , mask=mask) 
    ret, mask = cv2.threshold(mask, 127, 255, 0)

    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("segmented", segmented)
        cv2.waitKey()

    return mask, segmented

def compare(path1, path2):
    mask1, bin = makeMask(path1, False)
    mask2, bin = makeMask(path2, False)

    moments1 = cv2.moments(mask1)
    huMoments1 = cv2.HuMoments(moments1)
    moments2 = cv2.moments(mask2)
    huMoments2 = cv2.HuMoments(moments2)

    for i in range(0,7):
        huMoments1[i] = -1* copysign(1.0, huMoments1[i]) * log10(abs(huMoments1[i]))
        huMoments2[i] = -1* copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]))

    d2 = cv2.matchShapes(mask1,mask2,cv2.CONTOURS_MATCH_I2,0)
    print(d2)

compare(path1, path2)
compare(path1, path3)

