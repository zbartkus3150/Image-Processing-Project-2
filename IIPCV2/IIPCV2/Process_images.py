import numpy as np
import cv2
import mahotas
import os
import glob
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from skimage import feature

import warnings
warnings.filterwarnings('ignore')

cv2.imshow("test", cv2.imread("../../isolated/negundo/l38.jpg"))
cv2.waitKey()

train_path = '../../isolated'
bins = 8
num_trees = 100
test_size = 0.20
seed = 9
scoring = "accuracy"

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray, ignore_zeros = True).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# feature-descriptor-4: Local Binary Pattern
def fd_loc_bin_patt(image, numPoints=24, radius=8, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

# feature-descriptor-5: Zernike Moments
def fd_zernike(image):
     return mahotas.features.zernike_moments(image, min(image.shape))

# feature-descriptor-6: SIFT
def fd_SIFT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #freakExtractor = cv2.xfeatures2d.FREAK_create()
    #sift = cv2.xfeatures2d.SIFT_create()
    #keypoints, descriptors = sift.detectAndCompute(image,None)
    #keypoints,descriptors= freakExtractor.compute(image,None)
    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    return descriptors

def SegmentImage(path, debug):

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

    return segmented, mask



train_labels = os.listdir(train_path)
train_labels.sort()

global_features = []
labels          = []


# loop over the training data sub-folders
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name

    for x in range(1,len([im for im in os.listdir(dir)])+1):
        # get the image file name
        if x < 10:
            file = dir + "/l0" +  str(x) + ".jpg"
        else:
            file = dir + "/l" +  str(x) + ".jpg"

        image = cv2.imread(file)
        image_seg, image_mask = SegmentImage(file, False)

        #fv_hu_moments = fd_hu_moments(image_mask)
        #fv_haralick   = fd_haralick(image_seg)
        fv_histogram  = fd_histogram(image, None)
        fv_loc_bin_patt = fd_loc_bin_patt(image, 24, 4)
        #fv_zernike     = fd_zernike(image_mask)
        #fv_SIFT       = fd_SIFT(image)
        

        global_feature = np.hstack([fv_histogram, fv_loc_bin_patt])
        #global_feature = np.hstack([fv_hu_moments])
        #global_feature = np.hstack([fv_haralick])
        #global_feature = np.hstack([fv_histogram])
        #global_feature = np.hstack([fv_loc_bin_patt])
        #global_feature = np.hstack([fv_zernike])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")


# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# encode the target labels
le          = LabelEncoder()
target      = le.fit_transform(labels)
# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

# split the training and testing data
#(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(rescaled_features),
#                                                                                          np.array(target),
#                                                                                          test_size=test_size,
#                                                                                          shuffle=True,
#                                                                                          random_state=seed)
                                                                                          
#print("Train data  : {}".format(trainDataGlobal.shape))
#print("Test data   : {}".format(testDataGlobal.shape))


clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(clf, rescaled_features, target, cv=kfold, scoring=scoring)
print(cv_results.mean())
