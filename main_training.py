import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
import random
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import grid_search
from skimage.feature import hog
from functionset import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

def feature_vector_creation(spatial_feat=True, hist_feat = True, hog_feat = True,
                            sample_size=1000, color_space = 'RGB',
                            model_name = "trained.pickle", ):
    '''
    Create svm model to detect car and non-car images.
    sample size and method to create feature vector from color space are optional.
    model will be saved as "model_name" with pickle module.
    score of verification will be returned.
    :param spatial_feat: Spatial features on or off
    :param hist_feat: Histogram features on or off
    :param hog_feat: HOG features on or off
    :param sample_size: sample size to create future vector
    :param color_space: color space for traning Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    :param filename_car: file name of training data
    :return: verification score of model
    '''

    # Read in cars and notcars
    images_car_far = glob.glob('./training_data/vehicles/vehicles/GTI_Far/*.png')
    images_car_left = glob.glob('./training_data/vehicles/vehicles/GTI_Left/*.png')
    images_car_MiddleClose = glob.glob('./training_data/vehicles/vehicles/GTI_MiddleClose/*.png')
    images_car_right = glob.glob('./training_data/vehicles/vehicles/GTI_Right/*.png')
    images_car_KITTI_extracted = glob.glob('./training_data/vehicles/vehicles\KITTI_extracted/*.png')
    images_noncar_Extras = glob.glob('./training_data/non-vehicles/non-vehicles/Extras/*.png')
    images_noncar_GIT = glob.glob('./training_data/non-vehicles/non-vehicles/GTI/*.png')
    cars = images_car_far + images_car_left  + images_car_MiddleClose + images_car_right + images_car_KITTI_extracted
           # + images_car_right + images_car_KITTI_extracted
    notcars = images_noncar_Extras + images_noncar_GIT

    #Shuffle original data
    random.shuffle(cars) # length is 8792
    random.shuffle(notcars)  #length is 8968

    # Reduce the sample size
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    # Define input vector
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Use a linear SVC
    t = time.time()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # svc = LinearSVC()
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    # Check the training time for the SVC
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC

    svc = LinearSVC(C=clf.best_params_['C'])
    svc.fit(X_train, y_train)
    score_model = round(svc.score(X_test, y_test),4)
    print('Test Accuracy of SVC = ', score_model)

    output_sum = {'model':svc,'scaler':X_scaler,'orient':9,'pix_per_cell':8,
                  'cell_per_block':2,'hog_channel':0,'spatial_size':(16, 16),'hist_bins':16}

    with open(model_name, 'wb') as handle:
        pickle.dump(output_sum, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Check the prediction time for a single sample
    t = time.time()
    return score_model

if __name__ == "__main__":
    # execute only if run as a script
    #import os
    #os.chdir("C:/Users/hitoshi/AppData/Local/Programs/Python/" +
    #         "Python35/Scripts/Udacity/Udacity_CarND_P5_Vehicle_Detection")
    # Parameter study
    results = []
    results.append(feature_vector_creation(
        spatial_feat=True, hist_feat = True, hog_feat = True, sample_size=1000, color_space = 'RGB',
        model_name = "condition_1.pickle"))
    results.append(feature_vector_creation(
        spatial_feat=True, hist_feat = False, hog_feat = True, sample_size=1000, color_space = 'RGB',
        model_name = "condition_2.pickle"))
    results.append(feature_vector_creation(
        spatial_feat=False, hist_feat = False, hog_feat = True, sample_size=1000, color_space = 'RGB',
        model_name = "condition_3.pickle"))
    results.append(feature_vector_creation(
        spatial_feat=True, hist_feat = True, hog_feat = True, sample_size=1000, color_space = 'HLS',
        model_name = "condition_4.pickle"))
    results.append(feature_vector_creation(
        spatial_feat=True, hist_feat = False, hog_feat = True, sample_size=1000, color_space = 'HLS',
        model_name = "condition_5.pickle"))
    results.append(feature_vector_creation(
        spatial_feat=False, hist_feat = False, hog_feat = True, sample_size=1000, color_space = 'HLS',
        model_name = "condition_6.pickle"))
    print(results)
    with open('result.txt', 'w') as f:
        for x in results:
            f.write(str(x) + "\n")
    
    # results.append(feature_vector_creation(
    #     spatial_feat=True, hist_feat = True, hog_feat = True, sample_size=6000, color_space = 'RGB',
    #     model_name = "best_condition.pickle"))