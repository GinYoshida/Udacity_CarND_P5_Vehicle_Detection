import cv2
import pickle
import glob
import time
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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

def data_list_creator():
    '''
    Function to create the training and test dataset, which are including file paths for images of cars and non-cars.
    :return: list of the file paths for images of cars and non-cars
    '''
    # Read in cars and notcars
    images_car_far = glob.glob('../image_data_Udacity_CarND_P5/vehicles/GTI_Far/*.png')
    images_car_left = glob.glob('../image_data_Udacity_CarND_P5/vehicles/GTI_Left/*.png')
    images_car_MiddleClose = glob.glob('../image_data_Udacity_CarND_P5/vehicles/GTI_MiddleClose/*.png')
    images_car_right = glob.glob('../image_data_Udacity_CarND_P5/vehicles/GTI_Right/*.png')
    images_car_KITTI_extracted = glob.glob('../image_data_Udacity_CarND_P5/vehicles/KITTI_extracted/*.png')
    images_noncar_Extras = glob.glob('../image_data_Udacity_CarND_P5/non-vehicles/Extras/*.png')
    images_noncar_GIT = glob.glob('../image_data_Udacity_CarND_P5/non-vehicles/GTI/*.png')
    cars = images_car_far + images_car_left + images_car_MiddleClose + images_car_right + images_car_KITTI_extracted
    notcars = images_noncar_Extras + images_noncar_GIT
    print('Number of cars:{}, Number of non-cars: {}'.format(len(cars),len(notcars)))

    # Shuffle original data
    random.shuffle(cars)  # length is 8792
    random.shuffle(notcars)  # length is 8968

    return cars,notcars

def training_mode_creation(
        spatial_feat, hist_feat, hog_feat,
        sample_size, color_space,
        model_name, train_test_split_rate
):
    '''
    Function to create svm model to detect car and non-car images.
    sample size and method to create feature vector from color space are optional.
    model will be saved as the name of "filename_car" with pickle module.
    score of verification will be returned.
    :param spatial_feat: Spatial features on or off
    :param hist_feat: Histogram features on or off
    :param hog_feat: HOG features on or off
    :param sample_size: sample size to create future vector
    :param color_space: color space for traning Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    :param filename_car: file name of training data
    :return: verification score of model
    '''

    # Read the training/test data
    cars, notcars = data_list_creator()

    # Create an array stack of feature vectors
    X = np.hstack((cars, notcars))

    # Define the labels vector
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

    # Split data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_rate, random_state=1)

    # Reduce the sample size
    if sample_size == None:
        pass
    else:
        X_train = X_train[0:sample_size]
        y_train = y_train[0:sample_size]

    # Define the parameter to compute the features
    orient = 9  # HOG orientations
    pix_per_cell = 5  # HOG pixels per cell
    cell_per_block = 3  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins

    # Compute the features of the training set
    X_train = extract_features(
        X_train, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat
    )

    # Compute the features of the test set
    X_test = extract_features(
        X_test, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat
    )

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scalar to the both of training data and test data
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

    # Apply the best C for the training.
    svc = LinearSVC(C=clf.best_params_['C'])
    svc.fit(X_train, y_train)
    score_model = round(svc.score(X_test, y_test),4)
    print('Test Accuracy of {} is {} '.format(model_name,score_model))

    # Dump the model data and the related parameters
    output_sum = {
        'model':svc,'color_space':color_space,'scaler':X_scaler,'orient':orient,'pix_per_cell':pix_per_cell,
        'cell_per_block':cell_per_block,'spatial_size':spatial_size,'hist_bins':hist_bins,
        'spatial_feat': spatial_feat, 'hist_feat': hist_feat, 'hog_feat': hog_feat,
    }

    with open(model_name, 'wb') as handle:
        pickle.dump(output_sum, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return score_model

if __name__ == "__main__":

    # Train the 7 cases to find the best case.
    # For HOG, grayscale is applied. Color space type doesn't affect the HOG result.

    results = []
    training_mode_creation(
        spatial_feat=True, hist_feat = True, hog_feat = True, sample_size=None, color_space = 'RGB',
        model_name = "condition_1.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=True, hist_feat = False, hog_feat = True, sample_size=None, color_space = 'RGB',
        model_name = "condition_2.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=False, hist_feat = True, hog_feat = True, sample_size=None, color_space = 'RGB',
        model_name = "condition_3.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=True, hist_feat = True, hog_feat = True, sample_size=None, color_space = 'HLS',
        model_name = "condition_4.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=True, hist_feat = False, hog_feat = True, sample_size=None, color_space = 'HLS',
        model_name = "condition_5.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=False, hist_feat = True, hog_feat = True, sample_size=None, color_space = 'HLS',
        model_name = "condition_6.pickle", train_test_split_rate= 0.01
    )

    training_mode_creation(
        spatial_feat=False, hist_feat = False, hog_feat = True, sample_size=None, color_space = 'RGB',
        model_name="condition_7.pickle", train_test_split_rate=0.01)
