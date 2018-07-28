This is repository for write up report for Project 4 of Udacity Nano Degree. In this readme file, overview of this repository is shown.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, it is possible to apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Struction of this repository

The following resources can be found in this github repository: 
### Files
* enviroment_setup.py 
* main_training.py 
* main.py
* functionset.py 
* project_video.mp4 
* project_video_w_pipline.avi 
* Code_test.ipynb
* condition_1.pickle ... condition_7.pickle
* nohup.out
* RM_files_non-vehicle.csv
* non-vehicle_Arg.csv
* test_video.mp4
* writeup_report.mp

### Directory
* test_images 
* output_images
* for_report

## Details about files in this repository
### 'enviroment_setup.py'
 Python file to set up the image datas into "../image_data_Udacity_CarND_P5".

### 'main_training.py'
 Python file to create the training model to judge whether the images includes the vehicle or not.
 7 different models will be calculated for the parameter study.
 
### 'main.py'
 Python file to apply the pipe-line, which draws the rectangle on the area judged as vehicle, to the following items:
  * Images files saved into test_images 
  (For images, the heat maps and original file will be added as diagram.)
  * "./project_video.mp4"
 
### 'functionset.py'
 Python file includes the necessary functions for "main_training.py" and "main.py".

### 'project_video.mp4'
 This is the Original .mp4 file.
 The pipeline was applied to this mp4 file and "project_video_w_pipline.mp4" was created.
 
### 'project_video_w_pipline.mp4'
 .mp4 file with the rectangle drawn on the area judged as the vehicle.  
 This is the outcome of the main.py.

### 'Code_test.ipynb'
 Jupyter notebook file.
 This file was used to create the necessary images for "Writeup_report.md".

### condition_1.pickle ... condition_7.pickle
 svm model files in each condition.
 They are the outcome of the 'main_training.py'

### nohup.out
 The record during the execution of 'main.py' and 'main_training.py'.

### RM_files_non-vehicle.csv
 Used for 'enviroment_setup.py'.
 This file includes the file name, which need be removed from the data set, because it includes the vehicle image partially.
  
### non-vehicle_Arg.csv
 Used for "'enviroment_setup.py'"
 This file includes the file name of the image, including the shadows and complicated structure.
 These images will be used to argument non-vehicle data set instead of the removed data. 

### test_video.mp4
 short mp4 file for trial of the pipeline. 

### writeup_report.mp
 Writeup report of this project.
 
# Usage of this repository
 Please execute the following files.
 * Step1: 'enviroment_setup.py'
 * Step2: 'main_training.py'
 * Step3: 'main.py'
 
 Note:  
  For Code_test.ipynb, please execute the following command to set the proper working directory.   
  jupyter notebook --notebook-dir= "your working directory" 
