**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./for_report/Fig1_1_1-Vehicle_and_non_vehicle.png
[image4]: ./for_report/Fig1_2_1_HOG_example_1.png
[image5]: ./for_report/Fig1_2_2_HOG_example.png
[image6]: ./for_report/Fig1_2_3_Spatial_img.png
[image6]: ./for_report/Fig1_2_4_xxxx.png

[video1]: ./project_video.mp4

# 1. Training
## 1.1 Data preparation
 The process of data preparation is:
 * Step1: Download the data from the followin link
 * Step2: Remove a part of images files from "Non-vehicle" data set
 * Step3: Argument some Non-Vehicle data
 
These 3 processes were done with "".

The traiing data was provided by Udacity.
These images were selected because these type of images are difficult to be classified.
These images were selected because these type of images are difficult to be classified.
These images were selected because these type of images are difficult to be classified.

These images were selected because these type of images are difficult to be classified. 
 
__Fig1.1.1 Example of data set__  
![alt text][image1]

__Fig1.1.2 Example of removed image__  
<img src="./for_report/Fig1_1_2-rm.png" alt="image" width="128px"/>

__Fig1.1.3 Example of argumentation__  
<img src="./for_report/Fig1_1_3-ag.png" alt="image" width="128px"/>


 These process was done with "enviroment_setup.py"


## 1.2 Feture extraction
### Description for HOG

The parameter turning was done with the checking by my eyes.
Default parameters are:

But it is hard to recognize the difference between vehicle and non-vehicle images.
Then, I increase the number of XXXX step by step.
Once I recognize the difference of HOG image between vehicle and non-vehicle image, I stop to tune it.
The purpose of HOG feature should be to shape of the object. Therefor, I tuned the parameters which allow me to recognize the shape difference.

__Fig1.2.1 HOG image example__  
![alt text][image4]

 Note: Need update the image with low resonance.  

 The grayscale was used to compute the HOG features.
 Becuase, Grayscale represents the shape of the object same as a single channel of a color image. 
 Fig1.2.2 shows the comparsion between HOG images with Grayscale and L channel of HLS.
 I assumed that it is equivalant and Grayscale is enough.


__Fig1.2.2 HOG image comparison__  
![alt text][image5]

### Description for Binned color feature

The default parameter to compute the binned colored features is 12.
With this size, it is hard to recognize the difference.


__Fig1.2.3 Binned feature example__  
![alt text][image6]


### Description for Histogram

__Fig1.2.1 HOG image example__  
![alt text][image7]


### Description the selection of feature

 The followin 7 cases were computed as SVM model.  
 Then, the performance was checked in all cases with the test image of "test1".  
 The best performance was confirmed with the condition4.pickle and it was selected.  

| Output file name | Color space | HOG | Spatial | Histogram | Validation score |
|:----------------:|:-----------:|:---:|:-------:|:---------:|:----------------:|
|condition1.pickle | RGB         | With| With    | with      |            0.9831|
|condition2.pickle | RGB         | With| With    | without   |            0.9774|
|condition3.pickle | RGB         | With| Without | with      |            0.9661|
|condition4.pickle | HLS         | With| With    | with      |            0.9661|
|condition5.pickle | HLS         | With| With    | without   |            0.9831|
|condition6.pickle | HLS         | With| Without | with      |            0.9718|
|condition7.pickle | No          | With| Without | without   |            0.9435|


# 2. Pipeline for single image
## Basic flow
 Flow
 HOG image on the whole image
 Apply feature extraction for each subimage

## False positive
 Base threshold and dynamic threshould


# 3. Pipeline for video
## Adding previour image


# 4. Discussion
 Still
