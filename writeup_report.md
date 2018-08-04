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
[image4_2]: ./for_report/Fig1_2_1_HOG_example_1_def.png
[image5]: ./for_report/Fig1_2_2_HOG_example.png
[image6]: ./for_report/Fig1_2_3_Spatial_img.png
[image7]: ./for_report/Fig1_2_4_Histgram.png
[image8]: ./for_report/test1.png


[video1]: ./project_video.mp4

# 1. Training
## 1.1 Data preparation
 The process of data preparation is:
 * Step1: Download the data from the followin link.
 Link: 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip'
 Link: 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'
 * Step2: Remove a part of images files from "Non-vehicle" data set
 * Step3: Augment some Non-Vehicle data
 
These 3 processes were done with "enviroment_setup.py".

The training data was provided by Udacity.  
About step3, the images aurgued were selected because these type of images are difficult to be classified.
The example of each step is shown in the following figures.

__Fig1.1.1 Example of data set__  
![alt text][image1]

__Fig1.1.2 Example of removed image__  
<img src="./for_report/Fig1_1_2-rm.png" alt="image" width="128px"/>

__Fig1.1.3 Example of argumentation__  
<img src="./for_report/Fig1_1_3-ag.png" alt="image" width="128px"/>


 These processes were done with "enviroment_setup.py"


## 1.2 Feture extraction
### Description for HOG

The parameter turning was done to detect the difference of the object with my eyes.
Default parameters are:
 - Orient:9
 - pix_per_cell:8
 - cell_per_block:2

I tuned the HOG parameters with my subjective check by my eyes.
 It is hard to recognize the difference between vehicle and non-vehicle images because of less number of features.
Then, I increased the number of the feature by decreasing the pix_per_cell and increasing cell_per_block step by step.
Once I recognize the difference of HOG image between the vehicle and non-vehicle image, I stop to tune it.
Updated parameters are:
- Orient: 9
- pix_per_cell: 5
- cell_per_block: 3

__Fig1.2.1 HOG image example__  
Default  
![alt text][image4_2]  

Tuned  
![alt text][image4]

 Note: Need update the image with low resonance.  

 About the color space, the grayscale was used to compute the HOG features.
 Becuase, Grayscale represents the shape of the object same as a single channel of a color image. 
 Fig1.2.2 shows the comparison between HOG images with Grayscale and L channel of HLS.
 I assumed that it is equivalent and Grayscale is enough.


__Fig1.2.2 HOG image comparison__  
![alt text][image5]

### Description for Binned color feature

I selected 16 as the size of the bin of the color features.  
Then, I can recognize the vehicle and non-vehicle with my subjective check by my eyes.

__Fig1.2.3 Binned feature example__  
![alt text][image6]

### Description for Histogram
I selected 16 as the size of the bin of the Histgram features.  
Then, I can recognize the vehicle and non-vehicle with my subjective check by my eyes.

__Fig1.2.4 Histgram difference__  
![alt text][image7]


### Description the selection of feature

 The followin 7 cases were computed as SVM model.  
 Then, the performance was checked in all cases with the test image.  
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

__Fig1.2.5 Outcomes of test image__  
![alt text][image8]



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
