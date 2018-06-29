import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import functionset as myfc
from collections import Counter

"""
images = ['test1','test2','test3','test4','test5','test6','test7',
          'test8','test9','test10','test11','test12','test13','test14','test15','test16']
"""

"""for image creation, please activate the following lines"""
images = ['test4']
myfc.image_converter(images,svm_model_path='condition_5.pickle')

"""for video creation, please activate the following lines"""
# myfc.video_creation('project_video.mp4','project_video_w_pipline.avi','best_condition.pickle',50,0,True)
