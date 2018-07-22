import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import functionset as myfc
from collections import Counter

"""for image creation, please activate the following lines"""
# images = ['test1','test2','test3','test4','test5','test6','test7',
#           'test8','test9','test10','test11','test12']
# myfc.image_converter(images,svm_model_path='condition_4.pickle')

"""for video creation, please activate the following lines"""
myfc.video_creation('project_video.mp4','project_video_w_pipline.avi','condition_4.pickle',42,41,True)
