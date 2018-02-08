import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

with open('best_condition.pickle', 'rb') as handle:
    model = pickle.load(handle)


