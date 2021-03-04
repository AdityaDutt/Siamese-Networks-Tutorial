import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils.vis_utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# Import color encoder which uses siamese networks
from model import train_color_encoder




## Prepare positive and negative pais of data samples


# Prepare data for different shapes but same colors

dir = os.getcwd() + "/shapes/"

images = []
y_col = []

for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        fullname = os.path.join(root, name)
        if fullname.find(".png") != -1 :
            images.append(fullname)
            if fullname.find("red") != -1 :
                y_col.append(0)
            elif fullname.find("blue") != -1 :
                y_col.append(1)
            else :
                y_col.append(2)

y_col = np.array(y_col)
images = np.array(images)



# Generate positive samples

red_im = images[np.where(y_col==0)]
green_im = images[np.where(y_col==1)]
blue_im = images[np.where(y_col==2)]

# Test images
test_red_im = red_im[50:]
test_green_im = green_im[50:]
test_blue_im = blue_im[50:]

# Read only 20 images from each class for training
red_im = red_im[:20]
green_im = green_im[:20]
blue_im = blue_im[:20]



positive_red = list(itertools.combinations(red_im, 2))

positive_blue = list(itertools.combinations(blue_im, 2))

positive_green = list(itertools.combinations(green_im, 2))


# Generate negative samples

negative1 = itertools.product(red_im,green_im)
negative1 = list(negative1)

negative2 = itertools.product(green_im,blue_im)
negative2 = list(negative2)

negative3 = itertools.product(red_im,blue_im)
negative3 = list(negative3)


# Create pairs of images and set target label for them. Target output is 1 if pair of images have same color else it is 0.
color_X1 = []
color_X2 = []
color_y = []
positive_samples = positive_blue + positive_green + positive_red
negative_samples = negative1 + negative2 + negative3

for fname in positive_samples :
    im = cv2.imread(fname[0])
    color_X1.append(im)
    im = cv2.imread(fname[1])
    color_X2.append(im)
    color_y.append(1)

for fname in negative_samples :
    im = cv2.imread(fname[0])
    color_X1.append(im)
    im = cv2.imread(fname[1])
    color_X2.append(im)
    color_y.append(0)


color_y = np.array(color_y)
color_X1 = np.array(color_X1)
color_X2 = np.array(color_X2)
color_X1 = color_X1.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))
color_X2 = color_X2.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))

color_X1 = 1 - color_X1/255
color_X2 = 1 - color_X2/255

print("Color data : ", color_X1.shape, color_X2.shape, color_y.shape)

train_color_encoder(color_X1, color_X2, color_y)
