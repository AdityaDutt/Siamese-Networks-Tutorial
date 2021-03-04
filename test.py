from mpl_toolkits.mplot3d import Axes3D
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np
import sklearn, pandas as pd, seaborn as sn
from keras.models import Model, load_model, Sequential
from keras import backend as K
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


model = load_model(os.getcwd()+"/color_encoder.h5")
siamese_model = load_model(os.getcwd()+"/color_siamese_model.h5")


names = list(test_red_im) + list(test_blue_im) + list(test_green_im)# + list(test_cyan_im) #+ list(test_yellow_im)

names1 = [x for x in names if 'red' in x]
names2 = [x for x in names if 'blue' in x]
names3 = [x for x in names if 'green' in x]

test_im = []
for i in range(len(names)) :
    test_im.append(cv2.imread(names[i]))

r,c,_ = test_im[0].shape
test_im = np.array(test_im)
test_im = test_im.reshape((len(test_im), r,c,3))
names = [x.split("/")[-1] for x in names]

test_im = 1 - test_im/255

pred = model.predict(test_im)

num = int(pred.shape[0]/3)
colors = ['red', 'blue', 'green']
y = [colors[0] for i in range(num)]
y += [colors[1] for i in range(num)]
y += [colors[2] for i in range(num)]

feat1 = pred[:,0]
feat2 = pred[:,2]
feat3 = pred[:,1]


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(feat1, feat2, feat3, c=y, marker='.')
plt.show()
