import umap
import umap.plot
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread, imshow
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

dir= '/home/ganesh/Dataset'

categories=['Cloudy','Rainy','Sunrise','Sandstorm']

pick_in=open('data4_hog.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()


print(len(data1))
print(len(data1[0]))
print(len(data1[0][0]))



random.shuffle(data1)
features=[]
labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)
	
	
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(features)
feat=np.array(px,dtype=float)
lab=np.array(labels,dtype=int)
mapper = umap.UMAP(densmap=True,n_neighbors=50,min_dist=0.01,n_components=2).fit(feat)

umap.plot.points(mapper, labels=lab,theme='fire').legend(categories)
#umap.plot.connectivity(mapper, show_points=True)
#umap.plot.plt.legend(categories)
umap.plot.plt.show()
#plt.legend(categories)
#plt.show()
