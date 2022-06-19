import os
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import normalize,StandardScaler

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from sklearn.decomposition import PCA

dir='home/ganesh/Dataset/Cloudy'

data=[]

# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()
scaler = transforms.Resize(size=(224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name) 
    img = img.convert("RGB")  
    #print(img.getbands())
    #img.show()
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1,512,1,1)    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)    
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)    
    # 6. Run the model on our transformed image
    model(t_img)    
    # 7. Detach our copy function from the layer
    h.remove()    
    # 8. Return the feature vector
    return my_embedding

'''
for category in categories:
	path= os.path.join(dir, category)
	label=categories.index(category)
	
	for image in os.listdir(path):
			image_path=os.path.join(path,image)
			image_vector=get_vector(image_path)
			image_list_vector=image_vector.tolist()
			image_numpy_vector=np.array(image_list_vector,dtype=object)
			#print(image_numpy_vector.shape)
			row=image_numpy_vector.shape[0];
			col=image_numpy_vector.shape[1];
			#print(row,col)
			image_new=np.reshape(image_numpy_vector,(row*col));
			image_new1=image_new.astype('float')
			data.append([image_new1,label])
		
'''
path= os.path.join(dir, category)
label=categories.index(category)	
image_path=os.path.join(path,image)
image_vector=get_vector(image_path)
image_list_vector=image_vector.tolist()
image_numpy_vector=np.array(image_list_vector,dtype=object)
#print(image_numpy_vector.shape)
row=image_numpy_vector.shape[0];
col=image_numpy_vector.shape[1];
#print(row,col)
image_new=np.reshape(image_numpy_vector,(row*col));
image_new1=image_new.astype('float')

	
print(len(data))		

pick_in=open('resnet18_1125.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

# Load the model from the file
svm_from_joblib = joblib.load('svm_model.pkl')
 
# Use the loaded model to make predictions
svm_from_joblib.predict(X_test)
