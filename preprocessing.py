####Importing necessary libraries####
import cv2
import os
from tqdm import tqdm
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit as s_split

####Applying CLAHE for enhancement and Gaussian Blurring for denoising####
def process_data(path):
    X=[]
    y=[]
    for nextDir in os.listdir(path):
        if not nextDir.startswith('.'):
            label = 3
            if nextDir in ["BCC"]:
                label = 0
            elif nextDir in ["SCC"]:
                label = 1
            elif nextDir in ["Benign"]:
                label = 2
            else:
                label = 3
            temp = path + "/" + nextDir
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + "/" + file)
                if img is not None:
                    img = cv2.GaussianBlur(img, (5,5), 1)
                    im0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    planes=cv2.split(im0)
                    planes[0]=clahe.apply(planes[0])
                    im0 = cv2.merge(planes)
                    im1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    im1 = clahe.apply(im1)
                    im_ = cv2.merge([im0,im1])
                    X.append(im_)
                    y.append(label)
    return X,y
	
####Shuffling the data####
def shuffle_data(X_shu, y_shu):
    X_shu=np.asarray(X_shu)
    y_shu=np.asarray(y_shu)
    split = s_split(n_splits= 1, test_size = 0.20, random_state=18)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_id, test_id in split.split(X_shu, y_shu):
        X_train.append(X_shu[train_id])
        y_train.append(y_shu[train_id])
        X_test.append(X_shu[test_id])
        y_test.append(y_shu[test_id])
    del X_shu
    del y_shu
    return X_train, y_train, X_test, y_test
	
####Returning the processed data####
def get_data(path):
	X, y = process_data(path)
	X_train, y_train, X_test, y_test = shuffle_data(X,y)
	X_train = np.asarray(X_train, dtype = "float32")[0]
	X_test = np.asarray(X_test, dtype = "float32")[0]
	y_train = np.asarray(y_train)[0]
	y_test = np.asarray(y_test)[0]

	X_train /= 255
	X_test /= 255

	y_train = to_categorical(y_train, 3)
	y_test = to_categorical(y_test, 3)
	return [X_train, y_train, X_test, y_test]