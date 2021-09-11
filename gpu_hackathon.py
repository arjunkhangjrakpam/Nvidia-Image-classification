# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 23:57:46 2021

@author: arjun
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import tqdm
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import keras
import cv2
import os
import glob
import pickle

from keras.models import Sequential
from keras.layers import *
from keras import optimizers 
from numpy.random import seed
import json
seed(1)
import tensorflow as tf 

np.random.seed(100)
LEVEL = 'level_1'


os.chdir(r"C:/Users/arjun/CONTESTS/GPU_HACKATHON")
os.getcwd()

import json
import torchvision 
import torchvision.transforms as transforms
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class SigmoidNeuron:
  
  def __init__(self):
    self.w = None
    self.b = None
    
  def perceptron(self, x):
    return np.dot(x, self.w.T) + self.b
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def grad_w_mse(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred) * x
  
  def grad_b_mse(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred)
  
  def grad_w_ce(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    if y == 0:
      return y_pred * x
    elif y == 1:
      return -1 * (1 - y_pred) * x
    else:
      raise ValueError("y should be 0 or 1")
    
  def grad_b_ce(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    if y == 0:
      return y_pred 
    elif y == 1:
      return -1 * (1 - y_pred)
    else:
      raise ValueError("y should be 0 or 1")
  
  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):
    
    # initialise w, b
    if initialise:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0
      
    if display_loss:
      loss = {}
    
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
      dw = 0
      db = 0
      for x, y in zip(X, Y):
        if loss_fn == "mse":
          dw += self.grad_w_mse(x, y)
          db += self.grad_b_mse(x, y) 
        elif loss_fn == "ce":
          dw += self.grad_w_ce(x, y)
          db += self.grad_b_ce(x, y)
      self.w -= learning_rate * dw
      self.b -= learning_rate * db
      
      if display_loss:
        Y_pred = self.sigmoid(self.perceptron(X))
        if loss_fn == "mse":
          loss[i] = mean_squared_error(Y, Y_pred)
        elif loss_fn == "ce":
          loss[i] = log_loss(Y, Y_pred)
    
    if display_loss:
      plt.plot(loss.values())
      plt.xlabel('Epochs')
      if loss_fn == "mse":
        plt.ylabel('Mean Squared Error')
      elif loss_fn == "ce":
        plt.ylabel('Log Loss')
      plt.show()
      
  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.sigmoid(self.perceptron(x))
      Y_pred.append(y_pred)
    return np.array(Y_pred)

# preprocessing 
def preprocessing(X, y):
  X = np.array(X).reshape((-1,64,64,1))
  y = np.array(y).reshape((-1,1))
  return(X,y)

# CNN architecture
def cnn_arc(X_train,y_train,epochs = 10):
  model = Sequential()
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(64,64,1)))
  model.add(Conv2D(128, kernel_size=3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, kernel_size=3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(32, kernel_size=3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.25))

  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  print("Summary before fitting")
  print(model.summary())
  history = model.fit(X_train, y_train, epochs=epochs,validation_split = 0.2,batch_size=32,verbose="auto")
  print("Summary after fitting")
  print(model.summary())
  return (model,history)

# plot history of model
def plot_history(history):
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  print(plt.show())
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  print(plt.show())


def read_all(folder_path, key_prefix=""):
    '''
    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in tqdm.notebook.tqdm(enumerate(files), total=len(files)):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:-4]
        image = Image.open(file_path)
        image = image.convert("L")
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images


languages = ['hi']

images_train = read_all("C:/Users/arjun/CONTESTS/GPU_HACKATHON/train_test/train/background", key_prefix='bgr_') # change the path
for language in languages:
  images_train.update(read_all("C:/Users/arjun/CONTESTS/GPU_HACKATHON/train_test/train/"+language, key_prefix=language+"_" ))
print(len(images_train))

images_test = read_all("C:/Users/arjun/CONTESTS/GPU_HACKATHON/train_test/test/", key_prefix='') # change the path
print(len(images_test))



list(images_test.keys())[:5]


# In[ ]:


X_train = []
Y_train = []
for key, value in images_train.items():
    X_train.append(value)
    if key[:4] == "bgr_":
        Y_train.append(0)
    else:
        Y_train.append(1)

ID_test = []
X_test = []
for key, value in images_test.items():
      print(key)
      ID_test.append(int(key))
      X_test.append(value)
ID_test = sorted(ID_test)
ID_test_png = []
for i in ID_test:
      print(i)
      a = str(i)+".png"
      ID_test_png.append(a)
     

        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape)


# In[ ]:


scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

sn_mse = SigmoidNeuron()
sn_mse.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.015, loss_fn="mse", display_loss=True)


# In[ ]:


sn_ce = SigmoidNeuron()
sn_ce.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.010, loss_fn="ce", display_loss=True)


Xtrain,ytrain = preprocessing(X_train,Y_train)
cnn,history = cnn_arc(Xtrain,ytrain,epochs  = 8)
plot_history(history)

    
def print_accuracy(sn):
  Y_pred_train = sn.predict(X_scaled_train)
  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()
  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
  print("Train Accuracy : ", accuracy_train)
  print("-"*50)


# In[ ]:


print_accuracy(sn_mse)
print_accuracy(sn_ce)
print_accuracy(cnn)
Xtest,s = preprocessing(X_test,X_test)
Y_pred_test=cnn.predict(Xtest)
Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()
accuracy_train = accuracy_score(Y_pred_binarised_test, Y_train)
print("Train Accuracy : ", accuracy_train)
print("-"*50)
# ## Sample Submission

# In[ ]:


Y_pred_test = sn_ce.predict(X_scaled_test)
Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()

submission = {}
submission['ImageId'] = ID_test_png
submission['Class'] = Y_pred_binarised_test.astype("int")

submission = pd.DataFrame(submission)
submission = submission[['ImageId', 'Class']]



def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename):
    res = {}
    for i in range(submission.shape[0]):
        test_set = submission['ImageId'][i]
        res[test_set] = int(submission['Class'][i])
        print(type(res[test_set])) 
        
    write_json(filename, res)



if __name__ == '__main__':
    generate_sample_file('C:/Users/arjun/CONTESTS/GPU_HACKATHON/output_json/sample_result1.json')






