# import all necessary modeules
import pandas as pd
import numpy as np
import keras
import cv2
import os
import glob
from matplotlib import pyplot as plt
import pickle
from google.colab.patches import cv2_imshow
from keras.models import Sequential
from keras.layers import *
from keras import optimizers 
from numpy.random import seed
import json
seed(1)
import tensorflow as tf 


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

# Read Data from folders
def read_data(train_test = 0):
  Xtrain = []
  ytrain = []
  if train_test ==0:
    text_labels = "/content/drive/MyDrive/hackathon/train/training/hi"
    for filename in glob.glob(text_labels+"/*.jpg"):
      img = cv2.imread(filename,0)
      img = cv2.resize(img,(64,64))
      Xtrain.append(img)
      ytrain.append(1)

    non_text_labels = "/content/drive/MyDrive/hackathon/train/training/background"
    for filename in glob.glob(non_text_labels+"/*.jpg"):
      img = cv2.imread(filename,0)
      img = cv2.resize(img,(64,64))
      Xtrain.append(img)
      ytrain.append(0)
  else :
    test_path = '/content/drive/MyDrive/hackathon/test/test'
    for filename in glob.glob(test_path+"/*.jpg"):
      img = cv2.imread(filename,0)
      img = cv2.resize(img,(64,64))
      Xtrain.append(img)
    ytrain = pd.read_csv('/content/drive/MyDrive/hackathon/test/test/mannual_labeling.csv',header=None)
  return(Xtrain,ytrain)

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

# preprocessing 
def preprocessing(X, y):
  X = np.array(X).reshape((-1,64,64,1))
  y = np.array(y).reshape((-1,1))
  return(X,y)
# from utils.io import write_json
def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename,pred):
    res = {}
    for i in range(1,99):
        test_set = str(i) + '.png'
        res[test_set] = pred[i-1]

    write_json(filename, res)

if __name__ == '__main__':
  Xtrain,ytrain = read_data(0) # 0 is for train and 1 id for test
  Xtrain,ytrain = preprocessing(Xtrain,ytrain)
  model,history = cnn_arc(Xtrain,ytrain,epochs  = 10)
  plot_history(history)
  Xtrain,ytrain  = read_data(1)
  Xtrain,ytrain = preprocessing(Xtrain,ytrain)
  prediction  = model.predict(Xtrain)
  l =[]
  for i in prediction.flatten():
    if i > 0.5:
      l.append(1)
    else :
      l.append(0)
  generate_sample_file('./sample_result1.json',l)
