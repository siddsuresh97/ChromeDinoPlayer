import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from os import listdir
import cv2
import time
import os
import Input
from pynput.keyboard import Controller
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dropout
from keras.models import load_model
keyboard = Controller()
from keras import backend as K
import tensorflow as tf

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def a():
    i_shape=(10,40,1)
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same', name='conv_1',
                     input_shape=i_shape))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    '''model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))'''
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def train():

    print ('Gathering Data...')
    data = []
    labels = []
    s = time.time()
    for n in range(1, 11):
        for d in range(0, len(listdir('GamePics/game{}/'.format(n)))):
            im = cv2.imread('GamePics/game{}/'.format(n) + str(d) + '.png', 0)
            im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
            im = im / 255.0
            data.append(im)

        nl = np.load('labels/game{}.npy'.format(n))
        labels = np.append(labels, nl)
        print ('labels/game{}.npy'.format(n), nl.shape)
    print (time.time() - s)
    data = np.array(data).reshape(-1, 10, 40, 1)
    labels = labels.reshape(-1, 1)
    #data=np.array(data)
    #labels=np.array(labels)

    #data = np.array(data).reshape(10, 40, 1)
    #labels = labels.reshape(1)

    print (labels.shape, data.shape)

    try:
        model = load_model('models/unsupervised/my_model.h5')
    except:
        print ('did not restore')
        #model.train(data,labels,50,100)
    s = time.time()
    model=a()
    model.fit(data,labels,50,100)

    model.save('models/unsupervised/my_model.h5')
    print (time.time() - s)

def run():
  time.sleep(1)
  model=load_model("models/unsupervised/my_model.h5")
  print ('Starting...')
  Input.Reset(keyboard)
  while 1:
      im = np.reshape(getScrn(), [1,10, 40, 1])
      a = np.round(model.predict(im))
      print(a)
      '''if a == [[0]]
          Input.Wait(keyboard)
      else:
          Input.Up(keyboard)
      print (a)'''
      if a == [[0]]:
          Input.Wait(keyboard)
      elif a==[[1]]:
          Input.Up(keyboard)
     ''' else:
          Input.Down(keyboard)'''
      print (a)



def getScrn():
  os.system("screencapture -R60,125,600,150 holder.png")
  im = cv2.imread('holder.png', 0)
  im = cv2.resize(im, (40, 10), interpolation=cv2.INTER_AREA)
  im = im / 255.0
  return im
