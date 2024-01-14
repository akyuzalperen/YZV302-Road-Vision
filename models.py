import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
import os
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models





def cnn(num_classes):
  model = Sequential()

# 3 convolutional layers + maxpooling
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
  model.add(MaxPooling2D(2, 2))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Flatten())

  model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))
  model.add(Dropout(0.5))

  model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))
  model.add(Dropout(0.5))

  model.add(Dense(num_classes, activation='softmax'))

    #compile

  model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

  return model


def pretrainedcnn(base_model):

  #freeze the convolutional layers
  for layer in base_model.layers:
      layer.trainable = False

  #creating a new model on top
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())

  #adding custom fully connected layers
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))




  num_classes = 3
  model.add(Dense(num_classes, activation='softmax'))

  #Compile the model
  custom_optimizer = Adam(learning_rate=1e-5)
  model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
  return model





def firstcnn():
  #creating model and layers to train
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), input_shape=(500, 500, 3), activation="relu"))
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(64, (3, 3), activation="relu"))
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(128, (3, 3), activation="relu"))
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(128, (3, 3), activation="relu"))
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation="relu"))
  model.add(layers.Dense(1, activation="sigmoid"))  # Binary classification, so use sigmoid activation

  #compiling the model and setting parameters
  model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

  #setting Checkpoint to save the best model
  
  return model