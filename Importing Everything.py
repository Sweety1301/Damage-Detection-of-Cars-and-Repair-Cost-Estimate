import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers

training_path = '/kaggle/input/car-dmage-data-v4/data3a/training'
testing_path = '/kaggle/input/car-dmage-data-v4/data3a/validation'

batch_size = 32
img_height = 224
img_width = 224

train_data_gen = ImageDataGenerator(rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        validation_split=0.20,) 

train_ds = train_data_gen.flow_from_directory(
    training_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='training', 
    seed=123,
    shuffle=True  
)
valid_ds = train_data_gen.flow_from_directory(
    training_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='validation', 
    seed=123,
    shuffle=True  
)

test_data_gen = ImageDataGenerator(rescale=1./255,)

test_ds = test_data_gen.flow_from_directory(
    testing_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
