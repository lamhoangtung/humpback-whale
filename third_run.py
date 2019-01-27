import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import *
from config import *


img_size = 448
batch_size = 16

train = pd.read_csv(oversampled_csv)
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())

train_ims, train_labels = preprocess_data(train, img_size)

x_train, x_val, y_train, y_val = train_test_split(train_ims,
                                                  train_labels,
                                                  test_size=0.10,
                                                  random_state=42
                                                  )

# print(x_train.shape)
# print(x_val.shape)
# print(y_train.shape)
# print(y_val.shape)

# Generator with augmentation
gen = ImageDataGenerator(zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=False,
                         rotation_range=10,
                         brightness_range=(0, 0.2),
                         shear_range=15
                         )
batches = gen.flow(x_train, y_train, batch_size=batch_size)
val_batches = gen.flow(x_val, y_val, batch_size=batch_size)

model = create_resnet50(img_size=img_size, num_classes=num_classes)
model = load_model('path_to_checkpoint', custom_objects={
                   'top_5_accuracy': top_5_accuracy})

# Train with freeze
for layer in model.layers[:-9]:
    layer.trainable = False
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weight_path = "/model/third-run-freeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 2
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path))

# Unfreeze
for layer in model.layers[-9:]:
    layer.trainable = True
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weightpath = "/model/third-run-unfreeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 3
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path))
