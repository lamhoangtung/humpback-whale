import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import *

# Param world
train_imgs = "./data/train/"
test_imgs = "./data/test/"
img_size = 224
batch_size = 64

train = pd.read_csv("./data/train.csv")
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())

train_ims, train_labels = normalize_images(train, img_size)

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

# Callback for this run
reduceLROnPlat = ReduceLROnPlateau(monitor='val_top_5_accuracy',
                                   factor=0.50,
                                   patience=3,
                                   verbose=1,
                                   mode='max',
                                   min_delta=.001,
                                   min_lr=1e-5
                                   )

earlystop = EarlyStopping(monitor='val_top_5_accuracy',
                          mode='max',
                          patience=5)

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          batch_size=batch_size, write_images=True)

weightpath = "/model/weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_loss', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)

callbacks = [earlystop, reduceLROnPlat, tensorboard, checkpoint]

model = create_resnet50(img_size=img_size, num_classes=num_classes)
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
batches = gen.flow(x_train, y_train, batch_size=batch_size)
val_batches = gen.flow(x_val, y_val, batch_size=batch_size)

epochs = 40
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=callbacks)
