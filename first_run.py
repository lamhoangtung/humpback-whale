import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.optimizers import Adam
from tqdm import tqdm

from common import *
from config import *

img_size = 224
batch_size = 256

batches, val_batches, class_weights = get_data_generator(img_size, batch_size)

model = create_resnet50(num_classes=num_classes)

# Train with freeze
for layer in model.layers[:-9]:
    layer.trainable = False
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weight_path = "model/first-run-freeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 20
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path),
                              class_weight=class_weights)

# Unfreeze
for layer in model.layers[-9:]:
    layer.trainable = True
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weight_path = "model/first-run-unfreeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 40
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path),
                              class_weight=class_weights)

"""
SECOND STAGE
"""

img_size = 448
batch_size = 32

batches, val_batches, class_weights = get_data_generator(img_size, batch_size)

# Train with freeze
for layer in model.layers[:-9]:
    layer.trainable = False
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weight_path = "model/second-run-freeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 20
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path),
                              class_weight=class_weights)

# Unfreeze
for layer in model.layers[-9:]:
    layer.trainable = True
model.compile(optimizer=Adam(lr=.005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

weightpath = "model/second-run-unfreeze-weights-{epoch:03d}-{top_5_accuracy:.3f}.hdf5"
epochs = 40
history = model.fit_generator(generator=batches,
                              steps_per_epoch=batches.n//batch_size,
                              epochs=epochs,
                              validation_data=val_batches,
                              validation_steps=val_batches.n//batch_size,
                              callbacks=get_common_callback(batch_size, weight_path),
                              class_weight=class_weights)
